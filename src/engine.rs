use crate::operator::{Operator, OperatorNode};
use crate::probe::Probe;
use crate::signal::Signal;
use crate::sync::Event;
use futures::executor::ThreadPool;
use futures::future::{BoxFuture, Future, FutureExt, Shared};
use futures::stream::{FuturesOrdered, FuturesUnordered, StreamExt};
use std::sync::{Arc, RwLock};

pub struct Engine {
    signals: Vec<Arc<dyn Signal + Send + Sync>>,
    operators: Vec<Arc<OperatorNode>>,
    probes: Vec<Arc<RwLock<dyn Probe + Send + Sync>>>,
    thread_pool: ThreadPool,
    is_done: Arc<Event>,
}

impl Engine {
    pub fn new(
        signals: Vec<Arc<dyn Signal + Send + Sync>>,
        operators: Vec<Arc<OperatorNode>>,
        probes: Vec<Arc<RwLock<dyn Probe + Send + Sync>>>,
    ) -> Self {
        Self {
            signals,
            operators,
            probes,
            thread_pool: ThreadPool::new().unwrap(),
            is_done: Arc::new(Event::new()),
        }
    }

    pub fn run_step(&self) {
        self.run_threaded(Self::run_step_async(
            self.operators.clone(),
            self.probes.clone(),
        ));
    }

    pub fn run_steps(&self, n_steps: i64) {
        for _ in 0..n_steps {
            self.run_step();
        }
    }

    pub fn reset(&self) {
        self.signals.iter().for_each(|s| s.reset());
    }

    fn run_threaded<Fut: Future<Output = ()> + Send + 'static>(&self, fut: Fut) {
        self.is_done.clear();
        self.thread_pool
            .spawn_ok(Self::notify_when_done(fut, Arc::clone(&self.is_done)));
        self.is_done.wait();
    }

    async fn notify_when_done<Fut: Future<Output = ()> + Send + 'static>(
        fut: Fut,
        is_done: Arc<Event>,
    ) {
        fut.await;
        is_done.set();
    }

    async fn run_step_async(
        operators: Vec<Arc<OperatorNode>>,
        probes: Vec<Arc<RwLock<dyn Probe + Send + Sync>>>,
    ) {
        run_operators(&operators).await;
        probes
            .iter()
            .map(Self::probe_async)
            .collect::<FuturesUnordered<_>>()
            .collect::<()>()
            .await;
    }

    async fn probe_async(probe: &Arc<RwLock<dyn Probe + Send + Sync>>) {
        probe.write().unwrap().probe();
    }
}

pub async fn run_operators(nodes: &Vec<Arc<OperatorNode>>) {
    let mut tasks: Vec<Shared<BoxFuture<'_, ()>>> = Vec::with_capacity(nodes.len());
    for node in nodes.iter() {
        let dependencies = node
            .dependencies
            .iter()
            .map(|i| Shared::clone(&tasks[*i]))
            .collect::<FuturesUnordered<_>>();
        tasks.push(
            create_future(&(*node.operator), dependencies)
                .boxed()
                .shared(),
        );
    }
    tasks
        .iter()
        .map(|f| Shared::clone(f))
        .collect::<FuturesOrdered<_>>()
        .collect::<()>()
        .await;
}

async fn create_future(
    operator: &(dyn Operator + Send + Sync),
    dependencies: FuturesUnordered<Shared<BoxFuture<'_, ()>>>,
) {
    dependencies.collect::<()>().await;
    operator.step();
}
