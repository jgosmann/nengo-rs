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
        Self::run_operators(operators).await;
        Self::run_probes(probes).await;
    }

    async fn run_operators(nodes: Vec<Arc<OperatorNode>>) {
        let mut tasks: Vec<Shared<BoxFuture<'_, ()>>> = Vec::with_capacity(nodes.len());
        for node in nodes.iter() {
            let dependencies = node
                .dependencies
                .iter()
                .map(|i| Shared::clone(&tasks[*i]))
                .collect::<FuturesUnordered<_>>();
            tasks.push(
                Self::create_operator_future(&(*node.operator), dependencies)
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

    async fn run_probes(probes: Vec<Arc<RwLock<dyn Probe + Send + Sync>>>) {
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

    async fn create_operator_future(
        operator: &(dyn Operator + Send + Sync),
        dependencies: FuturesUnordered<Shared<BoxFuture<'_, ()>>>,
    ) {
        dependencies.collect::<()>().await;
        operator.step();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::signal::AnySignal;
    use ndarray::Ix;
    use std::any::Any;

    #[derive(Debug)]
    struct FakeSignal {
        name: String,
        num_reset_calls: RwLock<u32>,
    }

    impl FakeSignal {
        fn new(name: String) -> Self {
            Self {
                name,
                num_reset_calls: RwLock::new(0),
            }
        }
    }

    impl Signal for FakeSignal {
        fn as_any(&self) -> &dyn Any {
            self
        }

        fn as_any_arc(self: Arc<Self>) -> Arc<AnySignal> {
            self
        }

        fn name(&self) -> &String {
            &self.name
        }

        fn shape(&self) -> &[Ix] {
            &[]
        }

        fn reset(&self) {
            *self.num_reset_calls.write().unwrap() += 1;
        }
    }

    struct FakeOperator {
        call_counter: Arc<RwLock<u32>>,
        call_indices: Arc<RwLock<Vec<u32>>>,
    }

    impl FakeOperator {
        fn new(call_counter: Arc<RwLock<u32>>) -> (Self, Arc<RwLock<Vec<u32>>>) {
            let call_indices = Arc::new(RwLock::new(vec![]));
            (
                Self {
                    call_counter,
                    call_indices: Arc::clone(&call_indices),
                },
                call_indices,
            )
        }
    }

    impl Operator for FakeOperator {
        fn step(&self) {
            self.call_indices
                .write()
                .unwrap()
                .push(*self.call_counter.read().unwrap());
            *self.call_counter.write().unwrap() += 1;
        }
    }

    struct FakeProbe {
        call_counter: Arc<RwLock<u32>>,
        call_indices: Vec<u32>,
    }

    impl FakeProbe {
        fn new(call_counter: Arc<RwLock<u32>>) -> Self {
            Self {
                call_counter,
                call_indices: vec![],
            }
        }
    }

    impl Probe for FakeProbe {
        fn as_any(&self) -> &dyn Any {
            self
        }

        fn probe(&mut self) {
            self.call_indices.push(*self.call_counter.read().unwrap());
            *self.call_counter.write().unwrap() += 1;
        }
    }

    #[test]
    fn engine_steps_operator_before_probing() {
        let call_counter = Arc::new(RwLock::new(0));
        let (fake_operator, op_call_indices) = FakeOperator::new(Arc::clone(&call_counter));
        let operator_node = Arc::new(OperatorNode {
            operator: Box::new(fake_operator),
            dependencies: vec![],
        });
        let probe = Arc::new(RwLock::new(FakeProbe::new(Arc::clone(&call_counter))));
        let engine = Engine::new(
            vec![],
            vec![Arc::clone(&operator_node)],
            vec![Arc::clone(&probe) as Arc<_>],
        );

        engine.run_step();

        assert_eq!(*op_call_indices.read().unwrap(), vec![0]);
        assert_eq!(probe.read().unwrap().call_indices, vec![1]);
    }

    #[test]
    fn engine_runs_dependencies_first() {
        let call_counter = Arc::new(RwLock::new(0));
        let (fake_dependency, dependency_call_indices) =
            FakeOperator::new(Arc::clone(&call_counter));
        let (fake_dependent, dependent_call_indices) = FakeOperator::new(Arc::clone(&call_counter));
        let operators = vec![
            Arc::new(OperatorNode {
                operator: Box::new(fake_dependency),
                dependencies: vec![],
            }),
            Arc::new(OperatorNode {
                operator: Box::new(fake_dependent),
                dependencies: vec![0],
            }),
        ];
        let engine = Engine::new(vec![], operators, vec![]);

        engine.run_step();

        assert_eq!(*dependency_call_indices.read().unwrap(), vec![0]);
        assert_eq!(*dependent_call_indices.read().unwrap(), vec![1]);
    }

    #[test]
    fn engine_run_steps_runs_multiple_steps() {
        let call_counter = Arc::new(RwLock::new(0));
        let (fake_operator, op_call_indices) = FakeOperator::new(Arc::clone(&call_counter));
        let operator_node = Arc::new(OperatorNode {
            operator: Box::new(fake_operator),
            dependencies: vec![],
        });
        let probe = Arc::new(RwLock::new(FakeProbe::new(Arc::clone(&call_counter))));
        let engine = Engine::new(
            vec![],
            vec![Arc::clone(&operator_node)],
            vec![Arc::clone(&probe) as Arc<_>],
        );

        engine.run_steps(3);

        assert_eq!(*op_call_indices.read().unwrap(), vec![0, 2, 4]);
        assert_eq!(probe.read().unwrap().call_indices, vec![1, 3, 5]);
    }

    #[test]
    fn engine_reset_resets_all_signals() {
        let signals = vec![
            Arc::new(FakeSignal::new("s1".to_string())),
            Arc::new(FakeSignal::new("s2".to_string())),
        ];
        let engine = Engine::new(
            signals.iter().map(|s| Arc::clone(s) as Arc<_>).collect(),
            vec![],
            vec![],
        );

        engine.reset();

        for signal in signals.iter() {
            assert_eq!(*signal.num_reset_calls.read().unwrap(), 1);
        }
    }
}
