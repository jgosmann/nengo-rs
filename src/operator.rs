mod copy;
mod elementwise_inc;
mod reset;
mod time_update;

pub use crate::operator::copy::*;
pub use crate::operator::elementwise_inc::*;
pub use crate::operator::reset::*;
pub use crate::operator::time_update::*;
use futures::future::{BoxFuture, FutureExt, Shared};
use futures::stream::StreamExt;
use futures::stream::{FuturesOrdered, FuturesUnordered};
use std::sync::Arc;

pub trait Operator {
    fn step(&self);
}

pub struct OperatorNode {
    pub operator: Box<dyn Operator + Sync + Send>,
    pub dependencies: Vec<usize>,
}

pub async fn run_operators(nodes: &Vec<Arc<OperatorNode>>) {
    let mut tasks: Vec<Shared<BoxFuture<'_, ()>>> = Vec::with_capacity(nodes.len());
    for node in nodes.iter() {
        let dependencies = node
            .dependencies
            .iter()
            .map(|i| Shared::clone(tasks.get(*i).unwrap()))
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
