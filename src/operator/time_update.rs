use crate::operator::Operator;
use crate::signal::{ScalarSignal, SignalAccess};
use std::sync::Arc;

pub struct TimeUpdate<T, S> {
    pub dt: T,
    pub step_target: Arc<ScalarSignal<S>>,
    pub time_target: Arc<ScalarSignal<T>>,
}

impl Operator for TimeUpdate<f64, u64> {
    fn step(&self) {
        *self.step_target.write() += 1;
        *self.time_target.write() = *self.step_target.read() as f64 * self.dt;
    }
}
