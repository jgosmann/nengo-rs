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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::signal::Signal;

    #[test]
    fn it_increments_the_steps() {
        let op = TimeUpdate::<f64, u64> {
            dt: 0.001,
            step_target: Arc::new(ScalarSignal::new("step_target".to_string(), 0)),
            time_target: Arc::new(ScalarSignal::new("time_target".to_string(), 0.)),
        };
        op.step_target.reset();
        op.time_target.reset();

        for _ in 0..3 {
            op.step();
        }

        assert_eq!(*op.step_target.read(), 3);
    }

    #[test]
    fn it_increments_the_time() {
        let op = TimeUpdate::<f64, u64> {
            dt: 0.001,
            step_target: Arc::new(ScalarSignal::new("step_target".to_string(), 0)),
            time_target: Arc::new(ScalarSignal::new("time_target".to_string(), 0.)),
        };
        op.step_target.reset();
        op.time_target.reset();

        for _ in 0..3 {
            op.step();
        }

        assert_eq!(*op.time_target.read(), 3. * op.dt);
    }
}
