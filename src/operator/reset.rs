use crate::operator::Operator;
use crate::signal::{ArraySignal, Get, ScalarSignal, Signal};
use ndarray::ArrayD;
use numpy::TypeNum;
use std::fmt::Debug;
use std::sync::Arc;

pub struct Reset<T, S>
where
    S: Signal,
{
    pub value: T,
    pub target: Arc<S>,
}

impl<T: TypeNum + Send + Sync + 'static> Operator for Reset<ArrayD<T>, ArraySignal<T>> {
    fn step(&self) {
        self.target.get_mut().assign(&self.value);
    }
}

impl<T: Send + Sync + Copy + Debug + 'static> Operator for Reset<T, ScalarSignal<T>> {
    fn step(&self) {
        *self.target.get_mut() = self.value;
    }
}
