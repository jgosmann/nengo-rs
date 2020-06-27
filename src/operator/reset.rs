use crate::operator::Operator;
use crate::signal::{ArraySignal, Get, ScalarSignal, Signal};
use ndarray::ArrayD;
use numpy::TypeNum;
use std::rc::Rc;

pub struct Reset<T, S>
where
    S: Signal,
{
    pub value: T,
    pub target: Rc<S>,
}

impl<T: TypeNum + 'static> Operator for Reset<ArrayD<T>, ArraySignal<T>> {
    fn step(&self) {
        self.target.get_mut().assign(&self.value);
    }
}

impl<T: Copy + 'static> Operator for Reset<T, ScalarSignal<T>> {
    fn step(&self) {
        *self.target.get_mut() = self.value;
    }
}
