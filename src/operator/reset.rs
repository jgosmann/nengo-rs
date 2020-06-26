use crate::operator::Operator;
use crate::signal::{ArraySignal, ScalarSignal, Signal};
use ndarray::ArrayD;
use numpy::TypeNum;
use std::rc::Rc;

pub struct Reset<T, S>
where
    S: Signal<T>,
{
    pub value: T,
    pub target: Rc<S>,
}

impl<T: TypeNum> Operator for Reset<ArrayD<T>, ArraySignal<T>> {
    fn step(&self) {
        self.target.get_mut().assign(&self.value);
    }
}

impl<T> Operator for Reset<T, ScalarSignal<T>> {
    fn step(&self) {
        *self.target.get_mut() = self.value;
    }
}
