use crate::operator::Operator;
use crate::signal::{ArraySignal, Get, ScalarSignal};
use ndarray::ArrayD;
use numpy::TypeNum;
use std::marker::PhantomData;
use std::rc::Rc;

pub struct CopyOp<T, S> {
    pub src: Rc<S>,
    pub dst: Rc<S>,
    pub data_type: PhantomData<T>,
}

impl<T: TypeNum> Operator for CopyOp<ArrayD<T>, ArraySignal<T>> {
    fn step(&self) {
        self.dst.get_mut().assign(&self.src.get());
    }
}

impl<T> Operator for CopyOp<T, ScalarSignal<T>> {
    fn step(&self) {
        *self.dst.get_mut() = *self.src.get();
    }
}
