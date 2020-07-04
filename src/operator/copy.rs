use crate::operator::Operator;
use crate::signal::{ArraySignal, Get, ScalarSignal};
use ndarray::ArrayD;
use numpy::TypeNum;
use std::marker::PhantomData;
use std::sync::Arc;

pub struct CopyOp<T, S> {
    pub src: Arc<S>,
    pub dst: Arc<S>,
    pub data_type: PhantomData<T>,
}

impl<T: TypeNum> Operator for CopyOp<ArrayD<T>, ArraySignal<T>> {
    fn step(&self) {
        self.dst.get_mut().assign(&self.src.get());
    }
}

impl<T: Copy> Operator for CopyOp<T, ScalarSignal<T>> {
    fn step(&self) {
        *self.dst.get_mut() = *self.src.get();
    }
}
