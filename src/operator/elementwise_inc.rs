use crate::operator::Operator;
use crate::signal::{ArraySignal, SignalAccess};
use core::ops::{AddAssign, Mul};
use numpy::TypeNum;
use std::sync::Arc;

pub struct ElementwiseInc<T>
where
    T: TypeNum,
{
    pub target: Arc<ArraySignal<T>>,
    pub left: Arc<ArraySignal<T>>,
    pub right: Arc<ArraySignal<T>>,
}

impl<T> Operator for ElementwiseInc<T>
where
    T: TypeNum + Mul<T, Output = T> + AddAssign<T>,
{
    fn step(&self) {
        let left = &(*self.left.read());
        let right = &(*self.right.read());
        let mut target = self.target.write();
        *target += &(left * right);
    }
}
