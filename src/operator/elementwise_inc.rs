use crate::operator::Operator;
use crate::signal::{ArraySignal, Get};
use core::ops::{AddAssign, Mul};
use numpy::TypeNum;
use std::rc::Rc;

pub struct ElementwiseInc<T>
where
    T: TypeNum,
{
    pub target: Rc<ArraySignal<T>>,
    pub left: Rc<ArraySignal<T>>,
    pub right: Rc<ArraySignal<T>>,
}

impl<T> Operator for ElementwiseInc<T>
where
    T: TypeNum + Mul<T, Output = T> + AddAssign<T>,
{
    fn step(&self) {
        let left = &(*self.left.get());
        let right = &(*self.right.get());
        let mut target = self.target.get_mut();
        *target += &(left * right);
    }
}
