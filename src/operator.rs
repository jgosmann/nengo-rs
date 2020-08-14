mod copy;
mod elementwise_inc;
mod reset;
mod sim_pyfunc;
mod time_update;

pub use crate::operator::copy::*;
pub use crate::operator::elementwise_inc::*;
pub use crate::operator::reset::*;
pub use crate::operator::sim_pyfunc::*;
pub use crate::operator::time_update::*;

pub trait Operator {
    fn step(&self);
}

pub struct OperatorNode {
    pub operator: Box<dyn Operator + Sync + Send>,
    pub dependencies: Vec<usize>,
}
