mod copy;
mod dot_inc;
mod elementwise_inc;
mod reset;
mod sim_neurons;
mod sim_process;
mod sim_pyfunc;
mod time_update;

pub use crate::operator::copy::*;
pub use crate::operator::dot_inc::*;
pub use crate::operator::elementwise_inc::*;
pub use crate::operator::reset::*;
pub use crate::operator::sim_neurons::*;
pub use crate::operator::sim_process::*;
pub use crate::operator::sim_pyfunc::*;
pub use crate::operator::time_update::*;
use std::fmt::Debug;

pub trait Operator: Debug {
    fn step(&self);
}

pub struct OperatorNode {
    pub operator: Box<dyn Operator + Sync + Send>,
    pub dependencies: Vec<usize>,
}
