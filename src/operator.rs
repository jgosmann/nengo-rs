mod copy;
mod elementwise_inc;
mod reset;
mod time_update;

pub use crate::operator::copy::*;
pub use crate::operator::elementwise_inc::*;
pub use crate::operator::reset::*;
pub use crate::operator::time_update::*;

pub trait Operator {
    fn step(&self);
}
