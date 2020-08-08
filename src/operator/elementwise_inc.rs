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
        let left = self.left.read();
        let right = self.right.read();
        let mut target = self.target.write();
        **target += &(&**left * &**right);
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::signal::Signal;
    use crate::venv::activate_venv;
    use ndarray::prelude::*;
    use numpy::IntoPyArray;
    use pyo3::Python;
    use std::error::Error;

    #[test]
    fn it_performs_an_elementwise_increment() -> Result<(), Box<dyn Error>> {
        let gil = Python::acquire_gil();
        let py = gil.python();
        activate_venv(py);
        let op = ElementwiseInc::<u64> {
            target: Arc::new(ArraySignal::new(
                "target".to_string(),
                Array::ones(IxDyn(&[2])).into_pyarray(py),
            )),
            left: Arc::new(ArraySignal::new(
                "left".to_string(),
                array![2, 3]
                    .into_dimensionality::<IxDyn>()?
                    .into_pyarray(py),
            )),
            right: Arc::new(ArraySignal::new(
                "right".to_string(),
                array![4, 5]
                    .into_dimensionality::<IxDyn>()?
                    .into_pyarray(py),
            )),
        };
        for signal in vec![&op.target, &op.left, &op.right].iter() {
            signal.reset();
        }

        op.step();

        assert_eq!(
            **op.target.read(),
            array![9, 16].into_dimensionality::<IxDyn>()?
        );
        Ok(())
    }
}
