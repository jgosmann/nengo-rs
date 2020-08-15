use crate::operator::Operator;
use crate::signal::{ArraySignal, SignalAccess};
use core::ops::AddAssign;
use ndarray::LinalgScalar;
use numpy::Element;
use std::fmt::Debug;
use std::sync::Arc;

#[derive(Debug)]
pub struct DotInc<T>
where
    T: Element,
{
    pub target: Arc<ArraySignal<T>>,
    pub left: Arc<ArraySignal<T>>,
    pub right: Arc<ArraySignal<T>>,
}

impl<T> Operator for DotInc<T>
where
    T: Element + AddAssign<T> + LinalgScalar + Debug,
{
    fn step(&self) {
        let left = self.left.read();
        let right = self.right.read();
        let mut target = self.target.write();
        **target += &(**left).dot(&**right);
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
    fn it_performs_a_dot_product() -> Result<(), Box<dyn Error>> {
        let gil = Python::acquire_gil();
        let py = gil.python();
        activate_venv(py);
        let op = DotInc::<u64> {
            target: Arc::new(ArraySignal::new(
                "target".to_string(),
                Array::ones(IxDyn(&[1])).into_pyarray(py),
            )),
            left: Arc::new(ArraySignal::new(
                "left".to_string(),
                array![[2, 3]].into_dyn().into_pyarray(py),
            )),
            right: Arc::new(ArraySignal::new(
                "right".to_string(),
                array![6, 7].into_dyn().into_pyarray(py),
            )),
        };
        for signal in vec![&op.target, &op.left, &op.right].iter() {
            signal.reset();
        }

        op.step();

        assert_eq!(**op.target.read(), array![34].into_dyn());
        Ok(())
    }

    #[test]
    fn it_performs_a_matrix_vector_product() -> Result<(), Box<dyn Error>> {
        let gil = Python::acquire_gil();
        let py = gil.python();
        activate_venv(py);
        let op = DotInc::<u64> {
            target: Arc::new(ArraySignal::new(
                "target".to_string(),
                Array::ones(IxDyn(&[2])).into_pyarray(py),
            )),
            left: Arc::new(ArraySignal::new(
                "left".to_string(),
                array![[2, 3], [4, 5]].into_dyn().into_pyarray(py),
            )),
            right: Arc::new(ArraySignal::new(
                "right".to_string(),
                array![6, 7].into_dyn().into_pyarray(py),
            )),
        };
        for signal in vec![&op.target, &op.left, &op.right].iter() {
            signal.reset();
        }

        op.step();

        assert_eq!(**op.target.read(), array![34, 60].into_dyn());
        Ok(())
    }
}
