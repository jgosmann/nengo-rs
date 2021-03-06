use crate::operator::Operator;
use crate::signal::{ArraySignal, ScalarSignal, SignalAccess};
use ndarray::ArrayD;
use numpy::Element;
use std::fmt::Debug;
use std::marker::PhantomData;
use std::ops::AddAssign;
use std::sync::Arc;

#[derive(Debug)]
pub struct CopyOp<T, S> {
    pub inc: bool,
    pub src: Arc<S>,
    pub dst: Arc<S>,
    pub data_type: PhantomData<T>,
}

impl<T: Element + Debug + AddAssign<T>> Operator for CopyOp<ArrayD<T>, ArraySignal<T>> {
    fn step(&self) {
        if self.inc {
            **self.dst.write() += &**self.src.read();
        } else {
            self.dst.write().assign(&self.src.read());
        }
    }
}

impl<T: Copy + Debug + AddAssign<T>> Operator for CopyOp<T, ScalarSignal<T>> {
    fn step(&self) {
        if self.inc {
            **self.dst.write() += **self.src.read();
        } else {
            **self.dst.write() = **self.src.read();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::signal::Signal;
    use crate::venv::activate_venv;
    use ndarray::prelude::*;
    use numpy::IntoPyArray;
    use pyo3::Python;

    #[test]
    fn it_copies_scalar_signals() {
        let op = CopyOp::<u64, ScalarSignal<u64>> {
            inc: false,
            src: Arc::new(ScalarSignal::<u64>::new("src".to_string(), 42)),
            dst: Arc::new(ScalarSignal::<u64>::new("dst".to_string(), 0)),
            data_type: PhantomData,
        };
        op.src.reset();
        op.dst.reset();

        op.step();

        assert_eq!(**op.src.read(), **op.dst.read());
    }

    #[test]
    fn it_copies_array_signals() {
        let gil = Python::acquire_gil();
        let py = gil.python();
        activate_venv(py);
        let op = CopyOp::<ArrayD<u64>, ArraySignal<u64>> {
            inc: false,
            src: Arc::new(ArraySignal::new(
                "src".to_string(),
                Array::from_elem(IxDyn(&[2]), 42).into_pyarray(py),
            )),
            dst: Arc::new(ArraySignal::new(
                "dst".to_string(),
                Array::zeros(IxDyn(&[2])).into_pyarray(py),
            )),
            data_type: PhantomData,
        };
        op.src.reset();
        op.dst.reset();

        op.step();

        assert_eq!(**op.src.read(), **op.dst.read());
    }

    #[test]
    fn it_increments_scalar_signals() {
        let op = CopyOp::<u64, ScalarSignal<u64>> {
            inc: true,
            src: Arc::new(ScalarSignal::<u64>::new("src".to_string(), 42)),
            dst: Arc::new(ScalarSignal::<u64>::new("dst".to_string(), 1)),
            data_type: PhantomData,
        };
        op.src.reset();
        op.dst.reset();

        op.step();

        assert_eq!(**op.dst.read(), 43);
    }

    #[test]
    fn it_increments_array_signals() {
        let gil = Python::acquire_gil();
        let py = gil.python();
        activate_venv(py);
        let op = CopyOp::<ArrayD<u64>, ArraySignal<u64>> {
            inc: true,
            src: Arc::new(ArraySignal::new(
                "src".to_string(),
                Array::from_elem(IxDyn(&[2]), 42).into_pyarray(py),
            )),
            dst: Arc::new(ArraySignal::new(
                "dst".to_string(),
                Array::ones(IxDyn(&[2])).into_pyarray(py),
            )),
            data_type: PhantomData,
        };
        op.src.reset();
        op.dst.reset();

        op.step();

        assert_eq!(**op.dst.read(), array![43, 43].into_dyn());
    }
}
