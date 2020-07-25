use crate::operator::Operator;
use crate::signal::{ArraySignal, ScalarSignal, SignalAccess};
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
        self.dst.write().assign(&self.src.read());
    }
}

impl<T: Copy> Operator for CopyOp<T, ScalarSignal<T>> {
    fn step(&self) {
        **self.dst.write() = **self.src.read();
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
}
