use crate::operator::Operator;
use crate::signal::{ArraySignal, ScalarSignal, Signal, SignalAccess};
use ndarray::ArrayD;
use numpy::Element;
use std::fmt::Debug;
use std::sync::Arc;

pub struct Reset<T, S>
where
    S: Signal,
{
    pub value: T,
    pub target: Arc<S>,
}

impl<T: Element + Debug + Send + Sync + 'static> Operator for Reset<ArrayD<T>, ArraySignal<T>> {
    fn step(&self) {
        self.target.write().assign_array(&self.value);
    }
}

impl<T: Send + Sync + Copy + Debug + 'static> Operator for Reset<T, ScalarSignal<T>> {
    fn step(&self) {
        **self.target.write() = self.value;
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::signal::Signal;
    use ndarray::prelude::*;
    use numpy::IntoPyArray;
    use pyo3::Python;
    use std::error::Error;

    #[test]
    fn it_assigns_the_value_for_scalar_signals() {
        let op = Reset::<u64, ScalarSignal<u64>> {
            value: 42,
            target: Arc::new(ScalarSignal::new("target".to_string(), 0)),
        };
        op.target.reset();

        op.step();

        assert_eq!(**op.target.read(), 42);
    }

    #[test]
    fn it_assigns_the_value_for_array_signals() -> Result<(), Box<dyn Error>> {
        let gil = Python::acquire_gil();
        let py = gil.python();
        let op = Reset::<ArrayD<u64>, ArraySignal<u64>> {
            value: array![1, 2].into_dimensionality::<IxDyn>()?,
            target: Arc::new(ArraySignal::new(
                "target".to_string(),
                Array::zeros(IxDyn(&[2])).into_pyarray(py),
            )),
        };
        op.target.reset();

        op.step();

        assert_eq!(
            **op.target.read(),
            array![1, 2].into_dimensionality::<IxDyn>()?
        );
        Ok(())
    }
}
