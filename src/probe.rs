use crate::signal::{ArraySignal, ScalarSignal, Signal, SignalAccess};
use ndarray::ArrayD;
use numpy::TypeNum;
use std::any::Any;
use std::sync::Arc;

pub trait Probe {
    fn as_any(&self) -> &dyn Any;
    fn probe(&mut self);
}

pub struct SignalProbe<T, S: Signal> {
    signal: Arc<S>,
    data: Vec<T>,
}

impl<T, S: Signal> SignalProbe<T, S> {
    pub fn new(signal: &Arc<S>) -> Self {
        SignalProbe::<T, S> {
            signal: Arc::clone(signal),
            data: vec![],
        }
    }
}

impl<T: TypeNum + Send + Sync + 'static> Probe for SignalProbe<ArrayD<T>, ArraySignal<T>> {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn probe(&mut self) {
        self.data.push(self.signal.read().clone())
    }
}

impl<T: TypeNum + Send + Sync + 'static> SignalProbe<ArrayD<T>, ArraySignal<T>> {
    pub fn get_data(&self) -> &Vec<ArrayD<T>> {
        &self.data
    }

    pub fn shape(&self) -> &[usize] {
        self.signal.shape()
    }
}

impl<T: TypeNum + Send + Sync + 'static> Probe for SignalProbe<T, ScalarSignal<T>> {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn probe(&mut self) {
        self.data.push(*self.signal.read());
    }
}

impl<T: TypeNum + Send + Sync + 'static> SignalProbe<T, ScalarSignal<T>> {
    pub fn get_data(&self) -> &Vec<T> {
        &self.data
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::prelude::*;
    use ndarray::IxDyn;
    use numpy::IntoPyArray;
    use pyo3::Python;
    use std::error::Error;

    #[test]
    fn it_can_probe_scalar_signal() {
        let probed_signal = Arc::new(ScalarSignal::new("probed".to_string(), 0));
        let mut probe = SignalProbe::<u64, _>::new(&Arc::clone(&probed_signal));

        probe.probe();
        *probed_signal.write() = 1;
        probe.probe();
        *probed_signal.write() = 42;
        probe.probe();

        assert_eq!(probe.get_data(), &vec![0, 1, 42]);
    }

    #[test]
    fn it_can_probe_array_signal() -> Result<(), Box<dyn Error>> {
        let gil = Python::acquire_gil();
        let py = gil.python();

        let probed_signal = Arc::new(ArraySignal::new(
            "probed".to_string(),
            array![0, 0]
                .into_dimensionality::<IxDyn>()?
                .into_pyarray(py),
        ));
        probed_signal.reset();
        let mut probe = SignalProbe::<ArrayD<u64>, _>::new(&Arc::clone(&probed_signal));

        probe.probe();
        probed_signal.write().assign(&array![1, 1]);
        probe.probe();
        probed_signal.write().assign(&array![42, 43]);
        probe.probe();

        assert_eq!(
            probe.get_data(),
            &vec![
                array![0, 0].into_dimensionality::<IxDyn>()?,
                array![1, 1].into_dimensionality::<IxDyn>()?,
                array![42, 43].into_dimensionality::<IxDyn>()?
            ]
        );
        Ok(())
    }
}
