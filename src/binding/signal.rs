use crate::binding::Wrapper;
use crate::signal::{ArraySignal, ScalarSignal, Signal, SignalAccess};
use numpy::PyArrayDyn;
use pyo3::exceptions as exc;
use pyo3::prelude::*;
use std::any::type_name;
use std::sync::Arc;

#[pyclass(name=Signal)]
pub struct PySignal {
    signal: Arc<dyn Signal + Send + Sync>,
}

impl Wrapper<Arc<dyn Signal + Send + Sync>> for PySignal {
    fn get(&self) -> &Arc<dyn Signal + Send + Sync> {
        &self.signal
    }
}

impl PySignal {
    pub fn extract_signal<T: Signal + Send + Sync + 'static>(
        &self,
        name: &str,
    ) -> PyResult<Arc<T>> {
        Arc::downcast::<T>(Arc::clone(&self.signal).as_any_arc()).or(Err(PyErr::new::<
            exc::TypeError,
            _,
        >(format!(
            "Signal `{}` must be {}.",
            name,
            type_name::<T>()
        ))))
    }
}

#[pyclass(extends=PySignal, name=SignalArrayF64)]
pub struct PySignalArrayF64 {}

#[pymethods]
impl PySignalArrayF64 {
    #[new]
    fn new(signal: &PyAny) -> PyResult<(Self, PySignal)> {
        let name = signal.getattr("name")?.extract()?;
        let initial_value = signal.getattr("initial_value")?;
        let initial_value: &PyArrayDyn<f64> = initial_value.extract()?;
        let signal = Arc::new(ArraySignal::new(name, initial_value));
        Ok((Self {}, PySignal { signal }))
    }
}

#[pyclass(extends=PySignal, name=SignalU64)]
pub struct PySignalU64 {}

#[pymethods]
impl PySignalU64 {
    #[new]
    fn new(name: String, initial_value: u64) -> PyResult<(Self, PySignal)> {
        Ok((
            Self {},
            PySignal {
                signal: Arc::new(ScalarSignal::new(name, initial_value)),
            },
        ))
    }

    fn get(py_self: PyRef<Self>) -> u64 {
        *py_self
            .as_ref()
            .signal
            .as_any()
            .downcast_ref::<ScalarSignal<u64>>()
            .unwrap()
            .read()
    }
}

#[pyclass(extends=PySignal, name=SignalF64)]
pub struct PySignalF64 {}

#[pymethods]
impl PySignalF64 {
    #[new]
    fn new(name: String, initial_value: f64) -> PyResult<(Self, PySignal)> {
        Ok((
            Self {},
            PySignal {
                signal: Arc::new(ScalarSignal::new(name, initial_value)),
            },
        ))
    }

    fn get(py_self: PyRef<Self>) -> f64 {
        *py_self
            .as_ref()
            .signal
            .as_any()
            .downcast_ref::<ScalarSignal<f64>>()
            .unwrap()
            .read()
    }
}
