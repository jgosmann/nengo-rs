use crate::binding::Wrapper;
use crate::signal::{ArraySignal, ScalarSignal, Signal, SignalAccess};
use numpy::PyArrayDyn;
use pyo3::exceptions as exc;
use pyo3::prelude::*;
use std::any::type_name;
use std::sync::Arc;

#[pyclass]
pub struct RsSignal {
    signal: Arc<dyn Signal + Send + Sync>,
}

impl Wrapper<Arc<dyn Signal + Send + Sync>> for RsSignal {
    fn get(&self) -> &Arc<dyn Signal + Send + Sync> {
        &self.signal
    }
}

impl RsSignal {
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

#[pyclass(extends=RsSignal)]
pub struct RsSignalArrayF64 {}

#[pymethods]
impl RsSignalArrayF64 {
    #[new]
    fn new(signal: &PyAny) -> PyResult<(Self, RsSignal)> {
        let name = signal.getattr("name")?.extract()?;
        let initial_value = signal.getattr("initial_value")?;
        let initial_value: &PyArrayDyn<f64> = initial_value.extract()?;
        let signal = Arc::new(ArraySignal::new(name, initial_value));
        Ok((Self {}, RsSignal { signal }))
    }
}

#[pyclass(extends=RsSignal)]
pub struct RsSignalU64 {}

#[pymethods]
impl RsSignalU64 {
    #[new]
    fn new(name: String, initial_value: u64) -> PyResult<(Self, RsSignal)> {
        Ok((
            Self {},
            RsSignal {
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

#[pyclass(extends=RsSignal)]
pub struct RsSignalF64 {}

#[pymethods]
impl RsSignalF64 {
    #[new]
    fn new(name: String, initial_value: f64) -> PyResult<(Self, RsSignal)> {
        Ok((
            Self {},
            RsSignal {
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
