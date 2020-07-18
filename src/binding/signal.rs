use crate::binding::Wrapper;
use crate::signal::{ArraySignal, ScalarSignal, Signal, SignalAccess};
use numpy::PyArrayDyn;
use pyo3::exceptions as exc;
use pyo3::prelude::*;
use std::any::type_name;
use std::sync::Arc;

#[pyclass(name=Signal)]
#[derive(Debug)]
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

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::prelude::*;
    use ndarray::Ix;
    use pyo3::{types::IntoPyDict, wrap_pymodule, ToPyObject};
    use std::fmt::Debug;

    #[pymodule]
    fn signal(_py: Python, m: &PyModule) -> PyResult<()> {
        m.add_class::<PySignalArrayF64>()?;
        m.add_class::<PySignalF64>()?;
        m.add_class::<PySignalU64>()?;
        Ok(())
    }

    fn test_binding<T, S>(expr: &str, expected_name: &str, expected_shape: &[Ix], expected_value: T)
    where
        T: Debug + PartialEq,
        S: Signal + SignalAccess<T> + Send + Sync + 'static,
    {
        let gil = Python::acquire_gil();
        let py = gil.python();
        let nengo = PyModule::import(py, "nengo").unwrap();
        let numpy = PyModule::import(py, "numpy").unwrap();
        let signal_module = wrap_pymodule!(signal)(py);
        let locals = [
            ("nengo", nengo.to_object(py)),
            ("np", numpy.to_object(py)),
            ("s", signal_module),
        ]
        .into_py_dict(py);

        let py_signal = py.eval(expr, None, Some(locals)).unwrap();

        let py_signal: &PyCell<PySignal> = py_signal.extract().unwrap();

        let signal = py_signal.borrow();
        assert_eq!(signal.get().name(), expected_name);
        assert_eq!(signal.get().shape(), expected_shape);

        let signal: Arc<S> = py_signal.borrow().extract_signal("test").unwrap();
        signal.reset();
        assert_eq!(*signal.read(), expected_value);
    }

    #[test]
    fn test_py_signal_array_f64() {
        test_binding::<_, ArraySignal<f64>>(
            "s.SignalArrayF64(nengo.builder.signal.Signal(np.array([1., 2.]), name='TestSignal'))",
            "TestSignal",
            &[2],
            array![1., 2.].into_dimensionality::<IxDyn>().unwrap(),
        );
    }

    #[test]
    fn test_py_signal_array_f64_from_scalar() {
        test_binding::<_, ArraySignal<f64>>(
            "s.SignalArrayF64(nengo.builder.signal.Signal(np.float64(42.), name='TestSignal'))",
            "TestSignal",
            &[],
            array![42.].into_dimensionality::<IxDyn>().unwrap(),
        );
    }

    #[test]
    fn test_py_signal_u64() {
        test_binding::<_, ScalarSignal<u64>>("s.SignalU64('TestSignal', 2)", "TestSignal", &[], 2);
    }

    #[test]
    fn test_py_signal_f64() {
        test_binding::<_, ScalarSignal<f64>>(
            "s.SignalF64('TestSignal', 2.)",
            "TestSignal",
            &[],
            2.,
        );
    }
}
