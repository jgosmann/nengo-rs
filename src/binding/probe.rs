use crate::binding::signal::PySignal;
use crate::binding::Wrapper;
use crate::probe::{Probe, SignalProbe};
use crate::signal::ArraySignal;
use ndarray::ArrayD;
use ndarray::Axis;
use numpy::PyArrayDyn;
use pyo3::prelude::*;
use std::sync::Arc;
use std::sync::RwLock;

#[pyclass(name=Probe)]
pub struct PyProbe {
    probe: Arc<RwLock<dyn Probe + Send + Sync>>,
}

impl Wrapper<Arc<RwLock<dyn Probe + Send + Sync>>> for PyProbe {
    fn get(&self) -> &Arc<RwLock<dyn Probe + Send + Sync>> {
        &self.probe
    }
}

#[pymethods]
impl PyProbe {
    #[new]
    fn new(target: &PySignal) -> PyResult<Self> {
        Ok(Self {
            probe: Arc::new(RwLock::new(
                SignalProbe::<ArrayD<f64>, ArraySignal<f64>>::new(
                    &target.extract_signal("target")?,
                ),
            )),
        })
    }

    fn get_data(&self) -> PyResult<PyObject> {
        let probe = self.probe.read().unwrap();
        let probe = probe
            .as_any()
            .downcast_ref::<SignalProbe<ArrayD<f64>, ArraySignal<f64>>>()
            .unwrap();
        let data = probe.get_data();

        let gil = Python::acquire_gil();
        let py = gil.python();
        let copy = PyArrayDyn::new(py, [&[data.len()], probe.shape()].concat(), false);
        for (i, x) in data.iter().enumerate() {
            unsafe {
                copy.as_array_mut().index_axis_mut(Axis(0), i).assign(x);
            }
        }
        Ok(copy.to_object(py))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::binding::signal::PySignalArrayF64;
    use crate::signal::{ArraySignal, Signal, SignalAccess};
    use crate::venv::activate_venv;
    use ndarray::prelude::*;
    use pyo3::{types::IntoPyDict, wrap_pymodule, ToPyObject};

    #[pymodule]
    fn probe(_py: Python, m: &PyModule) -> PyResult<()> {
        m.add_class::<PyProbe>()?;
        m.add_class::<PySignalArrayF64>()?;
        Ok(())
    }

    #[test]
    fn test_probe_binding() {
        let gil = Python::acquire_gil();
        let py = gil.python();
        activate_venv(py);
        let nengo = PyModule::import(py, "nengo").unwrap();
        let numpy = PyModule::import(py, "numpy").unwrap();
        let probe_module = wrap_pymodule!(probe)(py);
        let locals = [
            ("nengo", nengo.to_object(py)),
            ("np", numpy.to_object(py)),
            ("p", probe_module),
        ]
        .into_py_dict(py);

        let py_signal = py
            .eval(
                "p.SignalArrayF64(nengo.builder.signal.Signal(np.array([1., 2.]), name='TestSignal'))",
                None,
                Some(locals),
            )
            .unwrap();
        let py_signal: &PyCell<PySignal> = py_signal.extract().unwrap();
        let py_probe = py
            .eval(
                "p.Probe(signal)",
                Some(locals),
                Some([("signal", py_signal)].into_py_dict(py)),
            )
            .unwrap();
        let py_probe: &PyCell<PyProbe> = py_probe.extract().unwrap();

        let signal: Arc<ArraySignal<f64>> = py_signal.borrow().extract_signal("test").unwrap();
        let probe: Arc<RwLock<dyn Probe + Send + Sync>> = Arc::clone(&py_probe.borrow().get());

        signal.reset();
        probe.write().unwrap().probe();
        signal.write().assign_array(&array![42., 42.]);
        probe.write().unwrap().probe();

        let data = py
            .eval(
                "probe.get_data()",
                Some(locals),
                Some([("probe", py_probe)].into_py_dict(py)),
            )
            .unwrap();
        let data: &PyArrayDyn<f64> = data.extract().unwrap();
        assert_eq!(
            data.readonly().as_array(),
            array![[1., 2.], [42., 42.]]
                .into_dimensionality::<IxDyn>()
                .unwrap()
        );
    }
}
