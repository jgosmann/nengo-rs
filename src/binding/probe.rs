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
            copy.as_array_mut().index_axis_mut(Axis(0), i).assign(x);
        }
        Ok(copy.to_object(py))
    }
}
