use crate::operator::{CopyOp, ElementwiseInc, Operator, Reset, TimeUpdate};
use crate::probe::{Probe, SignalProbe};
use crate::signal::{ArraySignal, Signal};
use ndarray::ArrayD;
use numpy::{PyArrayDyn, ToPyArray};
use pyo3::exceptions;
use pyo3::prelude::*;
use std::marker::PhantomData;
use std::rc::Rc;

struct SignalMap {
    map_i64: Vec<Rc<ArraySignal<i64>>>,
    map_f64: Vec<Rc<ArraySignal<f64>>>,
}

#[pyclass]
pub struct Engine {
    dt: f64,
    signals: SignalMap,
    operators: Vec<Box<dyn Operator>>,
    probes: Vec<Box<dyn Probe>>,
}

#[pymethods]
impl Engine {
    #[new]
    fn new(dt: f64) -> Self {
        // FIXME provide signals and ops size on construction
        Engine {
            dt,
            signals: SignalMap {
                map_i64: Vec::new(),
                map_f64: Vec::new(),
            },
            operators: Vec::new(),
            probes: Vec::new(),
        }
    }

    fn add_signal<'a>(&mut self, signal: &PyAny) -> PyResult<usize> {
        let name = signal.getattr("name")?.extract()?;
        let initial_value = signal.getattr("initial_value")?;
        match signal.getattr("dtype")?.getattr("name")?.extract()? {
            "float64" => {
                let initial_value: &PyArrayDyn<f64> = initial_value.extract()?;
                self.signals
                    .map_f64
                    .push(Rc::new(ArraySignal::new(name, initial_value)));
                Ok(self.signals.map_f64.len() - 1)
            }
            "int64" => {
                let initial_value: &PyArrayDyn<i64> = initial_value.extract()?;
                self.signals
                    .map_i64
                    .push(Rc::new(ArraySignal::new(name, initial_value)));
                Ok(self.signals.map_i64.len() - 1)
            }
            dtype => Err(PyErr::new::<exceptions::TypeError, _>(format!(
                "incompatible dtype: {}",
                dtype
            ))),
        }
    }

    fn push_reset(&mut self, value: &PyAny, target: usize) -> PyResult<()> {
        let value: &PyArrayDyn<f64> = value.extract()?;
        let value = value.to_owned_array();
        self.operators
            .push(Box::new(Reset::<ArrayD<f64>, ArraySignal<f64>> {
                value,
                target: Rc::clone(&self.signals.map_f64[target]),
            }));
        Ok(())
    }

    fn push_time_update(&mut self, step_target: usize, time_target: usize) -> PyResult<()> {
        self.operators.push(Box::new(TimeUpdate {
            dt: self.dt,
            step_target: Rc::clone(&self.signals.map_i64[step_target]),
            time_target: Rc::clone(&self.signals.map_f64[step_target]),
        }));
        Ok(())
    }

    fn push_elementwise_inc(&mut self, target: usize, left: usize, right: usize) -> PyResult<()> {
        self.operators.push(Box::new(ElementwiseInc {
            target: Rc::clone(&self.signals.map_f64[target]),
            left: Rc::clone(&self.signals.map_f64[left]),
            right: Rc::clone(&self.signals.map_f64[right]),
        }));
        Ok(())
    }

    fn push_copy(&mut self, src: usize, dst: usize) -> PyResult<()> {
        self.operators
            .push(Box::new(CopyOp::<ArrayD<f64>, ArraySignal<f64>> {
                src: Rc::clone(&self.signals.map_f64[src]),
                dst: Rc::clone(&self.signals.map_f64[dst]),
                data_type: PhantomData,
            }));
        Ok(())
    }

    fn run_step(&mut self) {
        for op in self.operators.iter() {
            op.step();
        }
        for probe in self.probes.iter_mut() {
            probe.probe();
        }
    }

    fn run_steps(&mut self, n_steps: i64) {
        for _ in 0..n_steps {
            self.run_step();
        }
    }

    fn reset(&self) {
        for s in self.signals.map_i64.iter() {
            s.reset();
        }
        for s in self.signals.map_f64.iter() {
            s.reset();
        }
    }

    fn get_signal_i64(&self, py: Python, id: usize) -> Py<PyArrayDyn<i64>> {
        Py::from(self.signals.map_i64[id].get().to_pyarray(py))
    }

    fn add_probe(&mut self, target: usize) -> usize {
        self.probes
            .push(Box::new(SignalProbe::new(&self.signals.map_f64[target])));
        self.probes.len() - 1
    }

    fn get_probe_data(&self, target: usize) -> PyResult<PyObject> {
        let gil = Python::acquire_gil();
        let py = gil.python();
        self.probes[target].get_data(py)
    }
}
