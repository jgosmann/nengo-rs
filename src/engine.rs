use crate::operator::{CopyOp, ElementwiseInc, Operator, Reset, TimeUpdate};
use crate::probe::{Probe, SignalProbe};
use crate::signal::{ArraySignal, Get, ScalarSignal, Signal};
use ndarray::ArrayD;
use numpy::{PyArrayDyn, ToPyArray};
use pyo3::exceptions;
use pyo3::prelude::*;
use std::any::Any;
use std::collections::HashMap;
use std::marker::PhantomData;
use std::rc::Rc;

#[pyclass]
struct PyRefToRust {
    wrapped: Box<dyn Any + 'static>,
}

impl PyRefToRust {
    fn new(wrapped: Box<dyn Any>) -> Self {
        PyRefToRust { wrapped }
    }

    fn get_wrapped<T>(&self) -> &'static T {
        &(*self.wrapped.downcast::<T>().unwrap())
    }
}

pub struct Engine {
    dt: f64,
    signals: Vec<Box<Rc<dyn Signal>>>,
    operators: Vec<Box<dyn Operator>>,
    probes: Vec<Box<dyn Probe>>,
    step: Rc<ScalarSignal<u64>>,
    time: Rc<ScalarSignal<f64>>,
}

impl Engine {
    pub fn new(dt: f64) -> Self {
        // FIXME provide signals and ops size on construction
        Engine {
            dt,
            signals: Vec::new(),
            operators: Vec::new(),
            probes: Vec::new(),
            step: Rc::new(ScalarSignal::new("step".to_string(), 0)),
            time: Rc::new(ScalarSignal::new("time".to_string(), 0.)),
        }
    }

    fn create_signal(signal: &PyAny) -> PyResult<PyRefToRust> {
        let name = signal.getattr("name")?.extract()?;
        let initial_value = signal.getattr("initial_value")?;
        let initial_value: &PyArrayDyn<f64> = initial_value.extract()?;
        let rust_signal = Rc::new(ArraySignal::new(name, initial_value));
        Ok(PyRefToRust {
            wrapped: Box::new(rust_signal),
        })
    }

    // fn add_signal<'a>(&mut self, signal: &PyAny) -> PyResult<()> {
    //     let name = signal.getattr("name")?.extract()?;
    //     let initial_value = signal.getattr("initial_value")?;
    //     match signal.getattr("dtype")?.getattr("name")?.extract()? {
    //         "float64" => {
    //             let initial_value: &PyArrayDyn<f64> = initial_value.extract()?;
    //             let rust_signal = Rc::new(ArraySignal::new(name, initial_value));
    //             self.signals.insert(signal.into(), Box::new(rust_signal));
    //             Ok(())
    //         }
    //         "int64" => {
    //             let initial_value: &PyArrayDyn<i64> = initial_value.extract()?;
    //             self.signals
    //                 .map_i64
    //                 .push(Rc::new(ArraySignal::new(name, initial_value)));
    //             Ok(self.signals.map_i64.len() - 1)
    //         }
    //         dtype => Err(PyErr::new::<exceptions::TypeError, _>(format!(
    //             "incompatible dtype: {}",
    //             dtype
    //         ))),
    //     }
    // }

    fn push_reset(&mut self, value: &PyAny, target: &PyRefToRust) -> PyResult<()> {
        let value: &PyArrayDyn<f64> = value.extract()?;
        let value = value.to_owned_array();
        self.operators
            .push(Box::new(Reset::<ArrayD<f64>, ArraySignal<f64>> {
                value,
                target: Rc::clone(target.get_wrapped()),
            }));
        Ok(())
    }

    fn push_time_update(
        &mut self,
        step_target: &PyRefToRust,
        time_target: &PyRefToRust,
    ) -> PyResult<()> {
        self.operators.push(Box::new(TimeUpdate {
            dt: self.dt,
            step_target: Rc::clone(step_target.get_wrapped()),
            time_target: Rc::clone(time_target.get_wrapped()),
        }));
        Ok(())
    }

    fn push_elementwise_inc(
        &mut self,
        target: &PyRefToRust,
        left: &PyRefToRust,
        right: &PyRefToRust,
    ) -> PyResult<()> {
        self.operators.push(Box::new(ElementwiseInc::<f64> {
            target: Rc::clone(target.get_wrapped()),
            left: Rc::clone(left.get_wrapped()),
            right: Rc::clone(right.get_wrapped()),
        }));
        Ok(())
    }

    fn push_copy(&mut self, src: &PyRefToRust, dst: &PyRefToRust) -> PyResult<()> {
        self.operators
            .push(Box::new(CopyOp::<ArrayD<f64>, ArraySignal<f64>> {
                src: Rc::clone(src.get_wrapped()),
                dst: Rc::clone(dst.get_wrapped()),
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
        for s in self.signals.iter() {
            s.reset();
        }
    }

    fn get_step(&self) -> u64 {
        *self.step.get()
    }

    fn get_time(&self) -> f64 {
        *self.time.get()
    }

    fn add_probe(&mut self, target: &PyRefToRust) -> usize {
        self.probes.push(Box::new(SignalProbe::new(
            target.get_wrapped::<Rc<ArraySignal<f64>>>(),
        )));
        self.probes.len() - 1
    }

    fn get_probe_data(&self, target: usize) -> PyResult<PyObject> {
        let gil = Python::acquire_gil();
        let py = gil.python();
        self.probes[target].get_data(py)
    }
}
