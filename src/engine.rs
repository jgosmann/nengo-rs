use crate::operator::{CopyOp, ElementwiseInc, Operator, Reset, TimeUpdate};
use crate::probe::{Probe, SignalProbe};
use crate::signal::{ArraySignal, Get, ScalarSignal, Signal};
use ndarray::ArrayD;
use numpy::PyArrayDyn;
use pyo3::prelude::*;
use pyo3::PyClass;
use std::cell::RefCell;
use std::marker::PhantomData;
use std::rc::Rc;

#[pyclass]
pub struct RsSignal {
    signal: Rc<dyn Signal>,
}

#[pyclass]
pub struct RsOperator {
    operator: Rc<dyn Operator>,
}

#[pyclass]
pub struct RsProbe {
    probe: Rc<RefCell<dyn Probe>>,
}

trait Wrapper<T> {
    fn get(&self) -> &T;
}

impl Wrapper<Rc<dyn Signal>> for RsSignal {
    fn get(&self) -> &Rc<dyn Signal> {
        &self.signal
    }
}

impl Wrapper<Rc<dyn Operator>> for RsOperator {
    fn get(&self) -> &Rc<dyn Operator> {
        &self.operator
    }
}
impl Wrapper<Rc<RefCell<dyn Probe>>> for RsProbe {
    fn get(&self) -> &Rc<RefCell<dyn Probe>> {
        &self.probe
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
        let signal = Rc::new(ArraySignal::new(name, initial_value));
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
                signal: Rc::new(ScalarSignal::new(name, initial_value)),
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
            .get()
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
                signal: Rc::new(ScalarSignal::new(name, initial_value)),
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
            .get()
    }
}

#[pyclass(extends=RsOperator)]
pub struct RsReset {}

#[pymethods]
impl RsReset {
    #[new]
    fn new(value: &PyAny, target: &RsSignal) -> PyResult<(Self, RsOperator)> {
        let value: &PyArrayDyn<f64> = value.extract()?;
        let value = value.to_owned_array();
        Ok((
            Self {},
            RsOperator {
                operator: Rc::new(Reset::<ArrayD<f64>, ArraySignal<f64>> {
                    value,
                    target: Rc::clone(target.signal.as_any().downcast_ref().unwrap()),
                }),
            },
        ))
    }
}

#[pyclass(extends=RsOperator)]
pub struct RsTimeUpdate {}

#[pymethods]
impl RsTimeUpdate {
    #[new]
    fn new(
        dt: f64,
        step_target: &RsSignal,
        time_target: &RsSignal,
    ) -> PyResult<(Self, RsOperator)> {
        Ok((
            Self {},
            RsOperator {
                operator: Rc::new(TimeUpdate::<f64, u64> {
                    dt,
                    step_target: Rc::clone(step_target.signal.as_any().downcast_ref().unwrap()),
                    time_target: Rc::clone(time_target.signal.as_any().downcast_ref().unwrap()),
                }),
            },
        ))
    }
}

#[pyclass(extends=RsOperator)]
pub struct RsElementwiseInc {}

#[pymethods]
impl RsElementwiseInc {
    #[new]
    fn new(target: &RsSignal, left: &RsSignal, right: &RsSignal) -> PyResult<(Self, RsOperator)> {
        Ok((
            Self {},
            RsOperator {
                operator: Rc::new(ElementwiseInc::<f64> {
                    target: Rc::clone(target.signal.as_any().downcast_ref().unwrap()),
                    left: Rc::clone(left.signal.as_any().downcast_ref().unwrap()),
                    right: Rc::clone(right.signal.as_any().downcast_ref().unwrap()),
                }),
            },
        ))
    }
}

#[pyclass(extends=RsOperator)]
pub struct RsCopy {}

#[pymethods]
impl RsCopy {
    #[new]
    fn new(src: &RsSignal, dst: &RsSignal) -> PyResult<(Self, RsOperator)> {
        Ok((
            Self {},
            RsOperator {
                operator: Rc::new(CopyOp::<ArrayD<f64>, ArraySignal<f64>> {
                    src: Rc::clone(src.signal.as_any().downcast_ref().unwrap()),
                    dst: Rc::clone(dst.signal.as_any().downcast_ref().unwrap()),
                    data_type: PhantomData,
                }),
            },
        ))
    }
}

#[pymethods]
impl RsProbe {
    #[new]
    fn new(target: &RsSignal) -> PyResult<Self> {
        Ok(Self {
            probe: Rc::new(RefCell::new(
                SignalProbe::<ArrayD<f64>, ArraySignal<f64>>::new(
                    target.signal.as_any().downcast_ref().unwrap(),
                ),
            )),
        })
    }

    fn get_probe_data(&self) -> PyResult<PyObject> {
        let gil = Python::acquire_gil();
        let py = gil.python();
        self.probe
            .borrow()
            .as_any()
            .downcast_ref::<SignalProbe<ArrayD<f64>, ArraySignal<f64>>>()
            .unwrap()
            .get_data(py)
    }
}

#[pyclass]
pub struct Engine {
    signals: Vec<Rc<dyn Signal>>,
    operators: Vec<Rc<dyn Operator>>,
    probes: Vec<Rc<RefCell<dyn Probe>>>,
}

#[pymethods]
impl Engine {
    #[new]
    fn new(signals: &PyAny, operators: &PyAny, probes: &PyAny) -> PyResult<Self> {
        fn from_any<T: PyClass + Wrapper<Rc<U>>, U: ?Sized>(
            sequence: &Vec<&PyCell<T>>,
        ) -> Vec<Rc<U>> {
            sequence
                .iter()
                .map(|s| Rc::clone(s.borrow().get()))
                .collect()
        }

        Ok(Self {
            signals: from_any::<RsSignal, dyn Signal>(&signals.extract()?),
            operators: from_any::<RsOperator, dyn Operator>(&operators.extract()?),
            probes: from_any::<RsProbe, RefCell<dyn Probe>>(&probes.extract()?),
        })
    }

    fn run_step(&mut self) {
        for op in self.operators.iter() {
            op.step();
        }
        for probe in self.probes.iter_mut() {
            probe.borrow_mut().probe();
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
}
