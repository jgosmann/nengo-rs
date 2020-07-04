use crate::operator::run_operators;
use crate::operator::{CopyOp, ElementwiseInc, OperatorNode, Reset, TimeUpdate};
use crate::probe::{Probe, SignalProbe};
use crate::signal::{ArraySignal, Get, ScalarSignal, Signal};
use futures::executor::ThreadPool;
use futures::stream::FuturesUnordered;
use futures::stream::StreamExt;
use ndarray::ArrayD;
use numpy::PyArrayDyn;
use pyo3::prelude::*;
use std::marker::PhantomData;
use std::sync::Arc;
use std::sync::Condvar;
use std::sync::Mutex;
use std::sync::RwLock;

#[pyclass]
pub struct RsSignal {
    signal: Arc<dyn Signal + Send + Sync>,
}

#[pyclass]
pub struct RsOperator {
    node: Arc<OperatorNode>,
}

#[pyclass]
pub struct RsProbe {
    probe: Arc<RwLock<dyn Probe + Send + Sync>>,
}

trait Wrapper<T> {
    fn get(&self) -> &T;
}

impl Wrapper<Arc<dyn Signal + Send + Sync>> for RsSignal {
    fn get(&self) -> &Arc<dyn Signal + Send + Sync> {
        &self.signal
    }
}

impl Wrapper<Arc<OperatorNode>> for RsOperator {
    fn get(&self) -> &Arc<OperatorNode> {
        &self.node
    }
}
impl Wrapper<Arc<RwLock<dyn Probe + Send + Sync>>> for RsProbe {
    fn get(&self) -> &Arc<RwLock<dyn Probe + Send + Sync>> {
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
            .get()
    }
}

#[pyclass(extends=RsOperator)]
pub struct RsReset {}

#[pymethods]
impl RsReset {
    #[new]
    fn new(
        value: &PyAny,
        target: &RsSignal,
        dependencies: Vec<usize>,
    ) -> PyResult<(Self, RsOperator)> {
        let value: &PyArrayDyn<f64> = value.extract()?;
        let value = value.to_owned_array();
        Ok((
            Self {},
            RsOperator {
                node: Arc::new(OperatorNode {
                    operator: Box::new(Reset::<ArrayD<f64>, ArraySignal<f64>> {
                        value,
                        target: Arc::downcast::<ArraySignal<f64>>(
                            Arc::clone(&target.signal).as_any_rc(),
                        )
                        .unwrap(),
                    }),
                    dependencies,
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
        dependencies: Vec<usize>,
    ) -> PyResult<(Self, RsOperator)> {
        Ok((
            Self {},
            RsOperator {
                node: Arc::new(OperatorNode {
                    operator: Box::new(TimeUpdate::<f64, u64> {
                        dt,
                        step_target: Arc::downcast::<ScalarSignal<u64>>(
                            Arc::clone(&step_target.signal).as_any_rc(),
                        )
                        .unwrap(),
                        time_target: Arc::downcast::<ScalarSignal<f64>>(
                            Arc::clone(&time_target.signal).as_any_rc(),
                        )
                        .unwrap(),
                    }),
                    dependencies,
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
    fn new(
        target: &RsSignal,
        left: &RsSignal,
        right: &RsSignal,
        dependencies: Vec<usize>,
    ) -> PyResult<(Self, RsOperator)> {
        Ok((
            Self {},
            RsOperator {
                node: Arc::new(OperatorNode {
                    operator: Box::new(ElementwiseInc::<f64> {
                        target: Arc::downcast::<ArraySignal<f64>>(
                            Arc::clone(&target.signal).as_any_rc(),
                        )
                        .unwrap(),
                        left: Arc::downcast::<ArraySignal<f64>>(
                            Arc::clone(&left.signal).as_any_rc(),
                        )
                        .unwrap(),
                        right: Arc::downcast::<ArraySignal<f64>>(
                            Arc::clone(&right.signal).as_any_rc(),
                        )
                        .unwrap(),
                    }),
                    dependencies,
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
    fn new(
        src: &RsSignal,
        dst: &RsSignal,
        dependencies: Vec<usize>,
    ) -> PyResult<(Self, RsOperator)> {
        Ok((
            Self {},
            RsOperator {
                node: Arc::new(OperatorNode {
                    operator: Box::new(CopyOp::<ArrayD<f64>, ArraySignal<f64>> {
                        src: Arc::downcast::<ArraySignal<f64>>(Arc::clone(&src.signal).as_any_rc())
                            .unwrap(),
                        dst: Arc::downcast::<ArraySignal<f64>>(Arc::clone(&dst.signal).as_any_rc())
                            .unwrap(),
                        data_type: PhantomData,
                    }),
                    dependencies,
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
            probe: Arc::new(RwLock::new(
                SignalProbe::<ArrayD<f64>, ArraySignal<f64>>::new(
                    &Arc::downcast::<ArraySignal<f64>>(Arc::clone(&target.signal).as_any_rc())
                        .unwrap(),
                ),
            )),
        })
    }

    fn get_data(&self) -> PyResult<PyObject> {
        let gil = Python::acquire_gil();
        let py = gil.python();
        (self.probe.read().unwrap())
            .as_any()
            .downcast_ref::<SignalProbe<ArrayD<f64>, ArraySignal<f64>>>()
            .unwrap()
            .get_data(py)
    }
}

#[pyclass]
pub struct Engine {
    signals: Vec<Arc<dyn Signal + Send + Sync>>,
    operators: Vec<Arc<OperatorNode>>,
    probes: Vec<Arc<RwLock<dyn Probe + Send + Sync>>>,
    thread_pool: ThreadPool,
    step_finished_indicator: Arc<(Mutex<bool>, Condvar)>,
}

#[pymethods]
impl Engine {
    #[new]
    fn new(signals: &PyAny, operators: &PyAny, probes: &PyAny) -> PyResult<Self> {
        let signals: Vec<&PyCell<RsSignal>> = signals.extract()?;
        let signals: Vec<Arc<dyn Signal + Send + Sync>> = signals
            .iter()
            .map(|s| Arc::clone(s.borrow().get()))
            .collect();

        let operators: Vec<&PyCell<RsOperator>> = operators.extract()?;
        let operators: Vec<Arc<OperatorNode>> = operators
            .iter()
            .map(|o| Arc::clone(o.borrow().get()))
            .collect();

        let probes: Vec<&PyCell<RsProbe>> = probes.extract()?;
        let probes: Vec<Arc<RwLock<dyn Probe + Send + Sync>>> = probes
            .iter()
            .map(|p| Arc::clone(p.borrow().get()))
            .collect();

        Ok(Self {
            signals,
            operators,
            probes,
            thread_pool: ThreadPool::new().unwrap(),
            step_finished_indicator: Arc::new((Mutex::new(false), Condvar::new())),
        })
    }

    fn run_step(&self) {
        let (lock, cvar) = &*self.step_finished_indicator;
        let mut finished = lock.lock().unwrap();
        *finished = false;

        self.thread_pool.spawn_ok(Self::run_step_async(
            self.operators.clone(),
            self.probes.clone(),
            Arc::clone(&self.step_finished_indicator),
        ));

        while !*finished {
            finished = cvar.wait(finished).unwrap();
        }
    }

    fn run_steps(&self, n_steps: i64) {
        for _ in 0..n_steps {
            self.run_step();
        }
    }

    fn reset(&self) {
        self.signals.iter().for_each(|s| s.reset());
    }
}

impl Engine {
    async fn run_step_async(
        operators: Vec<Arc<OperatorNode>>,
        probes: Vec<Arc<RwLock<dyn Probe + Send + Sync>>>,
        step_finished_indicator: Arc<(Mutex<bool>, Condvar)>,
    ) {
        run_operators(&operators).await;

        probes
            .iter()
            .map(Self::probe_async)
            .collect::<FuturesUnordered<_>>()
            .collect::<()>()
            .await;

        let (lock, cvar) = &*step_finished_indicator;
        *lock.lock().unwrap() = true;
        cvar.notify_one();
    }

    async fn probe_async(probe: &Arc<RwLock<dyn Probe + Send + Sync>>) {
        probe.write().unwrap().probe();
    }
}
