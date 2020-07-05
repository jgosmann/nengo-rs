use crate::operator::run_operators;
use crate::operator::{CopyOp, ElementwiseInc, OperatorNode, Reset, TimeUpdate};
use crate::probe::{Probe, SignalProbe};
use crate::signal::{ArraySignal, ScalarSignal, Signal, SignalAccess};
use futures::executor::ThreadPool;
use futures::future::Future;
use futures::stream::FuturesUnordered;
use futures::stream::StreamExt;
use ndarray::ArrayD;
use ndarray::Axis;
use numpy::PyArrayDyn;
use pyo3::exceptions as exc;
use pyo3::prelude::*;
use pyo3::PyClass;
use std::any::type_name;
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

fn extract_signal<T: Signal + Send + Sync + 'static>(
    name: &str,
    signal: &RsSignal,
) -> PyResult<Arc<T>> {
    Arc::downcast::<T>(Arc::clone(&signal.signal).as_any_arc()).or(Err(PyErr::new::<
        exc::TypeError,
        _,
    >(format!(
        "Signal `{}` must be {}.",
        name,
        type_name::<T>()
    ))))
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
                        target: extract_signal("target", target)?,
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
                        step_target: extract_signal("step_target", step_target)?,
                        time_target: extract_signal("time_target", time_target)?,
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
                        target: extract_signal("target", target)?,
                        left: extract_signal("left", left)?,
                        right: extract_signal("right", right)?,
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
                        src: extract_signal("src", src)?,
                        dst: extract_signal("dst", dst)?,
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
                SignalProbe::<ArrayD<f64>, ArraySignal<f64>>::new(&extract_signal(
                    "target", target,
                )?),
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

struct Event(Mutex<bool>, Condvar);

impl Event {
    pub fn new() -> Self {
        Event(Mutex::new(false), Condvar::new())
    }

    pub fn clear(&self) {
        self.set_value(false);
    }

    pub fn set(&self) {
        self.set_value(true);
    }

    fn set_value(&self, value: bool) {
        let Event(lock, cvar) = self;
        let mut finished = lock.lock().unwrap();
        if *finished != value {
            *finished = value;
            cvar.notify_all();
        }
    }

    pub fn wait(&self) {
        let Event(lock, cvar) = self;
        let mut finished = lock.lock().unwrap();

        while !*finished {
            finished = cvar.wait(finished).unwrap();
        }
    }
}

#[pyclass]
pub struct Engine {
    signals: Vec<Arc<dyn Signal + Send + Sync>>,
    operators: Vec<Arc<OperatorNode>>,
    probes: Vec<Arc<RwLock<dyn Probe + Send + Sync>>>,
    thread_pool: ThreadPool,
    is_done: Arc<Event>,
}

#[pymethods]
impl Engine {
    #[new]
    fn new(signals: &PyAny, operators: &PyAny, probes: &PyAny) -> PyResult<Self> {
        fn py_cells_to_pure_rust<T: PyClass + Wrapper<Arc<U>>, U: ?Sized>(
            cells: &Vec<&PyCell<T>>,
        ) -> Vec<Arc<U>> {
            cells.iter().map(|c| Arc::clone(c.borrow().get())).collect()
        }

        Ok(Self {
            signals: py_cells_to_pure_rust::<RsSignal, _>(&signals.extract()?),
            operators: py_cells_to_pure_rust::<RsOperator, _>(&operators.extract()?),
            probes: py_cells_to_pure_rust::<RsProbe, _>(&probes.extract()?),
            thread_pool: ThreadPool::new().unwrap(),
            is_done: Arc::new(Event::new()),
        })
    }

    fn run_step(&self) {
        self.run_threaded(Self::run_step_async(
            self.operators.clone(),
            self.probes.clone(),
        ));
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
    fn run_threaded<Fut: Future<Output = ()> + Send + 'static>(&self, fut: Fut) {
        self.is_done.clear();
        self.thread_pool
            .spawn_ok(Self::notify_when_done(fut, Arc::clone(&self.is_done)));
        self.is_done.wait();
    }

    async fn notify_when_done<Fut: Future<Output = ()> + Send + 'static>(
        fut: Fut,
        is_done: Arc<Event>,
    ) {
        fut.await;
        is_done.set();
    }

    async fn run_step_async(
        operators: Vec<Arc<OperatorNode>>,
        probes: Vec<Arc<RwLock<dyn Probe + Send + Sync>>>,
    ) {
        run_operators(&operators).await;
        probes
            .iter()
            .map(Self::probe_async)
            .collect::<FuturesUnordered<_>>()
            .collect::<()>()
            .await;
    }

    async fn probe_async(probe: &Arc<RwLock<dyn Probe + Send + Sync>>) {
        probe.write().unwrap().probe();
    }
}
