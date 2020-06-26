use core::ops::{AddAssign, Mul};
use ndarray::prelude::*;
use ndarray::{Array, ArrayD, ArrayViewMut, Axis, ScalarOperand};
use numpy::convert::IntoPyArray;
use numpy::npyffi;
use numpy::{PyArray, PyArray1, PyArrayDyn, ToPyArray, TypeNum};
use pyo3::conversion::FromPyObject;
use pyo3::exceptions;
use pyo3::prelude::*;
use pyo3::types::PyTuple;
use pyo3::PyDowncastError;
use std::any::TypeId;
use std::borrow::{Borrow, BorrowMut};
use std::cell::{Ref, RefCell, RefMut};
use std::marker::PhantomData;
use std::rc::Rc;

type SignalBuffer<T> = RefCell<ArrayD<T>>;

trait Signal<T> {
    fn name(&self) -> &String;
    fn get(&self) -> &T;
    fn get_mut(&self) -> &mut T;
    fn reset(&mut self);
}

struct ScalarSignal<T> {
    name: String,
    value: T,
    initial_value: T,
}

impl<T> ScalarSignal<T> {
    fn new(name: String, initial_value: T) -> Self {
        ScalarSignal {
            name,
            value: initial_value,
            initial_value,
        }
    }
}

impl<T> Signal<T> for ScalarSignal<T> {
    fn name(&self) -> &String {
        &self.name
    }

    fn get(&self) -> &T {
        &self.value
    }

    fn get_mut(&self) -> &mut T {
        &mut self.value
    }

    fn reset(&mut self) {
        self.value = self.initial_value;
    }
}

struct ArraySignal<T: TypeNum> {
    name: String,
    buffer: SignalBuffer<T>,
    initial_value: Py<PyArrayDyn<T>>,
}

impl<T: TypeNum> ArraySignal<T> {
    fn new(name: String, initial_value: &PyArrayDyn<T>) -> Self {
        ArraySignal {
            name,
            buffer: SignalBuffer::new(unsafe {
                Array::uninitialized(match initial_value.shape() {
                    [] => &[1],
                    x => x,
                })
            }),
            initial_value: Py::from(initial_value),
        }
    }
}

impl<T: TypeNum> Signal<ArrayD<T>> for ArraySignal<T> {
    fn name(&self) -> &String {
        &self.name
    }

    fn get(&self) -> &ArrayD<T> {
        &self.buffer.borrow()
    }

    fn get_mut(&self) -> &mut ArrayD<T> {
        &mut self.buffer.borrow_mut()
    }

    fn reset(&mut self) {
        let gil = Python::acquire_gil();
        let py = gil.python();
        self.get_mut()
            .assign(&self.initial_value.as_ref(py).as_array())
    }
}

trait Operator {
    fn step(&self);
}

struct Reset<T, S>
where
    S: Signal<T>,
{
    value: T,
    target: Rc<S>,
}

impl<T: TypeNum> Operator for Reset<ArrayD<T>, ArraySignal<T>> {
    fn step(&self) {
        self.target.get_mut().assign(&self.value);
    }
}

impl<T> Operator for Reset<T, ScalarSignal<T>> {
    fn step(&self) {
        *self.target.get_mut() = self.value;
    }
}

struct TimeUpdate {
    dt: f64,
    step_target: Rc<ScalarSignal<i64>>,
    time_target: Rc<ScalarSignal<f64>>,
}

impl Operator for TimeUpdate {
    fn step(&self) {
        *self.step_target.get_mut() += 1;
        *self.time_target.get_mut() = *self.step_target.get() as f64 * self.dt;
    }
}

struct ElementwiseInc<T>
where
    T: TypeNum,
{
    target: Rc<ArraySignal<T>>,
    left: Rc<ArraySignal<T>>,
    right: Rc<ArraySignal<T>>,
}

impl<T> Operator for ElementwiseInc<T>
where
    T: TypeNum + Mul<T, Output = T> + AddAssign<T>,
{
    fn step(&self) {
        let left = self.left.get();
        let right = self.right.get();
        let mut target = self.target.get_mut();
        *target += &(left * right);
    }
}

struct CopyOp<T, S> {
    src: Rc<S>,
    dst: Rc<S>,
    data_type: PhantomData<T>,
}

impl<T: TypeNum> Operator for CopyOp<ArrayD<T>, ArraySignal<T>> {
    fn step(&self) {
        self.dst.get_mut().assign(&self.src.get());
    }
}

impl<T> Operator for CopyOp<T, ScalarSignal<T>> {
    fn step(&self) {
        *self.dst.get_mut() = *self.src.get();
    }
}

trait Probe {
    fn probe(&mut self);
    fn get_data(&self, py: Python) -> PyResult<PyObject>;
}

struct SignalProbe<T, S>
where
    S: Signal<T>,
{
    signal: Rc<S>,
    data: Vec<T>,
}

impl<T: TypeNum> Probe for SignalProbe<ArrayD<T>, ArraySignal<T>> {
    fn probe(&mut self) {
        self.data.push(self.signal.get().clone())
    }

    fn get_data(&self, py: Python) -> PyResult<PyObject> {
        let copy = PyArrayDyn::new(
            py,
            [
                &[self.data.len()],
                self.signal.initial_value.as_ref(py).shape(),
            ]
            .concat(),
            false,
        );
        for (i, x) in self.data.iter().enumerate() {
            copy.as_array_mut().index_axis_mut(Axis(0), i).assign(x);
        }
        Ok(copy.to_object(py))
    }
}

impl<T: TypeNum> Probe for SignalProbe<T, ScalarSignal<T>> {
    fn probe(&mut self) {
        self.data.push(*self.signal.get());
    }

    fn get_data(&self, py: Python) -> PyResult<PyObject> {
        let copy = Array::from(self.data).into_pyarray(py);
        Ok(copy.to_object(py))
    }
}

struct SignalMap {
    map_i64: Vec<Rc<ArraySignal<i64>>>,
    map_f64: Vec<Rc<ArraySignal<f64>>>,
}

#[pyclass]
struct Engine {
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
        println!("id {:?}", self.signals.map_i64[id].buffer.borrow());
        Py::from(self.signals.map_i64[id].buffer.borrow().to_pyarray(py))
    }

    fn add_probe(&mut self, target: usize) -> usize {
        self.probes
            .push(Box::new(SignalProbe::<ArrayD<f64>, ArraySignal<f64>> {
                signal: Rc::clone(&self.signals.map_f64[target]),
                data: vec![],
            }));
        self.probes.len() - 1
    }

    fn get_probe_data(&self, target: usize) -> PyResult<PyObject> {
        let gil = Python::acquire_gil();
        let py = gil.python();
        self.probes[target].get_data(py)
    }
}

#[pymodule]
/// A Python module implemented in Rust.
fn engine(py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<Engine>()?;

    Ok(())
}

#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {
        assert_eq!(2 + 2, 4);
    }
}
