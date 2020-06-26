use crate::signal::{ArraySignal, Get, ScalarSignal, Signal};
use ndarray::{Array, ArrayD, Axis};
use numpy::convert::IntoPyArray;
use numpy::{PyArrayDyn, TypeNum};
use pyo3::prelude::*;
use std::rc::Rc;

pub trait Probe {
    fn probe(&mut self);
    fn get_data(&self, py: Python) -> PyResult<PyObject>;
}

pub struct SignalProbe<T, S: Signal> {
    signal: Rc<S>,
    data: Vec<T>,
}

impl<T, S: Signal> SignalProbe<T, S> {
    pub fn new(signal: &Rc<S>) -> Self {
        SignalProbe::<T, S> {
            signal: Rc::clone(signal),
            data: vec![],
        }
    }
}

impl<T: TypeNum> Probe for SignalProbe<ArrayD<T>, ArraySignal<T>> {
    fn probe(&mut self) {
        self.data.push(self.signal.get().clone())
    }

    fn get_data(&self, py: Python) -> PyResult<PyObject> {
        let copy = PyArrayDyn::new(
            py,
            [&[self.data.len()], self.signal.shape()].concat(),
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
