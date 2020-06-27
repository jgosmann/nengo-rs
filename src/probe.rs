use crate::signal::{ArraySignal, Get, ScalarSignal, Signal};
use ndarray::{ArrayD, Axis};
use numpy::{PyArray, PyArrayDyn, TypeNum};
use pyo3::prelude::*;
use std::any::Any;
use std::rc::Rc;

pub trait Probe {
    fn as_any(&self) -> &dyn Any;
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

impl<T: TypeNum + 'static> Probe for SignalProbe<ArrayD<T>, ArraySignal<T>> {
    fn as_any(&self) -> &dyn Any {
        self
    }

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

impl<T: TypeNum + 'static> Probe for SignalProbe<T, ScalarSignal<T>> {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn probe(&mut self) {
        self.data.push(*self.signal.get());
    }

    fn get_data(&self, py: Python) -> PyResult<PyObject> {
        Ok(PyArray::from_slice(py, &self.data).to_object(py))
    }
}
