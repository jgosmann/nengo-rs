use ndarray::{Array, ArrayD, Ix};
use numpy::{PyArrayDyn, TypeNum};
use pyo3::prelude::*;
use std::borrow::{Borrow, BorrowMut};
use std::cell::RefCell;

pub trait Signal {
    fn name(&self) -> &String;
    fn shape(&self) -> &[Ix];
    fn reset(&mut self);
}

pub trait Get<T> {
    fn get(&self) -> &T;
    fn get_mut(&self) -> &mut T;
}

pub struct ScalarSignal<T> {
    name: String,
    value: T,
    initial_value: T,
}

impl<T> ScalarSignal<T> {
    pub fn new(name: String, initial_value: T) -> Self {
        ScalarSignal {
            name,
            value: initial_value,
            initial_value,
        }
    }
}

impl<T> Signal for ScalarSignal<T> {
    fn name(&self) -> &String {
        &self.name
    }

    fn shape(&self) -> &[Ix] {
        &[]
    }

    fn reset(&mut self) {
        self.value = self.initial_value;
    }
}

impl<T> Get<T> for ScalarSignal<T> {
    fn get(&self) -> &T {
        &self.value
    }

    fn get_mut(&self) -> &mut T {
        &mut self.value
    }
}

pub struct ArraySignal<T: TypeNum> {
    name: String,
    buffer: RefCell<ArrayD<T>>,
    initial_value: Py<PyArrayDyn<T>>,
    shape: Vec<Ix>,
}

impl<T: TypeNum> ArraySignal<T> {
    pub fn new(name: String, initial_value: &PyArrayDyn<T>) -> Self {
        ArraySignal {
            name,
            buffer: RefCell::new(unsafe {
                Array::uninitialized(match initial_value.shape() {
                    [] => &[1],
                    x => x,
                })
            }),
            initial_value: Py::from(initial_value),
            shape: initial_value.shape().to_vec(),
        }
    }
}

impl<T: TypeNum> Signal for ArraySignal<T> {
    fn name(&self) -> &String {
        &self.name
    }

    fn shape(&self) -> &[Ix] {
        &self.shape
    }

    fn reset(&mut self) {
        let gil = Python::acquire_gil();
        let py = gil.python();
        self.get_mut()
            .assign(&self.initial_value.as_ref(py).as_array())
    }
}

impl<T: TypeNum> Get<ArrayD<T>> for ArraySignal<T> {
    fn get(&self) -> &ArrayD<T> {
        &self.buffer.borrow()
    }

    fn get_mut(&self) -> &mut ArrayD<T> {
        &mut self.buffer.borrow_mut()
    }
}
