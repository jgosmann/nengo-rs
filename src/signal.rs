use ndarray::{Array, ArrayD, Ix};
use numpy::{PyArrayDyn, TypeNum};
use pyo3::prelude::*;
use std::any::Any;
use std::cell::{Ref, RefCell, RefMut};
use std::fmt::Debug;
use std::rc::Rc;

pub trait Signal: Debug {
    fn as_any(&self) -> &dyn Any;
    fn as_any_rc(self: Rc<Self>) -> Rc<dyn Any>;
    fn name(&self) -> &String;
    fn shape(&self) -> &[Ix];
    fn reset(&self);
}

pub trait Get<T> {
    fn get(&self) -> Ref<T>;
    fn get_mut(&self) -> RefMut<T>;
}

#[derive(Debug)]
pub struct ScalarSignal<T> {
    name: String,
    value: RefCell<T>,
    initial_value: T,
}

impl<T: Copy> ScalarSignal<T> {
    pub fn new(name: String, initial_value: T) -> Self {
        ScalarSignal {
            name,
            value: RefCell::new(initial_value),
            initial_value,
        }
    }
}

impl<T: Copy + Debug + 'static> Signal for ScalarSignal<T> {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_rc(self: Rc<Self>) -> Rc<dyn Any> {
        self
    }

    fn name(&self) -> &String {
        &self.name
    }

    fn shape(&self) -> &[Ix] {
        &[]
    }

    fn reset(&self) {
        *self.value.borrow_mut() = self.initial_value;
    }
}

impl<T> Get<T> for ScalarSignal<T> {
    fn get(&self) -> Ref<T> {
        self.value.borrow()
    }

    fn get_mut(&self) -> RefMut<T> {
        self.value.borrow_mut()
    }
}

#[derive(Debug)]
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

impl<T: TypeNum + 'static> Signal for ArraySignal<T> {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_rc(self: Rc<Self>) -> Rc<dyn Any> {
        self
    }

    fn name(&self) -> &String {
        &self.name
    }

    fn shape(&self) -> &[Ix] {
        &self.shape
    }

    fn reset(&self) {
        let gil = Python::acquire_gil();
        let py = gil.python();
        self.buffer
            .borrow_mut()
            .assign(&self.initial_value.as_ref(py).as_array())
    }
}

impl<T: TypeNum> Get<ArrayD<T>> for ArraySignal<T> {
    fn get(&self) -> Ref<ArrayD<T>> {
        self.buffer.borrow()
    }

    fn get_mut(&self) -> RefMut<ArrayD<T>> {
        self.buffer.borrow_mut()
    }
}
