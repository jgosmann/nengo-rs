use ndarray::{Array, ArrayD, Ix};
use numpy::{PyArrayDyn, TypeNum};
use pyo3::prelude::*;
use std::any::Any;
use std::fmt::Debug;
use std::sync::Arc;
use std::sync::RwLock;
use std::sync::RwLockReadGuard;
use std::sync::RwLockWriteGuard;

type AnySignal = dyn Any + Send + Sync;

pub trait Signal: Debug {
    fn as_any(&self) -> &dyn Any;
    fn as_any_arc(self: Arc<Self>) -> Arc<AnySignal>;
    fn name(&self) -> &String;
    fn shape(&self) -> &[Ix];
    fn reset(&self);
}

pub trait SignalAccess<T> {
    fn read(&self) -> RwLockReadGuard<T>;
    fn write(&self) -> RwLockWriteGuard<T>;
}

#[derive(Debug)]
pub struct ScalarSignal<T> {
    name: String,
    value: RwLock<T>,
    initial_value: T,
}

impl<T: Copy> ScalarSignal<T> {
    pub fn new(name: String, initial_value: T) -> Self {
        ScalarSignal {
            name,
            value: RwLock::new(initial_value),
            initial_value,
        }
    }
}

impl<T: Copy + Send + Sync + Debug + 'static> Signal for ScalarSignal<T> {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_arc(self: Arc<Self>) -> Arc<AnySignal> {
        self
    }

    fn name(&self) -> &String {
        &self.name
    }

    fn shape(&self) -> &[Ix] {
        &[]
    }

    fn reset(&self) {
        *self.value.write().unwrap() = self.initial_value;
    }
}

impl<T> SignalAccess<T> for ScalarSignal<T> {
    fn read(&self) -> RwLockReadGuard<T> {
        self.value.read().unwrap()
    }

    fn write(&self) -> RwLockWriteGuard<T> {
        self.value.write().unwrap()
    }
}

#[derive(Debug)]
pub struct ArraySignal<T: TypeNum> {
    name: String,
    buffer: RwLock<ArrayD<T>>,
    initial_value: Py<PyArrayDyn<T>>,
    shape: Vec<Ix>,
}

impl<T: TypeNum> ArraySignal<T> {
    pub fn new(name: String, initial_value: &PyArrayDyn<T>) -> Self {
        ArraySignal {
            name,
            buffer: RwLock::new(unsafe {
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

impl<T: TypeNum + Send + Sync + 'static> Signal for ArraySignal<T> {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_arc(self: Arc<Self>) -> Arc<dyn Any + Send + Sync> {
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
            .write()
            .unwrap()
            .assign(&self.initial_value.as_ref(py).as_array())
    }
}

impl<T: TypeNum> SignalAccess<ArrayD<T>> for ArraySignal<T> {
    fn read(&self) -> RwLockReadGuard<ArrayD<T>> {
        self.buffer.read().unwrap()
    }

    fn write(&self) -> RwLockWriteGuard<ArrayD<T>> {
        self.buffer.write().unwrap()
    }
}
