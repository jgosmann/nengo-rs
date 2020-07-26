use ndarray::{
    Array, ArrayBase, ArrayD, Data, Dimension, Ix, IxDyn, RawData, SliceInfo, SliceOrIndex,
    StrideShape,
};
use numpy::{PyArrayDyn, TypeNum};
use pyo3::prelude::*;
use std::any::Any;
use std::fmt::Debug;
use std::ops::{AddAssign, Deref, DerefMut, Mul};
use std::sync::Arc;
use std::sync::RwLock;

pub type AnySignal = dyn Any + Send + Sync;

pub trait Signal: Debug {
    fn as_any(&self) -> &dyn Any;
    fn as_any_arc(self: Arc<Self>) -> Arc<AnySignal>;
    fn name(&self) -> &String;
    fn shape(&self) -> &[Ix];
    fn reset(&self);
}

pub trait SignalAccess<T> {
    fn read<'a>(&'a self) -> Box<dyn Deref<Target = T> + 'a>;
    fn write<'a>(&'a self) -> Box<dyn DerefMut<Target = T> + 'a>;
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
    fn read<'a>(&'a self) -> Box<dyn Deref<Target = T> + 'a> {
        Box::new(self.value.read().unwrap())
    }

    fn write<'a>(&'a self) -> Box<dyn DerefMut<Target = T> + 'a> {
        Box::new(self.value.write().unwrap())
    }
}

#[derive(Debug)]
pub enum ArrayRef<T: TypeNum> {
    Owned(ArrayD<T>),
    View(Arc<ArraySignal<T>>, Box<SliceInfo<[SliceOrIndex], IxDyn>>),
}

impl<T: TypeNum> ArrayRef<T> {
    pub fn assign(&mut self, src: &ArrayRef<T>) {
        match src {
            ArrayRef::Owned(src) => self.assign_array(&src),
            ArrayRef::View(src, slice) => {
                if let ArrayRef::Owned(base) = &*src.buffer.read().unwrap() {
                    self.assign_array(&base.slice(&*slice));
                }
                // FIXME error?
            }
        }
    }

    pub fn assign_array<S: RawData<Elem = T> + Data, D: Dimension>(
        &mut self,
        src: &ArrayBase<S, D>,
    ) {
        match self {
            ArrayRef::Owned(dst) => dst.assign(src),
            ArrayRef::View(dst, slice) => {
                if let ArrayRef::Owned(base) = &mut *dst.buffer.write().unwrap() {
                    base.slice_mut(&*slice).assign(src);
                }
                // FIXME error?
            }
        }
    }
}

impl<T: TypeNum> ArrayRef<T> {
    pub fn clone_array(&self) -> ArrayD<T> {
        match self {
            ArrayRef::Owned(src) => src.clone(),
            ArrayRef::View(src, slice) => {
                if let ArrayRef::Owned(base) = &*src.buffer.read().unwrap() {
                    base.slice(&*slice).to_owned()
                } else {
                    panic!("Base must be owned."); // FIXME error?
                }
            }
        }
    }
}

impl<T, S> AddAssign<&ArrayBase<S, IxDyn>> for ArrayRef<T>
where
    T: TypeNum + AddAssign<T> + Clone,
    S: RawData<Elem = T> + Data,
{
    fn add_assign(&mut self, rhs: &ArrayBase<S, IxDyn>) {
        match self {
            ArrayRef::Owned(lhs) => *lhs += rhs,
            ArrayRef::View(lhs, slice) => {
                if let ArrayRef::Owned(base) = &mut *lhs.buffer.write().unwrap() {
                    let mut view = base.slice_mut(&*slice);
                    view += rhs
                } else {
                    panic!("Base must be owned."); // FIXME error
                }
            }
        }
    }
}

impl<T> Mul<&ArrayRef<T>> for &ArrayRef<T>
where
    T: TypeNum + Mul<T, Output = T> + Clone,
{
    type Output = ArrayD<T>;

    fn mul(self, rhs: &ArrayRef<T>) -> Self::Output {
        match rhs {
            ArrayRef::Owned(rhs) => self * rhs,
            ArrayRef::View(rhs, slice) => {
                if let ArrayRef::Owned(base) = &*rhs.buffer.read().unwrap() {
                    self * &base.slice(&*slice)
                } else {
                    panic!("Base must be owned."); // FIXME error
                }
            }
        }
    }
}

impl<T, S> Mul<&ArrayBase<S, IxDyn>> for &ArrayRef<T>
where
    T: TypeNum + Mul<T, Output = T> + Clone,
    S: RawData<Elem = T> + Data,
{
    type Output = ArrayD<T>;

    fn mul(self, rhs: &ArrayBase<S, IxDyn>) -> Self::Output {
        match self {
            ArrayRef::Owned(lhs) => lhs * rhs,
            ArrayRef::View(lhs, slice) => {
                if let ArrayRef::Owned(base) = &*lhs.buffer.read().unwrap() {
                    &base.slice(&*slice) * rhs
                } else {
                    panic!("Base must be owned."); // FIXME error
                }
            }
        }
    }
}

impl<T: TypeNum + PartialEq> PartialEq for ArrayRef<T> {
    fn eq(&self, rhs: &ArrayRef<T>) -> bool {
        match rhs {
            ArrayRef::Owned(rhs) => self == rhs,
            ArrayRef::View(rhs, slice) => {
                if let ArrayRef::Owned(base) = &*rhs.buffer.read().unwrap() {
                    *self == base.slice(&*slice)
                } else {
                    panic!("Base must be owned."); // FIXME error
                }
            }
        }
    }
}

impl<T: TypeNum + PartialEq, S: RawData<Elem = T> + Data> PartialEq<ArrayBase<S, IxDyn>>
    for ArrayRef<T>
{
    fn eq(&self, rhs: &ArrayBase<S, IxDyn>) -> bool {
        match self {
            ArrayRef::Owned(lhs) => *lhs == *rhs,
            ArrayRef::View(lhs, slice) => {
                if let ArrayRef::Owned(base) = &*lhs.buffer.read().unwrap() {
                    base.slice(&*slice) == *rhs
                } else {
                    panic!("Base must be owned."); // FIXME error
                }
            }
        }
    }
}

#[derive(Debug)]
pub struct ArraySignal<T: TypeNum> {
    name: String,
    buffer: RwLock<ArrayRef<T>>,
    initial_value: Py<PyArrayDyn<T>>,
    shape: Vec<Ix>,
}

impl<T: TypeNum> ArraySignal<T> {
    pub fn new(name: String, initial_value: &PyArrayDyn<T>) -> Self {
        ArraySignal {
            name,
            buffer: RwLock::new(ArrayRef::Owned(unsafe {
                Array::uninitialized(match initial_value.shape() {
                    [] => &[1],
                    x => x,
                })
            })),
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
            .assign_array(&self.initial_value.as_ref(py).as_array())
    }
}

impl<T: TypeNum> SignalAccess<ArrayRef<T>> for ArraySignal<T> {
    fn read<'a>(&'a self) -> Box<dyn Deref<Target = ArrayRef<T>> + 'a> {
        Box::new(self.buffer.read().unwrap())
    }

    fn write<'a>(&'a self) -> Box<dyn DerefMut<Target = ArrayRef<T>> + 'a> {
        Box::new(self.buffer.write().unwrap())
    }
}

#[derive(Debug)]
pub struct ArrayViewSignal<T: TypeNum> {
    name: String,
    base: Arc<ArraySignal<T>>,
    shape: StrideShape<Ix>,
}
