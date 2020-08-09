use ndarray::{
    Array, ArrayBase, ArrayD, Data, Dimension, Ix, IxDyn, RawData, SliceInfo, SliceOrIndex,
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

pub enum ArrayRef<T: TypeNum> {
    Owned(ArrayD<T>),
    View(
        Arc<ArraySignal<T>>,
        Box<SliceInfo<Vec<SliceOrIndex>, IxDyn>>,
    ),
}

impl<T: TypeNum + Debug> Debug for ArrayRef<T> {
    fn fmt(
        &self,
        formatter: &mut std::fmt::Formatter<'_>,
    ) -> std::result::Result<(), std::fmt::Error> {
        match self {
            ArrayRef::Owned(array) => array.fmt(formatter),
            ArrayRef::View(base, slice) => match &*base.buffer.read().unwrap() {
                ArrayRef::Owned(base) => base.slice(slice.as_ref().as_ref()).fmt(formatter),
                ArrayRef::View(_, _) => formatter.write_str("transitive ArrayRef::View"),
            },
        }
    }
}

impl<T: TypeNum> ArrayRef<T> {
    pub fn assign(&mut self, src: &ArrayRef<T>) {
        match src {
            ArrayRef::Owned(src) => self.assign_array(&src),
            ArrayRef::View(src, slice) => match &*src.buffer.read().unwrap() {
                ArrayRef::Owned(base) => self.assign_array(&base.slice(slice.as_ref().as_ref())),
                ArrayRef::View(_, _) => panic!("Transitive array views are not supported."),
            },
        }
    }

    pub fn assign_array<S: RawData<Elem = T> + Data, D: Dimension>(
        &mut self,
        src: &ArrayBase<S, D>,
    ) {
        match self {
            ArrayRef::Owned(dst) => dst.assign(src),
            ArrayRef::View(dst, slice) => match &mut *dst.buffer.write().unwrap() {
                ArrayRef::Owned(base) => base.slice_mut(slice.as_ref().as_ref()).assign(src),
                ArrayRef::View(_, _) => panic!("Transitive array views are not supported."),
            },
        }
    }
}

impl<T: TypeNum> ArrayRef<T> {
    pub fn clone_array(&self) -> ArrayD<T> {
        match self {
            ArrayRef::Owned(src) => src.clone(),
            ArrayRef::View(src, slice) => match &*src.buffer.read().unwrap() {
                ArrayRef::Owned(base) => base.slice(slice.as_ref().as_ref()).to_owned(),
                ArrayRef::View(_, _) => panic!("Transitive array views are not supported."),
            },
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
            ArrayRef::View(lhs, slice) => match &mut *lhs.buffer.write().unwrap() {
                ArrayRef::Owned(base) => {
                    let mut view = base.slice_mut(slice.as_ref().as_ref());
                    view += rhs
                }
                ArrayRef::View(_, _) => panic!("Transitive array views are not supported."),
            },
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
            ArrayRef::View(rhs, slice) => match &*rhs.buffer.read().unwrap() {
                ArrayRef::Owned(base) => self * &base.slice(slice.as_ref().as_ref()),
                ArrayRef::View(_, _) => panic!("Transitive array views are not supported."),
            },
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
            ArrayRef::View(lhs, slice) => match &*lhs.buffer.read().unwrap() {
                ArrayRef::Owned(base) => &base.slice(slice.as_ref().as_ref()) * rhs,
                ArrayRef::View(_, _) => panic!("Transitive array views are not supported."),
            },
        }
    }
}

impl<T: TypeNum + PartialEq> PartialEq for ArrayRef<T> {
    fn eq(&self, rhs: &ArrayRef<T>) -> bool {
        match rhs {
            ArrayRef::Owned(rhs) => self == rhs,
            ArrayRef::View(rhs, slice) => match &*rhs.buffer.read().unwrap() {
                ArrayRef::Owned(base) => *self == base.slice(slice.as_ref().as_ref()),
                ArrayRef::View(_, _) => panic!("Transitive array views are not supported."),
            },
        }
    }
}

impl<T: TypeNum + PartialEq, S: RawData<Elem = T> + Data> PartialEq<ArrayBase<S, IxDyn>>
    for ArrayRef<T>
{
    fn eq(&self, rhs: &ArrayBase<S, IxDyn>) -> bool {
        match self {
            ArrayRef::Owned(lhs) => *lhs == *rhs,
            ArrayRef::View(lhs, slice) => match &*lhs.buffer.read().unwrap() {
                ArrayRef::Owned(base) => base.slice(slice.as_ref().as_ref()) == *rhs,
                ArrayRef::View(_, _) => panic!("Transitive array views are not supported."),
            },
        }
    }
}

#[derive(Debug)]
pub struct ArraySignal<T: TypeNum> {
    name: String,
    buffer: RwLock<ArrayRef<T>>,
    initial_value: Option<Py<PyArrayDyn<T>>>,
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
            initial_value: Some(Py::from(initial_value)),
            shape: initial_value.shape().to_vec(),
        }
    }

    pub fn new_view(
        name: String,
        base: Arc<Self>,
        slice: Box<SliceInfo<Vec<SliceOrIndex>, IxDyn>>,
    ) -> Self {
        let shape = match &*base.buffer.read().unwrap() {
            ArrayRef::Owned(base) => base.slice(slice.as_ref().as_ref()).shape().to_vec(),
            ArrayRef::View(_, _) => panic!("Transitive array views are not supported."),
        };
        ArraySignal {
            name,
            buffer: RwLock::new(ArrayRef::View(base, slice)),
            initial_value: None,
            shape,
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
        if let Some(initial_value) = &self.initial_value {
            let gil = Python::acquire_gil();
            let py = gil.python();
            self.buffer
                .write()
                .unwrap()
                .assign_array(&initial_value.as_ref(py).as_array())
        }
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
