use crate::binding::signal::PySignal;
use crate::binding::Wrapper;
use crate::operator::{CopyOp, ElementwiseInc, OperatorNode, Reset, TimeUpdate};
use crate::signal::ArraySignal;
use ndarray::ArrayD;
use numpy::PyArrayDyn;
use pyo3::prelude::*;
use std::marker::PhantomData;
use std::sync::Arc;

#[pyclass(name=Operator)]
pub struct PyOperator {
    node: Arc<OperatorNode>,
}

impl Wrapper<Arc<OperatorNode>> for PyOperator {
    fn get(&self) -> &Arc<OperatorNode> {
        &self.node
    }
}

#[pyclass(extends=PyOperator, name=Reset)]
pub struct PyReset {}

#[pymethods]
impl PyReset {
    #[new]
    fn new(
        value: &PyAny,
        target: &PySignal,
        dependencies: Vec<usize>,
    ) -> PyResult<(Self, PyOperator)> {
        let value: &PyArrayDyn<f64> = value.extract()?;
        let value = value.to_owned_array();
        Ok((
            Self {},
            PyOperator {
                node: Arc::new(OperatorNode {
                    operator: Box::new(Reset::<ArrayD<f64>, ArraySignal<f64>> {
                        value,
                        target: target.extract_signal("target")?,
                    }),
                    dependencies,
                }),
            },
        ))
    }
}

#[pyclass(extends=PyOperator, name=TimeUpdate)]
pub struct PyTimeUpdate {}

#[pymethods]
impl PyTimeUpdate {
    #[new]
    fn new(
        dt: f64,
        step_target: &PySignal,
        time_target: &PySignal,
        dependencies: Vec<usize>,
    ) -> PyResult<(Self, PyOperator)> {
        Ok((
            Self {},
            PyOperator {
                node: Arc::new(OperatorNode {
                    operator: Box::new(TimeUpdate::<f64, u64> {
                        dt,
                        step_target: step_target.extract_signal("step_target")?,
                        time_target: time_target.extract_signal("time_target")?,
                    }),
                    dependencies,
                }),
            },
        ))
    }
}

#[pyclass(extends=PyOperator, name=ElementwiseInc)]
pub struct PyElementwiseInc {}

#[pymethods]
impl PyElementwiseInc {
    #[new]
    fn new(
        target: &PySignal,
        left: &PySignal,
        right: &PySignal,
        dependencies: Vec<usize>,
    ) -> PyResult<(Self, PyOperator)> {
        Ok((
            Self {},
            PyOperator {
                node: Arc::new(OperatorNode {
                    operator: Box::new(ElementwiseInc::<f64> {
                        target: target.extract_signal("target")?,
                        left: left.extract_signal("left")?,
                        right: right.extract_signal("right")?,
                    }),
                    dependencies,
                }),
            },
        ))
    }
}

#[pyclass(extends=PyOperator, name=Copy)]
pub struct PyCopy {}

#[pymethods]
impl PyCopy {
    #[new]
    fn new(
        src: &PySignal,
        dst: &PySignal,
        dependencies: Vec<usize>,
    ) -> PyResult<(Self, PyOperator)> {
        Ok((
            Self {},
            PyOperator {
                node: Arc::new(OperatorNode {
                    operator: Box::new(CopyOp::<ArrayD<f64>, ArraySignal<f64>> {
                        src: src.extract_signal("src")?,
                        dst: dst.extract_signal("dst")?,
                        data_type: PhantomData,
                    }),
                    dependencies,
                }),
            },
        ))
    }
}