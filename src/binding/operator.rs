use crate::binding::signal::RsSignal;
use crate::binding::Wrapper;
use crate::operator::{CopyOp, ElementwiseInc, OperatorNode, Reset, TimeUpdate};
use crate::signal::ArraySignal;
use ndarray::ArrayD;
use numpy::PyArrayDyn;
use pyo3::prelude::*;
use std::marker::PhantomData;
use std::sync::Arc;

#[pyclass]
pub struct RsOperator {
    node: Arc<OperatorNode>,
}

impl Wrapper<Arc<OperatorNode>> for RsOperator {
    fn get(&self) -> &Arc<OperatorNode> {
        &self.node
    }
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
                        target: target.extract_signal("target")?,
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
                        step_target: step_target.extract_signal("step_target")?,
                        time_target: time_target.extract_signal("time_target")?,
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
