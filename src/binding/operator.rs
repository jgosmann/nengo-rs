use crate::binding::signal::PySignal;
use crate::binding::Wrapper;
use crate::operator;
use crate::operator::OperatorNode;
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

macro_rules! bind_op {
    (
        $name:ident: $op_type:ident $(< $($op_typearg:ty),+ >)?,
        ($(arg $aname:ident : $atype:ty,)*
        $(sig $sig:ident),*), {$($fname:ident : $expr:expr),*}
    ) => {
        #[pymethods]
        impl $name {
            #[new]
            fn new(
                $($aname: $atype,)*
                $(
                    $sig : &PySignal,
                )*
                dependencies: Vec<usize>,
            ) -> PyResult<(Self, PyOperator)> {
                Ok((
                    Self {},
                    PyOperator {
                        node: Arc::new(OperatorNode {
                            operator: Box::new(operator::$op_type$(::<$($op_typearg,)*>)? {
                                $(
                                    $sig : $sig.extract_signal("$sig")?,
                                )*
                                $($fname: $expr,)*
                            }),
                            dependencies,
                        }),
                    },
                ))
            }
        }
    };
}

#[pyclass(extends=PyOperator, name=Reset)]
pub struct PyReset {}

bind_op!(
    PyReset: Reset<ArrayD<f64>, ArraySignal<f64>>,
    (arg value: &PyAny, sig target),
    {value: value.extract::<&PyArrayDyn<f64>>()?.to_owned_array()}
);

#[pyclass(extends=PyOperator, name=TimeUpdate)]
pub struct PyTimeUpdate {}

bind_op!(
    PyTimeUpdate: TimeUpdate<f64, u64>,
    (arg dt: f64, sig step_target, sig time_target),
    { dt: dt }
);

#[pyclass(extends=PyOperator, name=ElementwiseInc)]
pub struct PyElementwiseInc {}

bind_op!(
    PyElementwiseInc: ElementwiseInc<f64>,
    (sig target, sig left, sig right),
    {}
);

#[pyclass(extends=PyOperator, name=Copy)]
pub struct PyCopy {}

bind_op!(
    PyCopy: CopyOp<ArrayD<f64>, ArraySignal<f64>>,
    (sig src, sig dst),
    { data_type: PhantomData }
);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::binding::signal::{PySignalArrayF64, PySignalF64, PySignalU64};
    use crate::venv::activate_venv;
    use pyo3::{types::IntoPyDict, wrap_pymodule, ToPyObject};

    #[pymodule]
    fn operator(_py: Python, m: &PyModule) -> PyResult<()> {
        m.add_class::<PyCopy>()?;
        m.add_class::<PyElementwiseInc>()?;
        m.add_class::<PyReset>()?;
        m.add_class::<PyTimeUpdate>()?;

        m.add_class::<PySignalF64>()?;
        m.add_class::<PySignalU64>()?;
        m.add_class::<PySignalArrayF64>()?;

        Ok(())
    }

    const DUMMY_SIGNAL_CONSTRUCTOR: &str =
        "o.SignalArrayF64(nengo.builder.signal.Signal(np.zeros(1)))";

    fn can_instantiate(expr: &str) -> PyResult<()> {
        let gil = Python::acquire_gil();
        let py = gil.python();
        activate_venv(py);
        let nengo = PyModule::import(py, "nengo")?;
        let numpy = PyModule::import(py, "numpy")?;
        let operator_module = wrap_pymodule!(operator)(py);
        let locals = [
            ("nengo", nengo.to_object(py)),
            ("np", numpy.to_object(py)),
            ("o", operator_module),
        ]
        .into_py_dict(py);

        let py_operator = py.eval(expr, None, Some(locals))?;
        py_operator.extract::<&PyCell<PyOperator>>()?;
        Ok(())
    }

    #[test]
    fn can_instantiate_copy() {
        can_instantiate(&format!(
            "o.Copy({}, {}, [0])",
            DUMMY_SIGNAL_CONSTRUCTOR, DUMMY_SIGNAL_CONSTRUCTOR
        ))
        .unwrap();
    }

    #[test]
    fn can_instantiate_elementwise_inc() {
        can_instantiate(&format!(
            "o.ElementwiseInc({}, {}, {}, [0])",
            DUMMY_SIGNAL_CONSTRUCTOR, DUMMY_SIGNAL_CONSTRUCTOR, DUMMY_SIGNAL_CONSTRUCTOR
        ))
        .unwrap();
    }

    #[test]
    fn can_instantiate_reset() {
        can_instantiate(&format!(
            "o.Reset(np.zeros(1), {}, [0])",
            DUMMY_SIGNAL_CONSTRUCTOR
        ))
        .unwrap();
    }

    #[test]
    fn can_instantiate_time_update() {
        can_instantiate(
            "o.TimeUpdate(0.001, o.SignalU64('step', 0), o.SignalF64('time', 0.), [0])",
        )
        .unwrap();
    }
}
