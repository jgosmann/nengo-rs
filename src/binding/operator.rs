use crate::binding::signal::PySignal;
use crate::binding::Wrapper;
use crate::operator;
use crate::operator::OperatorNode;
use crate::signal::ArraySignal;
use ndarray::ArrayD;
use numpy::PyArrayDyn;
use pyo3::prelude::*;
use pyo3::types::PyList;
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
        {
            $(args: ($($aname:ident : $atype:ty),*),)?
            $(signals: [$($sig:ident),*],)?
            $(optionals: [$($optsig:ident),*],)?
        }, {$($fname:ident : $expr:expr),*}
    ) => {
        #[pymethods]
        impl $name {
            #[new]
            fn new(
                $($($aname: $atype,)*)?
                $($(
                    $sig : &PySignal,
                )*)?
                $($(
                    $optsig : Option<&PySignal>,
                )*)?
                dependencies: Vec<usize>,
            ) -> PyResult<(Self, PyOperator)> {
                Ok((
                    Self {},
                    PyOperator {
                        node: Arc::new(OperatorNode {
                            operator: Box::new(operator::$op_type$(::<$($op_typearg,)*>)? {
                                $($(
                                    $sig : $sig.extract_signal(stringify!($sig))?,
                                )*)?
                                $($(
                                    $optsig: match $optsig {
                                        Some(sig) => Some(sig.extract_signal(stringify!($optsig))?),
                                        None => None
                                    },
                                )*)?
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
    {
        args: (value: &PyAny),
        signals: [target],
    },
    {value: value.extract::<&PyArrayDyn<f64>>()?.to_owned_array()}
);

#[pyclass(extends=PyOperator, name=TimeUpdate)]
pub struct PyTimeUpdate {}

bind_op!(
    PyTimeUpdate: TimeUpdate<f64, u64>,
    {
        args: (dt: f64),
        signals: [step_target, time_target],
    },
    { dt: dt }
);

#[pyclass(extends=PyOperator, name=ElementwiseInc)]
pub struct PyElementwiseInc {}

bind_op!(
    PyElementwiseInc: ElementwiseInc<f64>,
    {signals: [target, left, right],},
    {}
);

#[pyclass(extends=PyOperator, name=Copy)]
pub struct PyCopy {}

bind_op!(
    PyCopy: CopyOp<ArrayD<f64>, ArraySignal<f64>>,
    {signals: [src, dst],},
    { data_type: PhantomData }
);

#[pyclass(extends=PyOperator, name=DotInc)]
pub struct PyDotInc {}

bind_op!(
    PyDotInc: DotInc<f64>,
    {signals: [target, left, right],},
    {}
);

#[pyclass(extends=PyOperator, name=SimNeurons)]
pub struct PySimNeurons {}

bind_op!(
    PySimNeurons: SimNeurons<f64>,
    {
        args: (dt: f64, step_fn: &PyAny, state: &PyList),
        signals: [input_current, output],
    },
    {
        dt: dt,
        step_fn: step_fn.into(),
        state: state.into()
    }
);

#[pyclass(extends=PyOperator, name=SimProcess)]
pub struct PySimProcess {}

bind_op!(
    PySimProcess: SimProcess<f64>,
    {
        args: (mode_inc: bool, step_fn: &PyAny),
        signals: [t, output],
        optionals: [input],
    },
    {
        mode_inc: mode_inc,
        step_fn: step_fn.into()
    }
);

#[pyclass(extends=PyOperator, name=SimPyFunc)]
pub struct PySimPyFunc {}

bind_op!(
    PySimPyFunc: SimPyFunc<f64>,
    {
        args: (py_fn: &PyAny),
        signals: [output],
        optionals: [t, x],
    },
    {py_fn: py_fn.into()}
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
        m.add_class::<PyDotInc>()?;
        m.add_class::<PyElementwiseInc>()?;
        m.add_class::<PyReset>()?;
        m.add_class::<PySimNeurons>()?;
        m.add_class::<PySimProcess>()?;
        m.add_class::<PySimPyFunc>()?;
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
    fn can_instantiate_dot_inc() {
        can_instantiate(&format!(
            "o.DotInc({}, {}, {}, [0])",
            DUMMY_SIGNAL_CONSTRUCTOR, DUMMY_SIGNAL_CONSTRUCTOR, DUMMY_SIGNAL_CONSTRUCTOR
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
    fn can_instantiate_sim_neurons() {
        can_instantiate(&format!(
            "o.SimNeurons(0.001, lambda dt, J, output: None, [], {}, {}, [0])",
            DUMMY_SIGNAL_CONSTRUCTOR, DUMMY_SIGNAL_CONSTRUCTOR
        ))
        .unwrap();
    }

    #[test]
    fn can_instantiate_sim_process() {
        can_instantiate(&format!(
            "o.SimProcess(False, lambda t, input: None, o.SignalF64('time', 0.), {}, {}, [0])",
            DUMMY_SIGNAL_CONSTRUCTOR, DUMMY_SIGNAL_CONSTRUCTOR
        ))
        .unwrap();
    }

    #[test]
    fn can_instantiate_sim_process_without_optional_signals() {
        can_instantiate(&format!(
            "o.SimProcess(False, lambda t: None, o.SignalF64('time', 0.), {}, None, [0])",
            DUMMY_SIGNAL_CONSTRUCTOR
        ))
        .unwrap();
    }

    #[test]
    fn can_instantiate_sim_py_func() {
        can_instantiate(&format!(
            "o.SimPyFunc(lambda t, x: None, {}, o.SignalF64('time', 0.), {}, [0])",
            DUMMY_SIGNAL_CONSTRUCTOR, DUMMY_SIGNAL_CONSTRUCTOR,
        ))
        .unwrap();
    }

    #[test]
    fn can_instantiate_sim_py_func_without_optional_signals() {
        can_instantiate(&format!(
            "o.SimPyFunc(lambda t, x: None, {}, None, None, [0])",
            DUMMY_SIGNAL_CONSTRUCTOR,
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
