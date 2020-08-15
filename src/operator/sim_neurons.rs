use crate::operator::Operator;
use crate::signal::{ArraySignal, Signal, SignalAccess};
use numpy::Element;
use numpy::PyArrayDyn;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use pyo3::types::PyTuple;
use std::fmt::Debug;
use std::sync::Arc;

// TODO: implement support for probing state signals
pub struct SimNeurons<T>
where
    T: Element,
{
    pub dt: T,
    pub input_current: Arc<ArraySignal<T>>,
    pub output: Arc<ArraySignal<T>>,
    pub state: Py<PyDict>,
    pub step_fn: PyObject,
}

impl<T> Operator for SimNeurons<T>
where
    T: Element + Copy + Debug + Send + Sync + ToPyObject + 'static,
{
    fn step(&self) {
        let gil = Python::acquire_gil();
        let py = gil.python();

        let dt = self.dt.to_object(py);
        let input_current = self.input_current.read().to_py_array(py);
        let output = PyArrayDyn::new(py, self.output.shape(), false);
        let args = PyTuple::new(
            py,
            vec![dt, input_current.to_object(py), output.to_object(py)],
        );

        &self
            .step_fn
            .as_ref(py)
            .call(args, Some(self.state.as_ref(py)))
            .expect("Call to Python function failed.");
        let mut output_sig = self.output.write();
        output_sig.assign_array(&output.readonly().as_array());
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::signal::{ArrayRef, Signal};
    use crate::venv::activate_venv;
    use ndarray::prelude::*;
    use pyo3::Python;

    #[test]
    fn it_calls_the_step_function_and_copies_the_output() {
        let gil = Python::acquire_gil();
        let py = gil.python();
        activate_venv(py);

        let step_module = PyModule::from_code(
            py,
            r#"
def step(dt, J, output):
    output[:] = dt * J
        "#,
            "step.py",
            "step",
        )
        .unwrap();

        let op = SimNeurons::<f64> {
            dt: 2.,
            input_current: Arc::new(ArraySignal::new(
                String::from("input_current"),
                PyArrayDyn::from_array(py, &array![1.].into_dimensionality::<IxDyn>().unwrap()),
            )),
            output: Arc::new(ArraySignal::new(
                String::from("output"),
                PyArrayDyn::from_array(py, &array![0.].into_dimensionality::<IxDyn>().unwrap()),
            )),
            state: PyDict::new(py).into(),
            step_fn: step_module.getattr("step").unwrap().into(),
        };
        op.input_current.reset();
        op.output.reset();

        op.step();

        assert_eq!(
            **op.output.read(),
            ArrayRef::Owned(array![2.].into_dimensionality::<IxDyn>().unwrap())
        );
    }

    #[test]
    fn it_calls_the_step_function_with_the_state() {
        let gil = Python::acquire_gil();
        let py = gil.python();
        activate_venv(py);

        let step_module = PyModule::from_code(
            py,
            r#"
def step(dt, J, output, state_var):
    output[:] = dt * J + state_var
        "#,
            "step.py",
            "step",
        )
        .unwrap();

        let state = PyDict::new(py);
        state.set_item("state_var", 4.).unwrap();

        let op = SimNeurons::<f64> {
            dt: 2.,
            input_current: Arc::new(ArraySignal::new(
                String::from("input_current"),
                PyArrayDyn::from_array(py, &array![1.].into_dimensionality::<IxDyn>().unwrap()),
            )),
            output: Arc::new(ArraySignal::new(
                String::from("output"),
                PyArrayDyn::from_array(py, &array![0.].into_dimensionality::<IxDyn>().unwrap()),
            )),
            state: state.into(),
            step_fn: step_module.getattr("step").unwrap().into(),
        };
        op.input_current.reset();
        op.output.reset();

        op.step();

        assert_eq!(
            **op.output.read(),
            ArrayRef::Owned(array![6.].into_dimensionality::<IxDyn>().unwrap())
        );
    }
}
