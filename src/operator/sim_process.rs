use crate::operator::Operator;
use crate::signal::{ArraySignal, ScalarSignal, SignalAccess};
use numpy::Element;
use numpy::PyArrayDyn;
use pyo3::prelude::*;
use pyo3::types::{PyFloat, PyTuple};
use std::ops::AddAssign;
use std::sync::Arc;

pub struct SimProcess<T>
where
    T: Element,
{
    pub mode_inc: bool,
    pub t: Arc<ScalarSignal<f64>>,
    pub input: Option<Arc<ArraySignal<T>>>,
    pub output: Arc<ArraySignal<T>>,
    pub step_fn: PyObject,
}

impl<T> Operator for SimProcess<T>
where
    T: Element + AddAssign<T>,
{
    fn step(&self) {
        let gil = Python::acquire_gil();
        let py = gil.python();

        let t: &PyAny = PyFloat::new(py, **self.t.read());
        let args = PyTuple::new(
            py,
            match &self.input {
                Some(input) => vec![t, input.read().to_py_array(py)],
                None => vec![t],
            },
        );

        let result = &self
            .step_fn
            .as_ref(py)
            .call(args, None)
            .expect("Call to Python function failed.")
            .extract::<Option<&PyArrayDyn<T>>>()
            .expect("Python function failed.");
        if let Some(result) = result {
            let mut output = self.output.write();
            if self.mode_inc {
                **output += &result.readonly().as_array();
            } else {
                output.assign_array(&result.readonly().as_array());
            }
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::signal::{ArrayRef, Signal};
    use crate::venv::activate_venv;
    use ndarray::prelude::*;
    use pyo3::types::IntoPyDict;
    use pyo3::Python;

    #[test]
    fn it_calls_the_function_without_input() {
        let gil = Python::acquire_gil();
        let py = gil.python();
        activate_venv(py);

        let numpy = PyModule::import(py, "numpy").unwrap();
        let locals = [("np", numpy.to_object(py))].into_py_dict(py);

        let op = SimProcess::<f64> {
            mode_inc: false,
            t: Arc::new(ScalarSignal::new(String::from("t"), 1.)),
            input: None,
            output: Arc::new(ArraySignal::new(
                String::from("output"),
                PyArrayDyn::from_array(py, &array![0.].into_dyn()),
            )),
            step_fn: py
                .eval("lambda t: np.array([t])", Some(locals), None)
                .unwrap()
                .into(),
        };
        op.t.reset();
        op.output.reset();

        op.step();

        assert_eq!(**op.output.read(), ArrayRef::Owned(array![1.].into_dyn()));
    }

    #[test]
    fn it_calls_the_function_with_input() {
        let gil = Python::acquire_gil();
        let py = gil.python();
        activate_venv(py);

        let numpy = PyModule::import(py, "numpy").unwrap();
        let locals = [("np", numpy.to_object(py))].into_py_dict(py);

        let op = SimProcess::<f64> {
            mode_inc: false,
            t: Arc::new(ScalarSignal::new(String::from("t"), 1.)),
            input: Some(Arc::new(ArraySignal::new(
                String::from("input"),
                PyArrayDyn::from_array(py, &array![2.].into_dyn()),
            ))),
            output: Arc::new(ArraySignal::new(
                String::from("output"),
                PyArrayDyn::from_array(py, &array![0.].into_dyn()),
            )),
            step_fn: py
                .eval("lambda t, input: np.array(input)", Some(locals), None)
                .unwrap()
                .into(),
        };
        op.t.reset();
        op.input.as_ref().map(|input| input.reset());
        op.output.reset();

        op.step();

        assert_eq!(**op.output.read(), ArrayRef::Owned(array![2.].into_dyn()));
    }

    #[test]
    fn it_calls_the_function_and_increments_the_output() {
        let gil = Python::acquire_gil();
        let py = gil.python();
        activate_venv(py);

        let numpy = PyModule::import(py, "numpy").unwrap();
        let locals = [("np", numpy.to_object(py))].into_py_dict(py);

        let op = SimProcess::<f64> {
            mode_inc: true,
            t: Arc::new(ScalarSignal::new(String::from("t"), 1.)),
            input: None,
            output: Arc::new(ArraySignal::new(
                String::from("output"),
                PyArrayDyn::from_array(py, &array![1.].into_dyn()),
            )),
            step_fn: py
                .eval("lambda t: np.array([t])", Some(locals), None)
                .unwrap()
                .into(),
        };
        op.t.reset();
        op.output.reset();

        op.step();

        assert_eq!(**op.output.read(), ArrayRef::Owned(array![2.].into_dyn()));
    }
}
