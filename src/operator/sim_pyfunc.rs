use crate::operator::Operator;
use crate::signal::{ArraySignal, ScalarSignal, SignalAccess};
use numpy::Element;
use numpy::PyArrayDyn;
use pyo3::prelude::*;
use pyo3::types::{PyFloat, PyTuple};
use std::sync::Arc;

pub struct SimPyFunc<T>
where
    T: Element,
{
    pub x: Option<Arc<ArraySignal<T>>>,
    pub t: Option<Arc<ScalarSignal<f64>>>,
    pub output: Arc<ArraySignal<T>>,
    pub py_fn: PyObject,
}

impl<T> Operator for SimPyFunc<T>
where
    T: Element,
{
    fn step(&self) {
        let gil = Python::acquire_gil();
        let py = gil.python();

        let args = PyTuple::new(
            py,
            match &self.t {
                Some(t) => {
                    let t: &PyAny = PyFloat::new(py, **t.read());
                    match &self.x {
                        Some(x) => vec![t, x.read().to_py_array(py)],
                        None => vec![t],
                    }
                }
                None => vec![],
            },
        );

        let result = &self
            .py_fn
            .as_ref(py)
            .call(args, None)
            .expect("Call to Python function failed.")
            .extract::<Option<&PyArrayDyn<T>>>()
            .expect("Python function failed.");
        if let Some(result) = result {
            let mut output = self.output.write();
            output.assign_array(&result.readonly().as_array());
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
    fn it_calls_the_function_without_arguments() {
        let gil = Python::acquire_gil();
        let py = gil.python();
        activate_venv(py);

        let numpy = PyModule::import(py, "numpy").unwrap();
        let locals = [("np", numpy.to_object(py))].into_py_dict(py);

        let op = SimPyFunc::<i64> {
            x: None,
            t: None,
            output: Arc::new(ArraySignal::new(
                String::from("output"),
                PyArrayDyn::from_array(py, &array![0].into_dimensionality::<IxDyn>().unwrap()),
            )),
            py_fn: py
                .eval("lambda: np.array([42])", Some(locals), None)
                .unwrap()
                .into(),
        };
        op.output.reset();

        op.step();

        assert_eq!(
            **op.output.read(),
            ArrayRef::Owned(array![42].into_dimensionality::<IxDyn>().unwrap())
        );
    }

    #[test]
    fn it_calls_the_function_with_time_argument() {
        let gil = Python::acquire_gil();
        let py = gil.python();
        activate_venv(py);

        let numpy = PyModule::import(py, "numpy").unwrap();
        let locals = [("np", numpy.to_object(py))].into_py_dict(py);

        let op = SimPyFunc::<f64> {
            x: None,
            t: Some(Arc::new(ScalarSignal::new(String::from("t"), 1.))),
            output: Arc::new(ArraySignal::new(
                String::from("output"),
                PyArrayDyn::from_array(py, &array![0.].into_dimensionality::<IxDyn>().unwrap()),
            )),
            py_fn: py
                .eval("lambda t: np.array([t])", Some(locals), None)
                .unwrap()
                .into(),
        };
        op.t.as_ref().map(|t| t.reset());
        op.output.reset();

        op.step();

        assert_eq!(
            **op.output.read(),
            ArrayRef::Owned(array![1.].into_dimensionality::<IxDyn>().unwrap())
        );
    }

    #[test]
    fn it_calls_the_function_with_time_and_x_argument() {
        let gil = Python::acquire_gil();
        let py = gil.python();
        activate_venv(py);

        let numpy = PyModule::import(py, "numpy").unwrap();
        let locals = [("np", numpy.to_object(py))].into_py_dict(py);

        let op = SimPyFunc::<f64> {
            x: Some(Arc::new(ArraySignal::new(
                String::from("x"),
                PyArrayDyn::from_array(py, &array![2., 3.].into_dimensionality::<IxDyn>().unwrap()),
            ))),
            t: Some(Arc::new(ScalarSignal::new(String::from("t"), 1.))),
            output: Arc::new(ArraySignal::new(
                String::from("output"),
                PyArrayDyn::from_array(
                    py,
                    &array![0., 0., 0.].into_dimensionality::<IxDyn>().unwrap(),
                ),
            )),
            py_fn: py
                .eval(
                    "lambda t, x: np.hstack((np.array([t]), x))",
                    Some(locals),
                    None,
                )
                .unwrap()
                .into(),
        };
        op.x.as_ref().map(|x| x.reset());
        op.t.as_ref().map(|t| t.reset());
        op.output.reset();

        op.step();

        assert_eq!(
            **op.output.read(),
            ArrayRef::Owned(array![1., 2., 3.].into_dimensionality::<IxDyn>().unwrap())
        );
    }
}
