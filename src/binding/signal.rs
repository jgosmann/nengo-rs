use crate::binding::Wrapper;
use crate::signal::{ArraySignal, ScalarSignal, Signal, SignalAccess};
use ndarray::{SliceInfo, SliceOrIndex};
use numpy::PyArrayDyn;
use pyo3::exceptions as exc;
use pyo3::prelude::*;
use std::any::type_name;
use std::cmp::min;
use std::sync::Arc;

#[pyclass(name=Signal)]
#[derive(Debug)]
pub struct PySignal {
    signal: Arc<dyn Signal + Send + Sync>,
}

impl Wrapper<Arc<dyn Signal + Send + Sync>> for PySignal {
    fn get(&self) -> &Arc<dyn Signal + Send + Sync> {
        &self.signal
    }
}

impl PySignal {
    pub fn extract_signal<T: Signal + Send + Sync + 'static>(
        &self,
        name: &str,
    ) -> PyResult<Arc<T>> {
        Arc::downcast::<T>(Arc::clone(&self.signal).as_any_arc()).or(Err(PyErr::new::<
            exc::TypeError,
            _,
        >(format!(
            "Signal `{}` must be {}.",
            name,
            type_name::<T>()
        ))))
    }
}

#[pyclass(extends=PySignal, name=SignalArrayF64)]
pub struct PySignalArrayF64 {}

#[pymethods]
impl PySignalArrayF64 {
    #[new]
    fn new(signal: &PyAny) -> PyResult<(Self, PySignal)> {
        let name = signal.getattr("name")?.extract()?;
        let initial_value = signal.getattr("initial_value")?;
        let initial_value: &PyArrayDyn<f64> = initial_value.extract()?;
        let signal = Arc::new(ArraySignal::new(name, initial_value));
        Ok((Self {}, PySignal { signal }))
    }
}

#[pyclass(extends=PySignal, name=SignalArrayViewF64)]
pub struct PySignalArrayViewF64 {}

#[pymethods]
impl PySignalArrayViewF64 {
    #[new]
    fn new(signal: &PyAny, base: &PyCell<PySignal>) -> PyResult<(Self, PySignal)> {
        let name = signal.getattr("name")?.extract()?;
        let base: &PyCell<PySignal> = base.extract().unwrap();
        let base: Arc<ArraySignal<f64>> = base.borrow().extract_signal("base")?;
        let initial_value = signal.getattr("initial_value")?;
        let initial_value: Option<&PyArrayDyn<f64>> = initial_value.extract().ok();

        let offset: isize = signal.getattr("elemoffset")?.extract()?;
        let strides: Vec<isize> = signal.getattr("elemstrides")?.extract()?;
        let shape: Vec<isize> = signal.getattr("shape")?.extract()?;
        let base_strides: Vec<isize> = signal.getattr("base")?.getattr("elemstrides")?.extract()?;
        let base_shape: Vec<isize> = signal.getattr("base")?.getattr("shape")?.extract()?;
        let slice_info = Box::new(
            SliceInfo::new(
                Self::offset_to_multiindex(offset, &base_strides)
                    .iter()
                    .zip(shape)
                    .zip(Self::strides_to_steps(&strides, &base_strides))
                    .zip(base_shape)
                    .map(|(((start, size), step), base_size)| SliceOrIndex::Slice {
                        start: *start,
                        step: step,
                        end: Some(min(start + step * size, base_size)),
                    })
                    .collect(),
            )
            .unwrap(),
        );

        let signal = Arc::new(ArraySignal::new_view(name, base, slice_info, initial_value));
        Ok((Self {}, PySignal { signal }))
    }
}

impl PySignalArrayViewF64 {
    fn offset_to_multiindex(offset: isize, base_strides: &Vec<isize>) -> Vec<isize> {
        base_strides
            .iter()
            .scan(offset, |remainder, &divisor| {
                let index = *remainder / divisor;
                *remainder = *remainder % divisor;
                Some(index)
            })
            .collect()
    }

    fn strides_to_steps(strides: &Vec<isize>, base_strides: &Vec<isize>) -> Vec<isize> {
        strides
            .iter()
            .zip(base_strides)
            .map(|(stride, base_stride)| stride / base_stride)
            .collect()
    }
}

#[pyclass(extends=PySignal, name=SignalU64)]
pub struct PySignalU64 {}

#[pymethods]
impl PySignalU64 {
    #[new]
    fn new(name: String, initial_value: u64) -> PyResult<(Self, PySignal)> {
        Ok((
            Self {},
            PySignal {
                signal: Arc::new(ScalarSignal::new(name, initial_value)),
            },
        ))
    }

    fn get(py_self: PyRef<Self>) -> u64 {
        **py_self
            .as_ref()
            .signal
            .as_any()
            .downcast_ref::<ScalarSignal<u64>>()
            .unwrap()
            .read()
    }
}

#[pyclass(extends=PySignal, name=SignalF64)]
pub struct PySignalF64 {}

#[pymethods]
impl PySignalF64 {
    #[new]
    fn new(name: String, initial_value: f64) -> PyResult<(Self, PySignal)> {
        Ok((
            Self {},
            PySignal {
                signal: Arc::new(ScalarSignal::new(name, initial_value)),
            },
        ))
    }

    fn get(py_self: PyRef<Self>) -> f64 {
        **py_self
            .as_ref()
            .signal
            .as_any()
            .downcast_ref::<ScalarSignal<f64>>()
            .unwrap()
            .read()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::signal::ArrayRef;
    use crate::venv::activate_venv;
    use ndarray::prelude::*;
    use ndarray::Ix;
    use pyo3::{types::IntoPyDict, wrap_pymodule, ToPyObject};
    use std::fmt::Debug;

    #[pymodule]
    fn signal(_py: Python, m: &PyModule) -> PyResult<()> {
        m.add_class::<PySignalArrayF64>()?;
        m.add_class::<PySignalArrayViewF64>()?;
        m.add_class::<PySignalF64>()?;
        m.add_class::<PySignalU64>()?;
        Ok(())
    }

    fn test_binding<T, S>(expr: &str, expected_name: &str, expected_shape: &[Ix], expected_value: T)
    where
        T: Debug + PartialEq,
        S: Signal + SignalAccess<T> + Send + Sync + 'static,
    {
        let gil = Python::acquire_gil();
        let py = gil.python();
        activate_venv(py);
        let nengo = PyModule::import(py, "nengo").unwrap();
        let numpy = PyModule::import(py, "numpy").unwrap();
        let signal_module = wrap_pymodule!(signal)(py);
        let locals = [
            ("nengo", nengo.to_object(py)),
            ("np", numpy.to_object(py)),
            ("s", signal_module),
        ]
        .into_py_dict(py);

        let py_signal = py.eval(expr, None, Some(locals)).unwrap_or_else(|e| {
            e.print_and_set_sys_last_vars(py);
            panic!();
        });

        let py_signal: &PyCell<PySignal> = py_signal.extract().unwrap();

        let signal = py_signal.borrow();
        assert_eq!(signal.get().name(), expected_name);
        assert_eq!(signal.get().shape(), expected_shape);

        let signal: Arc<S> = py_signal.borrow().extract_signal("test").unwrap();
        signal.reset();
        assert_eq!(**signal.read(), expected_value);
    }

    #[test]
    fn test_py_signal_array_f64() {
        test_binding::<_, ArraySignal<f64>>(
            "s.SignalArrayF64(nengo.builder.signal.Signal(np.array([1., 2.]), name='TestSignal'))",
            "TestSignal",
            &[2],
            ArrayRef::Owned(array![1., 2.].into_dimensionality::<IxDyn>().unwrap()),
        );
    }

    #[test]
    fn test_py_signal_array_f64_from_scalar() {
        test_binding::<_, ArraySignal<f64>>(
            "s.SignalArrayF64(nengo.builder.signal.Signal(np.float64(42.), name='TestSignal'))",
            "TestSignal",
            &[],
            ArrayRef::Owned(array![42.].into_dimensionality::<IxDyn>().unwrap()),
        );
    }

    #[test]
    fn test_py_signal_array_view_f64() {
        test_binding::<_, ArraySignal<f64>>(
            "(lambda s, nengo, base: s.SignalArrayViewF64(
                base[1::2],
                s.SignalArrayF64(base)))(s, nengo, nengo.builder.signal.Signal(np.array([0., 1., 0., 2.]), name='BaseSignal'))",
            "BaseSignal[(slice(1, None, 2),)]",
            &[2],
            ArrayRef::Owned(array![1., 2.].into_dimensionality::<IxDyn>().unwrap()),
        );
    }

    #[test]
    fn test_py_signal_array_view_f64_x() {
        let gil = Python::acquire_gil();
        let py = gil.python();
        activate_venv(py);
        let nengo = PyModule::import(py, "nengo").unwrap();
        let numpy = PyModule::import(py, "numpy").unwrap();
        let signal_module = wrap_pymodule!(signal)(py);
        let locals = [("nengo", nengo.to_object(py)), ("np", numpy.to_object(py))].into_py_dict(py);

        let py_base_nengo_signal = py
            .eval(
                "nengo.builder.signal.Signal(np.array([0., 1., 0., 2.]), name='BaseSignal')",
                None,
                Some(locals),
            )
            .unwrap_or_else(|e| {
                e.print_and_set_sys_last_vars(py);
                panic!();
            });

        let locals = [
            ("nengo", nengo.to_object(py)),
            ("np", numpy.to_object(py)),
            ("s", signal_module),
            ("base_nengo_signal", py_base_nengo_signal.to_object(py)),
        ]
        .into_py_dict(py);
        let py_base_signal = py
            .eval("s.SignalArrayF64(base_nengo_signal)", None, Some(locals))
            .unwrap_or_else(|e| {
                e.print_and_set_sys_last_vars(py);
                panic!();
            });
        let py_base_signal: &PyCell<PySignal> = py_base_signal.extract().unwrap();
        let base_signal: Arc<ArraySignal<f64>> = py_base_signal
            .borrow()
            .extract_signal("base_signal")
            .unwrap();

        let signal_module = wrap_pymodule!(signal)(py);
        let locals = [
            ("nengo", nengo.to_object(py)),
            ("np", numpy.to_object(py)),
            ("s", signal_module),
            ("base_nengo_signal", py_base_nengo_signal.to_object(py)),
            ("base_signal", py_base_signal.to_object(py)),
        ]
        .into_py_dict(py);

        let py_signal = py
            .eval(
                "s.SignalArrayViewF64(base_nengo_signal[1::2], base_signal)",
                None,
                Some(locals),
            )
            .unwrap_or_else(|e| {
                e.print_and_set_sys_last_vars(py);
                panic!();
            });
        let py_signal: &PyCell<PySignal> = py_signal.extract().unwrap();

        let signal = py_signal.borrow();
        assert_eq!(signal.get().shape(), &[2]);

        let signal: Arc<ArraySignal<f64>> = py_signal.borrow().extract_signal("test").unwrap();
        base_signal.reset();
        assert_eq!(
            **signal.read(),
            ArrayRef::Owned(array![1., 2.].into_dimensionality::<IxDyn>().unwrap())
        );
    }

    #[test]
    fn test_py_signal_array_view_f64_y() {
        let gil = Python::acquire_gil();
        let py = gil.python();
        activate_venv(py);
        let nengo = PyModule::import(py, "nengo").unwrap();
        let numpy = PyModule::import(py, "numpy").unwrap();
        let signal_module = wrap_pymodule!(signal)(py);
        let locals = [("nengo", nengo.to_object(py)), ("np", numpy.to_object(py))].into_py_dict(py);

        let py_base_nengo_signal = py
            .eval(
                "nengo.builder.signal.Signal(np.arange(3 * 4 * 5, dtype=float).reshape((3, 4, 5)), name='BaseSignal')",
                None,
                Some(locals),
            )
            .unwrap_or_else(|e| {
                e.print_and_set_sys_last_vars(py);
                panic!();
            });

        let locals = [
            ("nengo", nengo.to_object(py)),
            ("np", numpy.to_object(py)),
            ("s", signal_module),
            ("base_nengo_signal", py_base_nengo_signal.to_object(py)),
        ]
        .into_py_dict(py);
        let py_base_signal = py
            .eval("s.SignalArrayF64(base_nengo_signal)", None, Some(locals))
            .unwrap_or_else(|e| {
                e.print_and_set_sys_last_vars(py);
                panic!();
            });
        let py_base_signal: &PyCell<PySignal> = py_base_signal.extract().unwrap();
        let base_signal: Arc<ArraySignal<f64>> = py_base_signal
            .borrow()
            .extract_signal("base_signal")
            .unwrap();

        let signal_module = wrap_pymodule!(signal)(py);
        let locals = [
            ("nengo", nengo.to_object(py)),
            ("np", numpy.to_object(py)),
            ("s", signal_module),
            ("base_nengo_signal", py_base_nengo_signal.to_object(py)),
            ("base_signal", py_base_signal.to_object(py)),
        ]
        .into_py_dict(py);

        let py_signal = py
            .eval(
                "s.SignalArrayViewF64(base_nengo_signal[1:2, ::2, 1:4:2], base_signal)",
                None,
                Some(locals),
            )
            .unwrap_or_else(|e| {
                e.print_and_set_sys_last_vars(py);
                panic!();
            });
        let py_signal: &PyCell<PySignal> = py_signal.extract().unwrap();

        let signal = py_signal.borrow();
        assert_eq!(signal.get().shape(), &[1, 2, 2]);

        let signal: Arc<ArraySignal<f64>> = py_signal.borrow().extract_signal("test").unwrap();
        base_signal.reset();
        assert_eq!(
            **signal.read(),
            ArrayRef::Owned(
                array![[[21., 23.], [31., 33.]]]
                    .into_dimensionality::<IxDyn>()
                    .unwrap()
            )
        );
    }

    #[test]
    fn test_py_signal_u64() {
        test_binding::<_, ScalarSignal<u64>>("s.SignalU64('TestSignal', 2)", "TestSignal", &[], 2);
    }

    #[test]
    fn test_py_signal_f64() {
        test_binding::<_, ScalarSignal<f64>>(
            "s.SignalF64('TestSignal', 2.)",
            "TestSignal",
            &[],
            2.,
        );
    }

    #[test]
    fn test_offset_to_multiindex() {
        assert_eq!(
            PySignalArrayViewF64::offset_to_multiindex(3, &vec![1]),
            vec![3]
        );
        assert_eq!(
            PySignalArrayViewF64::offset_to_multiindex(10, &vec![8, 4, 1]),
            vec![1, 0, 2]
        );
        assert_eq!(
            PySignalArrayViewF64::offset_to_multiindex(215, &vec![60, 12, 6, 1]),
            vec![3, 2, 1, 5]
        );
    }

    #[test]
    fn test_strides_to_steps() {
        assert_eq!(
            PySignalArrayViewF64::strides_to_steps(&vec![1], &vec![1]),
            vec![1]
        );
        assert_eq!(
            PySignalArrayViewF64::strides_to_steps(&vec![480, 192, 192, 24], &vec![480, 96, 48, 8]),
            vec![1, 2, 4, 3]
        );
    }
}
