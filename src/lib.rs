mod binding;
mod engine;
mod operator;
mod probe;
mod signal;
mod sync;

use crate::binding::{
    engine::PyEngine,
    operator::{PyCopy, PyDotInc, PyElementwiseInc, PyReset, PySimPyFunc, PyTimeUpdate},
    probe::PyProbe,
    signal::{PySignalArrayF64, PySignalArrayViewF64, PySignalF64, PySignalU64},
};
use pyo3::prelude::*;

#[pymodule]
fn engine(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<PyEngine>()?;
    m.add_class::<PySignalArrayF64>()?;
    m.add_class::<PySignalArrayViewF64>()?;
    m.add_class::<PySignalF64>()?;
    m.add_class::<PySignalU64>()?;
    m.add_class::<PyReset>()?;
    m.add_class::<PySimPyFunc>()?;
    m.add_class::<PyTimeUpdate>()?;
    m.add_class::<PyElementwiseInc>()?;
    m.add_class::<PyCopy>()?;
    m.add_class::<PyDotInc>()?;
    m.add_class::<PyProbe>()?;

    Ok(())
}

#[cfg(test)]
pub mod venv;
