mod binding;
mod engine;
mod operator;
mod probe;
mod signal;
mod sync;

use crate::binding::{
    engine::RsEngine,
    operator::{RsCopy, RsElementwiseInc, RsReset, RsTimeUpdate},
    probe::RsProbe,
    signal::{RsSignalArrayF64, RsSignalF64, RsSignalU64},
};
use pyo3::prelude::*;

#[pymodule]
fn engine(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<RsEngine>()?;
    m.add_class::<RsSignalArrayF64>()?;
    m.add_class::<RsSignalF64>()?;
    m.add_class::<RsSignalU64>()?;
    m.add_class::<RsReset>()?;
    m.add_class::<RsTimeUpdate>()?;
    m.add_class::<RsElementwiseInc>()?;
    m.add_class::<RsCopy>()?;
    m.add_class::<RsProbe>()?;

    Ok(())
}
