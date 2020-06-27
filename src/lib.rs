mod engine;
mod operator;
mod probe;
mod signal;

use crate::engine::{
    Engine, RsCopy, RsElementwiseInc, RsProbe, RsReset, RsSignalArrayF64, RsSignalF64, RsSignalU64,
    RsTimeUpdate,
};
use pyo3::prelude::*;

#[pymodule]
fn engine(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<Engine>()?;
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

#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {
        assert_eq!(2 + 2, 4);
    }
}
