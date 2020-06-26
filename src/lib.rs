mod engine;
mod operator;
mod probe;
mod signal;

use crate::engine::Engine;
use pyo3::prelude::*;

#[pymodule]
fn engine(py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<Engine>()?;

    Ok(())
}

#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {
        assert_eq!(2 + 2, 4);
    }
}
