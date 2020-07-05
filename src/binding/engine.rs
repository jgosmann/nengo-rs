use crate::binding::operator::RsOperator;
use crate::binding::probe::RsProbe;
use crate::binding::signal::RsSignal;
use crate::binding::Wrapper;
use crate::engine::Engine;
use pyo3::prelude::*;
use pyo3::PyClass;
use std::sync::Arc;

#[pyclass]
pub struct RsEngine {
    engine: Engine,
}

#[pymethods]
impl RsEngine {
    #[new]
    fn new(signals: &PyAny, operators: &PyAny, probes: &PyAny) -> PyResult<Self> {
        fn py_cells_to_pure_rust<T: PyClass + Wrapper<Arc<U>>, U: ?Sized>(
            cells: &Vec<&PyCell<T>>,
        ) -> Vec<Arc<U>> {
            cells.iter().map(|c| Arc::clone(c.borrow().get())).collect()
        }

        Ok(Self {
            engine: Engine::new(
                py_cells_to_pure_rust::<RsSignal, _>(&signals.extract()?),
                py_cells_to_pure_rust::<RsOperator, _>(&operators.extract()?),
                py_cells_to_pure_rust::<RsProbe, _>(&probes.extract()?),
            ),
        })
    }

    fn run_step(&self) {
        self.engine.run_step();
    }

    fn run_steps(&self, n_steps: i64) {
        self.engine.run_steps(n_steps);
    }

    fn reset(&self) {
        self.engine.reset();
    }
}
