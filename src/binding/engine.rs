use crate::binding::operator::PyOperator;
use crate::binding::probe::PyProbe;
use crate::binding::signal::PySignal;
use crate::binding::Wrapper;
use crate::engine::Engine;
use pyo3::prelude::*;
use pyo3::PyClass;
use std::sync::Arc;

#[pyclass(name = Engine)]
pub struct PyEngine {
    engine: Engine,
}

#[pymethods]
impl PyEngine {
    #[new]
    fn new(signals: &PyAny, operators: &PyAny, probes: &PyAny) -> PyResult<Self> {
        fn py_cells_to_pure_rust<T: PyClass + Wrapper<Arc<U>>, U: ?Sized>(
            cells: &Vec<&PyCell<T>>,
        ) -> Vec<Arc<U>> {
            cells.iter().map(|c| Arc::clone(c.borrow().get())).collect()
        }

        Ok(Self {
            engine: Engine::new(
                py_cells_to_pure_rust::<PySignal, _>(&signals.extract()?),
                py_cells_to_pure_rust::<PyOperator, _>(&operators.extract()?),
                py_cells_to_pure_rust::<PyProbe, _>(&probes.extract()?),
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
