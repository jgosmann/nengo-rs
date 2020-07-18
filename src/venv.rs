use pyo3::prelude::*;
use std::env;
use std::sync::Once;

static VENV_INIT: Once = Once::new();

pub fn activate_venv(py: Python) {
    py.allow_threads(|| {
        VENV_INIT.call_once(|| {
            let gil = Python::acquire_gil();
            let py = gil.python();
            if let Ok(venv) = env::var("VIRTUAL_ENV") {
                py.run(
                    &format!(
                        r#"
import os, site, sys

orig_path = list(sys.path)
sys.path[:] = []
site.main()
sys.path[:] = [item for item in orig_path if item not in set(sys.path)]

if "__PYVENV_LAUNCHER__" in os.environ:
    del os.environ["__PYVENV_LAUNCHER__"]

sys.executable = "{}/bin/python"
sys.prefix = "{}"
sys.exec_prefix = "{}"
site.main()
"#,
                        venv, venv, venv
                    ),
                    None,
                    None,
                )
                .expect(format!("Failed to initialize virtual env: {}", venv).as_str());
            }
        });
    });
}
