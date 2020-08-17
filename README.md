# Nengo RS

Nengo RS is an _experimental_ [Nengo](https://www.nengo.ai/) backend
written in Rust.

This was a purely experimental project that I used to learn Rust
(and it really did its job in this regard).
It is
in no-way suitable for production,
missing several features,
and not even faster than the Python reference implementation!

## Getting started

Clone the repository and run:

```bash
pip install maturin
maturin develop
```

To run the rust tests:

```bash
cargo test
```

To run the bundled Python tests:

```bash
pytest
```

To run the Nengo tests against the provided backend
(only a subset will succeed):

```bash
pytest --pyargs nengo
```

To use the Nengo RS backend in your Python code:

```python
import nengo_rs

with nengo_rs.Simulator(model) as sim:
    sim.run(duration)
```

Be aware that the `Simulator` interface is only partially implemented.
In particular, the `seed` argument is not respected.

## Approach and limitations

To implement a minimal working Nengo backend with reasonable effort,
I am using the Python reference backend to build the model,
and convert the signals and operators into Rust equivalents.
The optimization step is currently skipped
as it would require an additional operator (`BsrDotInc`)
to be implemented.

Operator execution is scheduled with asynchronous tasks
using async/await
which allows to somewhat nicely wait for operators
providing dependency signals
to be done.
However,
it is not clear whether this actually improves performance
through parallelization
or the scheduling overhead is too much.

Though,
the largest bottlenecks
at the moment,
should be
the missing support for the optimizer,
and that some core operators call back into Python.
All of `SimNeurons`, `SimProcess`, and `SimPyFunc`
execute within the Python interpreter.
While we cannot get around that for `SimPyFunc`,
it would be possible to fully implement `SimNeurons` and `SimProcess`
in Rust for a given set of processes and neuron types.

One of my major gripes with the current implementation is
that it is not well suited for Rusts memory model
with ownership and borrow checking.
Essentially,
each signal is a block of shared memory
and not owned by a single operator.
Thus, there is certain syntax overhead with that.
In addition,
the current approach requires to map
between the Python signals/operators
and the Rust equivalents,
adding another level of sharing.

Operators do not only use owned signals,
but access views of these signals.
For a view in Rust in must be ensured
that the owned array outlives the view
which is not really possible
due to the shared ownership.
Thus it is necessary to use a custom struct
to track owned arrays and views (`ArrayRef`)
and only resolve these when needed.
Gives two arrays/views and an operation,
there are four combinations
of what is an owned array or view.
In effect this leads to a lot of boilerplate
that I also do not find highly readable.

I think at least two things are required
to remedy this situation:

1. Implement the builder itself in Rust,
   so that the signals/operators can fully live in Rust
   and no mapping with Python instances is required.
2. Operators do not keep references to their signals,
   but the "engine" owns the signals
   and lends them to the operator functions
   when it is their respective time to run.

Maybe I will try such an implementation one day,
but for now I will turn towards other projects.
