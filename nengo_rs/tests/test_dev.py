import nengo
import nengo_rs
import numpy as np


def test_dev():
    with nengo.Network() as model:
        node = nengo.Node(0.5)
        probe = nengo.Probe(node)

    dt = 0.001
    with nengo_rs.Simulator(model, dt=dt) as sim:
        sim.run(1.0)

    assert np.allclose(sim.trange(), np.arange(0.0, 1.0, dt) + dt)
    assert np.allclose(sim.data[probe], 0.5)
