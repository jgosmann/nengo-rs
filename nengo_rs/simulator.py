from nengo.builder import Model
from nengo.builder import operator as core_op
from nengo.cache import get_default_decoder_cache
from nengo.utils.graphs import BidirectionalDAG, toposort
from nengo.utils.simulator import operator_dependency_graph
import numpy as np

from .engine import (
    Engine,
    SignalArrayF64,
    SignalF64,
    SignalU64,
    Reset,
    TimeUpdate,
    ElementwiseInc,
    Copy,
    Probe,
)


class Simulator:
    def __init__(self, network, dt=0.001, seed=None):
        self.model = Model(
            dt=float(dt),
            label="Nengo RS model",
            decoder_cache=get_default_decoder_cache(),
        )
        self.model.build(network)

        signal_to_engine_id = {}
        for signal_dict in self.model.sig.values():
            for signal in signal_dict.values():
                if signal is not None:
                    signal_to_engine_id[signal] = SignalArrayF64(signal)
        x = SignalU64("step", 0)
        signal_to_engine_id[self.model.step] = x
        signal_to_engine_id[self.model.time] = SignalF64("time", 0.0)
        self._sig_to_ngine_id = signal_to_engine_id

        dg = BidirectionalDAG(operator_dependency_graph(self.model.operators))
        toposorted_dg = toposort(dg.forward)
        node_indices = {node: idx for idx, node in enumerate(toposorted_dg)}

        ops = []
        for op in toposorted_dg:
            dependencies = [node_indices[node] for node in dg.backward[op]]
            if isinstance(op, core_op.Reset):
                ops.append(
                    Reset(
                        np.asarray(op.value, dtype=np.float64),
                        signal_to_engine_id[op.dst],
                        dependencies,
                    )
                )
            elif isinstance(op, core_op.TimeUpdate):
                ops.append(
                    TimeUpdate(
                        dt,
                        signal_to_engine_id[self.model.step],
                        signal_to_engine_id[self.model.time],
                        dependencies,
                    )
                )
            elif isinstance(op, core_op.ElementwiseInc):
                ops.append(
                    ElementwiseInc(
                        signal_to_engine_id[op.Y],
                        signal_to_engine_id[op.A],
                        signal_to_engine_id[op.X],
                        dependencies,
                    )
                )
            elif isinstance(op, core_op.Copy):
                assert op.src_slice is None and op.dst_slice is None
                ops.append(
                    Copy(
                        signal_to_engine_id[op.src],
                        signal_to_engine_id[op.dst],
                        dependencies,
                    )
                )
            else:
                print("missing:", op)

        self.probe_mapping = {}
        for probe in self.model.probes:
            self.probe_mapping[probe] = Probe(
                signal_to_engine_id[self.model.sig[probe]["in"]]
            )

        self._engine = Engine(
            list(signal_to_engine_id.values()), ops, list(self.probe_mapping.values())
        )
        self.data = SimData(self)

        self._engine.reset()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def close(self):
        pass

    @property
    def dt(self):
        return self.model.dt

    def run(self, time_in_seconds):
        n_steps = int(time_in_seconds / self.dt)
        self._engine.run_steps(n_steps)

    def run_step(self):
        self._engine.run_step()

    def trange(self):
        step = self._sig_to_ngine_id[self.model.step].get()
        return np.arange(1, step + 1) * self.dt


class SimData:
    def __init__(self, sim):
        self._sim = sim

    def __getitem__(self, key):
        return self._sim.probe_mapping[key].get_data()

