from nengo.builder import Model
from nengo.builder.operator import Copy, ElementwiseInc, Reset, TimeUpdate
from nengo.cache import get_default_decoder_cache
from nengo.utils.graphs import toposort
from nengo.utils.simulator import operator_dependency_graph
import numpy as np

from .engine import (
    Engine,
    RsSignalArrayF64,
    RsSignalF64,
    RsSignalU64,
    RsReset,
    RsTimeUpdate,
    RsElementwiseInc,
    RsCopy,
    RsProbe,
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
                    signal_to_engine_id[signal] = RsSignalArrayF64(signal)
        x = RsSignalU64("step", 0)
        print("a", x)
        signal_to_engine_id[self.model.step] = x
        signal_to_engine_id[self.model.time] = RsSignalF64("time", 0.0)
        self._sig_to_ngine_id = signal_to_engine_id

        dg = operator_dependency_graph(self.model.operators)
        ops = []
        for op in toposort(dg):
            if isinstance(op, Reset):
                print(op.dst)
                ops.append(
                    RsReset(
                        np.asarray(op.value, dtype=np.float64),
                        signal_to_engine_id[op.dst],
                    )
                )
            elif isinstance(op, TimeUpdate):
                print("b", signal_to_engine_id[self.model.step])
                ops.append(
                    RsTimeUpdate(
                        dt,
                        signal_to_engine_id[self.model.step],
                        signal_to_engine_id[self.model.time],
                    )
                )
            elif isinstance(op, ElementwiseInc):
                ops.append(
                    RsElementwiseInc(
                        signal_to_engine_id[op.Y],
                        signal_to_engine_id[op.A],
                        signal_to_engine_id[op.X],
                    )
                )
            elif isinstance(op, Copy):
                assert op.src_slice is None and op.dst_slice is None
                ops.append(
                    RsCopy(signal_to_engine_id[op.src], signal_to_engine_id[op.dst])
                )
            else:
                print("missing:", op)

        self.probe_mapping = {}
        for probe in self.model.probes:
            self.probe_mapping[probe] = RsProbe(
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
        print(step)
        return np.arange(1, step + 1) * self.dt


class SimData:
    def __init__(self, sim):
        self._sim = sim

    def __getitem__(self, key):
        return self._sim.probe_mapping[key].get_data()

