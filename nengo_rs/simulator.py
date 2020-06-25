from nengo.builder import Model
from nengo.builder.operator import Copy, ElementwiseInc, Reset, TimeUpdate
from nengo.cache import get_default_decoder_cache
from nengo.utils.graphs import toposort
from nengo.utils.simulator import operator_dependency_graph
import numpy as np

from .engine import Engine


class Simulator:
    def __init__(self, network, dt=0.001, seed=None):
        self.model = Model(
            dt=float(dt),
            label="Nengo RS model",
            decoder_cache=get_default_decoder_cache(),
        )
        self.model.build(network)

        self._engine = Engine(dt)
        signal_to_engine_id = {}
        for signal_dict in self.model.sig.values():
            for signal in signal_dict.values():
                if signal is not None:
                    signal_to_engine_id[signal] = self._engine.add_signal(signal)
        signal_to_engine_id[self.model.step] = self._engine.add_signal(self.model.step)
        signal_to_engine_id[self.model.time] = self._engine.add_signal(self.model.time)
        self._sig_to_ngine_id = signal_to_engine_id

        dg = operator_dependency_graph(self.model.operators)
        for op in toposort(dg):
            if isinstance(op, Reset):
                self._engine.push_reset(
                    np.asarray(op.value, dtype=np.float64), signal_to_engine_id[op.dst]
                )
            elif isinstance(op, TimeUpdate):
                self._engine.push_time_update(
                    signal_to_engine_id[op.step], signal_to_engine_id[op.time],
                )
            elif isinstance(op, ElementwiseInc):
                self._engine.push_elementwise_inc(
                    signal_to_engine_id[op.Y],
                    signal_to_engine_id[op.A],
                    signal_to_engine_id[op.X],
                )
            elif isinstance(op, Copy):
                assert op.src_slice is None and op.dst_slice is None
                self._engine.push_copy(
                    signal_to_engine_id[op.src], signal_to_engine_id[op.dst]
                )
            else:
                print("missing:", op)

        self.probe_mapping = {}
        for probe in self.model.probes:
            self.probe_mapping[probe] = self._engine.add_probe(
                signal_to_engine_id[self.model.sig[probe]["in"]]
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
        step = self._engine.get_signal_i64(self._sig_to_ngine_id[self.model.step])
        print(step)
        return np.arange(1, step + 1) * self.dt


class SimData:
    def __init__(self, sim):
        self._sim = sim

    def __getitem__(self, key):
        return self._sim._engine.get_probe_data(self._sim.probe_mapping[key])

