from nengo.builder import Model
from nengo.builder import operator as core_op
from nengo.builder import neurons
from nengo.builder import processes
from nengo.builder.signal import SignalDict
from nengo.cache import get_default_decoder_cache
from nengo.utils.graphs import BidirectionalDAG, toposort
from nengo.utils.simulator import operator_dependency_graph
import numpy as np

from .index_conv import slices_from_signal
from .engine import (
    Engine,
    SignalArrayF64,
    SignalArrayViewF64,
    SignalF64,
    SignalU64,
    Reset,
    TimeUpdate,
    ElementwiseInc,
    Copy,
    DotInc,
    Probe,
    SimNeurons,
    SimProcess,
    SimPyFunc,
)


class Simulator:
    def add_sig(self, signal_to_engine_id, signal):
        if signal is None or signal in signal_to_engine_id:
            pass
        elif signal.base is None or signal is signal.base:
            signal_to_engine_id[signal] = SignalArrayF64(signal)
        else:
            current = signal
            sliceinfo = slices_from_signal(signal)
            while (
                current.base.base is not None and current.base.base is not current.base
            ):
                current = current.base
                sliceinfo = tuple(
                    slice(
                        b.start + b.step * a.start,
                        b.start + b.step * a.stop,
                        a.step * b.step,
                    )
                    for a, b in zip(sliceinfo, slices_from_signal(current))
                )
            self.add_sig(signal_to_engine_id, current.base)
            try:
                signal_to_engine_id[signal] = SignalArrayViewF64(
                    signal.name, sliceinfo, signal_to_engine_id[current.base]
                )
            except TypeError:
                print(
                    f"TypeError: {signal.name} {sliceinfo} {current.base} {signal_to_engine_id[current.base]}"
                )
                raise

    def get_sig(self, signal_to_engine_id, signal):
        self.add_sig(signal_to_engine_id, signal)
        return signal_to_engine_id[signal]

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
                self.add_sig(signal_to_engine_id, signal)
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
                        self.get_sig(signal_to_engine_id, op.dst),
                        dependencies,
                    )
                )
            elif isinstance(op, core_op.TimeUpdate):
                ops.append(
                    TimeUpdate(
                        dt,
                        self.get_sig(signal_to_engine_id, self.model.step),
                        self.get_sig(signal_to_engine_id, self.model.time),
                        dependencies,
                    )
                )
            elif isinstance(op, core_op.ElementwiseInc):
                ops.append(
                    ElementwiseInc(
                        self.get_sig(signal_to_engine_id, op.Y),
                        self.get_sig(signal_to_engine_id, op.A),
                        self.get_sig(signal_to_engine_id, op.X),
                        dependencies,
                    )
                )
            elif isinstance(op, core_op.Copy):
                assert op.src_slice is None and op.dst_slice is None
                ops.append(
                    Copy(
                        self.get_sig(signal_to_engine_id, op.src),
                        self.get_sig(signal_to_engine_id, op.dst),
                        dependencies,
                    )
                )
            elif isinstance(op, core_op.DotInc):
                ops.append(
                    DotInc(
                        self.get_sig(signal_to_engine_id, op.Y),
                        self.get_sig(signal_to_engine_id, op.A),
                        self.get_sig(signal_to_engine_id, op.X),
                        dependencies,
                    )
                )
            elif isinstance(op, neurons.SimNeurons):
                signals = SignalDict()
                op.init_signals(signals)
                ops.append(
                    SimNeurons(
                        self.dt,
                        op.neurons.step_math,
                        [signals[s] for s in op.states]
                        if hasattr(op, "states")
                        else [],
                        self.get_sig(signal_to_engine_id, op.J),
                        self.get_sig(signal_to_engine_id, op.output),
                        dependencies,
                    )
                )
            elif isinstance(op, processes.SimProcess):
                signals = SignalDict()
                op.init_signals(signals)
                shape_in = (0,) if op.input is None else op.input.shape
                shape_out = op.output.shape
                rng = None
                state = {k: signals[s] for k, s in op.state.items()}
                step_fn = op.process.make_step(shape_in, shape_out, self.dt, rng, state)
                ops.append(
                    SimProcess(
                        op.mode == "inc",
                        lambda *args, step_fn=step_fn: np.asarray(
                            step_fn(*args), dtype=float
                        ),
                        self.get_sig(signal_to_engine_id, op.t),
                        self.get_sig(signal_to_engine_id, op.output),
                        None
                        if op.input is None
                        else self.get_sig(signal_to_engine_id, op.input),
                        dependencies,
                    )
                )
            elif isinstance(op, core_op.SimPyFunc):
                ops.append(
                    SimPyFunc(
                        lambda *args, op=op: np.asarray(op.fn(*args), dtype=float),
                        self.get_sig(signal_to_engine_id, op.output),
                        None
                        if op.t is None
                        else self.get_sig(signal_to_engine_id, op.t),
                        None
                        if op.x is None
                        else self.get_sig(signal_to_engine_id, op.x),
                        dependencies,
                    )
                )
            else:
                raise Exception(f"missing: {op}")

        self.probe_mapping = {}
        for probe in self.model.probes:
            self.probe_mapping[probe] = Probe(
                signal_to_engine_id[self.model.sig[probe]["in"]]
            )

        self._engine = Engine(
            list(signal_to_engine_id.values()), ops, list(self.probe_mapping.values())
        )
        self.data = SimData(self)
        print("initialized")

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
        print("run")
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

