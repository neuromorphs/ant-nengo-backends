import nengo
import numpy as np

import lmu_wta

class BuildEnsemble(object):
    def __init__(self, ensemble, rng):
        self.encoders = ensemble.encoders.sample(ensemble.n_neurons, ensemble.dimensions, rng=rng)
        self.max_rates = ensemble.max_rates.sample(ensemble.n_neurons, rng=rng)
        self.intercepts = ensemble.intercepts.sample(ensemble.n_neurons, rng=rng)

        # TODO: handle neuron models other than rectified linear
        self.gain = self.max_rates / (1 - self.intercepts)
        self.bias = -self.intercepts * self.gain

        self.scaled_encoders = self.encoders * self.gain[:, None]

class Simulator(object):
    def __init__(self, model, dt=0.001):
        self.dt = dt
        self.n_steps = 0

        self.data = {}

        self.ensemble_map = {}
        rng = np.random.RandomState(model.seed)
        for ens in model.all_ensembles:
            lmu = lmu_wta.LMUWTABlock(q=1, theta=dt, n_neurons=ens.n_neurons,
                                      size_in=ens.dimensions,
                                      size_out=1)
            self.data[ens] = BuildEnsemble(ens, rng)
            self.ensemble_map[ens] = lmu

            lmu.bias[:] = self.data[ens].bias
            lmu.E[:] = self.data[ens].scaled_encoders

        self.node_inputs = {}
        for node in model.all_nodes:
            if node.size_in > 0:
                self.node_inputs[node] = np.zeros(node.size_in)

        self.neuron_outputs = {}

        for conn in model.all_connections:
            if isinstance(conn.pre, nengo.ensemble.Neurons) and isinstance(conn.post, nengo.Node):
                ens = conn.pre.ensemble
                lmu = self.ensemble_map[ens]
                node = conn.post
                self.neuron_outputs[lmu] = self.node_inputs[node]

        self.neuron_probes = {}
        for p in model.all_probes:
            if isinstance(p.target, nengo.ensemble.Neurons) and p.attr=='output':
                ens = p.target.ensemble
                lmu = self.ensemble_map[ens]
                self.data[p] = []
                self.neuron_probes[lmu] = self.data[p]


    def run(self, time_in_seconds):
        step_count = int(time_in_seconds / self.dt)
        for i in range(step_count):
            self.step()

    def step(self):
        for lmu in self.ensemble_map.values():
            lmu.step(np.zeros(lmu.m.shape[1]))
        for lmu, node_input in self.neuron_outputs.items():
            node_input[:] += lmu.a
        for lmu, probe_data in self.neuron_probes.items():
            probe_data.append(lmu.a.copy())
        for node, node_input in self.node_inputs.items():
            node.output(self.n_steps * self.dt, node_input.copy())
            node_input[:] = 0
        self.n_steps += 1


    def trange(self, sample_every=None):
        if sample_every is None:
            sample_every = self.dt

        return (np.arange(self.n_steps)+1)*sample_every
