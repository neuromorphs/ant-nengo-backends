import pytry

import nengo
import nengo_lmublock
import numpy as np
import matplotlib.pyplot as plt

def test_sequence0():
    model = nengo.Network()
    with model:
        ens = nengo.Ensemble(n_neurons=100, dimensions=1)
        
        data = []
        def output_func(t, x):
            data.append(x)
        output = nengo.Node(output_func, size_in=100)
        nengo.Connection(ens.neurons, output, synapse=None)

    sim = nengo_lmublock.Simulator(model)
    sim.run(1)
    plt.plot(data)
    plt.show()

    
def test_sequence1():
    model = nengo.Network()
    with model:
        ens = nengo.Ensemble(n_neurons=100, dimensions=1)
        p = nengo.Probe(ens.neurons)

    sim = nengo_lmublock.Simulator(model)
    sim.run(1)
    plt.plot(sim.trange(), sim.data[p])
    plt.show()

