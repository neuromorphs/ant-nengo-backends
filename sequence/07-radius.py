import nengo
import matplotlib.pyplot as plt
import numpy as np

model = nengo.Network()
with model:
    stim = nengo.Node(lambda t: np.sin(t*2*np.pi))

    ens1 = nengo.Ensemble(n_neurons=100, dimensions=1)
    nengo.Connection(stim, ens1, synapse=None)

    ens2 = nengo.Ensemble(n_neurons=99, dimensions=2, radius=1.5)
    nengo.Connection(ens1, ens2, function=lambda x: (x,x**2), synapse=0.01)
    
    p = nengo.Probe(ens2, synapse=0.01)

sim = nengo.Simulator(model)
with sim:
    sim.run(1)

plt.plot(sim.trange(), sim.data[p])
plt.show()
