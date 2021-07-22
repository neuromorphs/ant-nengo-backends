import nengo
import matplotlib.pyplot as plt
import numpy as np

model = nengo.Network()
with model:
    stim = nengo.Node(lambda t: np.sin(t*2*np.pi))

    ens = nengo.Ensemble(n_neurons=100, dimensions=1)
    nengo.Connection(stim, ens, synapse=None)
    
    p = nengo.Probe(ens, synapse=0.01)

sim = nengo.Simulator(model)
with sim:
    sim.run(1)

plt.plot(sim.trange(), sim.data[p])
plt.show()
