import nengo
import matplotlib.pyplot as plt
import numpy as np

model = nengo.Network()
with model:
    ens = nengo.Ensemble(n_neurons=100, dimensions=1)
    
    p = nengo.Probe(ens, synapse=0.01)

sim = nengo.Simulator(model)
with sim:
    sim.run(1)

plt.plot(sim.trange(), sim.data[p])
plt.show()
