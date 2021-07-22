import nengo
import matplotlib.pyplot as plt
import numpy as np

model = nengo.Network()
with model:
    ens = nengo.Ensemble(n_neurons=100, dimensions=1)
    
    p = nengo.Probe(ens.neurons)

sim = nengo.Simulator(model)
with sim:
    sim.run(1)

plt.imshow(np.array(sim.data[p]).T, aspect='auto')
plt.show()
