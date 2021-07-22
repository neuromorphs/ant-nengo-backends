import nengo
import matplotlib.pyplot as plt
import numpy as np

model = nengo.Network()
with model:
    ens = nengo.Ensemble(n_neurons=100, dimensions=1)
    
    data = []
    def output_func(t, x):
        data.append(x)
    output = nengo.Node(output_func, size_in=100)
    nengo.Connection(ens.neurons, output, synapse=None)

sim = nengo.Simulator(model)
with sim:
    sim.run(1)

plt.imshow(np.array(data).T, aspect='auto')
plt.show()
