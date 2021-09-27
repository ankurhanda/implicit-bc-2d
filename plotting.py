import matplotlib.pyplot as plt #
import numpy as np

pred_data = np.loadtxt('vanilla_coords_pred.txt')

x = pred_data[:, 0]
x = x.astype(int)

y = pred_data[:, 1]
y = y.astype(int)

err = pred_data[:, 2]

idx = np.where(err<=1.0)
err[idx] = 0 

c = err / np.amax(err)
c = c/2.0 + 0.5 

c[idx] = 0

colours = np.zeros((len(err), 3))
colours[:, 0] = 1-c 

print(np.amin(err), np.amax(err), len(err[idx]))

plt.scatter(x, y, c=colours, s=20)

training_data = np.loadtxt('train_dataset.txt')

x = training_data[:,0].astype(int)
y = training_data[:,1].astype(int)

plt.scatter(x, y, c='green', marker = "v", s=50)

plt.show()