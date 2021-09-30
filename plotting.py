import matplotlib.pyplot as plt #
import numpy as np

pred_data = np.loadtxt('vanilla_coords_pred.txt')
# pred_data = np.loadtxt('implicit_trained_models/vanilla_coords_pred.txt')

x = pred_data[:, 0]
x = x.astype(int)

y = pred_data[:, 1]
y = y.astype(int)

err = pred_data[:, 2]

idx = np.where(np.logical_and(err>=5, err<=50)) #np.logical_and(np.where(err>=2), np.where(err<=5))
err[idx] = 0 

c = err / np.amax(err)

c[idx] = 0

colours = np.zeros((len(err), 4))

colours[:, 0] = np.minimum(c+0.7, 1.0) #1-c
colours[:, 3] = 1-c #np.maximum(1-c, 0.0)#np.minimum(1-c/2.0, 1.0)

colours[idx, 0] = 0
colours[idx, 2] = 1
colours[idx, 3] = 1

print(np.amin(err), np.amax(err), len(err[idx]))

plt.scatter(x, y, c=colours, s=20)
plt.colorbar() #ticks=range(6))


training_data = np.loadtxt('training_dataset.txt')
# training_data = np.loadtxt('implicit_trained_models/train_dataset.txt')

x = training_data[:,0].astype(int)
y = training_data[:,1].astype(int)

plt.scatter(x, y, c='green', marker = "v", s=50)
# plt.colorbar(ticks=range(6))
plt.show()