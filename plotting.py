import matplotlib.pyplot as plt #
import numpy as np

import matplotlib as mpl 

pred_data = np.loadtxt('vanilla_coords_pred.txt')
# pred_data = np.loadtxt('implicit_trained_models/vanilla_coords_pred.txt')

x = pred_data[:, 0]
x = x.astype(int)

y = pred_data[:, 1]
y = y.astype(int)

err = pred_data[:, 2]

# idx = np.where(np.logical_and(err>1, err<=2)) #np.logical_and(np.where(err>=2), np.where(err<=5))
idx = np.where(err<=1) #np.logical_and(np.where(err>=2), np.where(err<=5))
err[idx] = 0 


c = err # / np.amax(err)
c[idx] = 0
err_threshold = 50 
c[err>err_threshold] = err_threshold


cmap = mpl.cm.OrRd
norm = mpl.colors.Normalize(vmin=2, vmax=err_threshold)


print(np.amin(err), np.amax(err), len(err[idx]))
plt.scatter(x, y, c=c, cmap=cmap, s=20)
plt.scatter(x[idx], y[idx], c='b', s=20)
plt.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap))

training_data = np.loadtxt('training_dataset.txt')
# training_data = np.loadtxt('implicit_trained_models/train_dataset.txt')

x = training_data[:,0].astype(int)
y = training_data[:,1].astype(int)

plt.scatter(x, y, c='green', marker = "v", s=50)
plt.show()