import numpy as np

t_train = np.load('t_train.npy')
x_train = np.load('x_train.npy')

inputs = np.stack([t_train,x_train],axis=1)

print(inputs[511::512])