import scipy.io
import os
import tensorflow as tf
import model

working_dir = os.getcwd()

file_path = os.path.join(working_dir, 'jaxpi', 'examples', 'allen_cahn','data', 'allen_cahn.mat')

data = scipy.io.loadmat(file_path)

# get data for model
x = data['x'].flatten()
t = data['t'].flatten()
usol = data['usol']

model.AllenCahnPINN(t,x,usol)