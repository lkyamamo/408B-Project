import scipy.io
import os
import model

working_dir = os.getcwd()

file_path = os.path.join(working_dir, 'jaxpi', 'examples', 'allen_cahn','data', 'allen_cahn.mat')

data = scipy.io.loadmat(file_path)

# get data for model
x = data['x'].flatten()
t = data['t'].flatten()
usol = data['usol']

pinn = model.ConventionalAllenCahnPINN(t,x,usol, f=5)

pinn.train(10)

print(pinn.log['ic_loss_grad'])