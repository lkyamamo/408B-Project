import scipy.io
import os

working_dir = os.getcwd()

"""
print("PINNs AC data")


file_path = os.path.join(working_dir, 'PINNs', 'main', 'Data', 'AC.mat')

data = scipy.io.loadmat(file_path)

print(f"tt: {data['tt']}")
print(f"uu: {data['uu']}")
print(f"x: {data['x']}")

"""
print("jaxpi AC data")
file_path = os.path.join(working_dir, 'jaxpi', 'examples', 'allen_cahn','data', 'allen_cahn.mat')

data = scipy.io.loadmat(file_path)
print(f"tt: {data['t']} {data['t'].shape}")
print(f"uu: {data['usol']} {data['usol'].shape}")
print(f"x: {data['x']} {data['x'].shape}")