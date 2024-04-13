import numpy as np

arr = np.array([1, 2, 3])  # 1-dimensional array
print(arr.shape)  # Output: (3,)

# Adding a new axis using np.newaxis
new_arr = arr[:, np.newaxis]
print(new_arr.shape)  # Output: (3, 1)

# Now you can perform operations on new_arr that require 2-dimensional arrays, like matrix multiplication.

x = [0,1,2,3,4]
t = [0,1,2,3,4]

x_initial = []
t_initial = []

for element in t:
    t_initial += [element]*(len(x))
    x_initial += x

print(x_initial)
print(t_initial)