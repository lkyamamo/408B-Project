import numpy as np

class AllenCahnPINN():
    def __init__(self, t, x, usol):
        print("Shape of Data")
        print(f"t: {t.shape}")
        print(f"x: {x.shape}")
        print(f"usol: {usol.shape}")

        # initial conditions
        self.x_initial = x
        self.t_train = np.array([0.]*len(x))
        self.usol_initial = usol[0]

        self.t_train = []
        self.x_train = []
        self.usol_train = []
        
        for i in range(len(t)):
            self.t_train += [t[i]]*(len(x))
            self.x_train.append(x)
            self.usol_train.append(usol[i])

        self.t_train = np.array(self.t_train)
        self.x_train = np.array(self.x_train).flatten()
        self.usol_train = np.array(self.usol_train).flatten()
        

        print(self.x_train.shape)
        print(self.t_train.shape)
        print(self.usol_train.shape)


        # boundary conditions
