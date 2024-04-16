import numpy as np
import tensorflow as tf

class ConventionalAllenCahnPINN():
    def __init__(self, t, x, usol):
        print("Shape of Data")
        print(f"t: {t.shape}")
        print(f"x: {x.shape}")
        print(f"usol: {usol.shape}")

        # initial condition inputs
        self.x_initial = x.T
        self.t_initial = np.array([0.]*len(x)).T
        self.usol_initial = usol[0].T

        print(f"t_initial: {self.t_initial.shape}")
        print(f"x_initial: {self.x_initial.shape}")
        print(f"usol_initial: {self.usol_initial.shape}")

        self.t_train = []
        self.x_train = []
        self.usol_train = []

        # restructure input data
        # shape of data (t_train, x_train, usol_train): (102912,0)
        for i in range(len(t)):
            self.t_train += [t[i]]*(len(x))
            self.x_train.append(x)
            self.usol_train.append(usol[i])

        self.t_train = tf.cast(tf.convert_to_tensor(np.array(self.t_train)), tf.float32)
        self.x_train = tf.cast(tf.convert_to_tensor(np.array(self.x_train).flatten()), tf.float32)
        self.usol_train = tf.cast(tf.convert_to_tensor(np.array(self.usol_train).flatten()), tf.float32)

        #np.save(file="t_train", arr=self.t_train)
        #np.save(file="x_train", arr=self.x_train)
        
        print(f"x_train: {self.x_train.shape}")
        print(f"t_train: {self.t_train.shape}")
        print(f"usol_train: {self.usol_train.shape}")

        # boundary condition inputs ((t,-1) and (t,1))
        self.negative_bound = np.stack([t, np.array([-1]*len(t))]).T
        self.positive_bound = np.stack([t, np.array([1]*len(t))]).T

        print(f"negative_bound: {self.negative_bound.shape}")
        print(f"positive_bound: {self.positive_bound.shape}")

        # initialize model
        self.model = tf.keras.models.Sequential([
            tf.keras.layers.Dense(2, dtype=tf.float32),
            tf.keras.layers.Dense(200, activation='tanh', dtype=tf.float32),
            tf.keras.layers.Dense(200, activation='tanh', dtype=tf.float32),
            tf.keras.layers.Dense(200, activation='tanh', dtype=tf.float32),
            tf.keras.layers.Dense(200, activation='tanh', dtype=tf.float32),
            tf.keras.layers.Dense(1, name='output', dtype=tf.float32),
        ])


    def loss_function(self, u, t_tape, x_tape):
        print("Calculating Loss Value")
        u_t = tf.expand_dims(t_tape.gradient(u, self.t_train), axis = -1)
        u_x = tf.expand_dims(x_tape.gradient(u, self.x_train), axis=-1)
        u_xx = tf.expand_dims(x_tape.gradient(u_x, self.x_train), axis=-1)

        print(f"u_t: {u_t.shape}")
        print(f"u_x: {u_x.shape}")
        print(f"u_xx: {u_xx.shape}")

        # residue loss 
        loss_r = tf.math.square(tf.math.reduce_mean(u_t - 0.0001 * u_xx + 5 * tf.math.pow(u_xx,[3]) - 5*u))

        # initial condition loss
        loss_ic = tf.keras.losses.MSE(self.usol_initial, self.model(np.stack([self.t_initial, self.x_initial]).T))

        # boundary condition loss
        loss_bc = tf.math.square(tf.math.reduce_mean((u[0::512] - u[511:512]))) + tf.math.square(tf.math.reduce_mean((u_x[0::512] - u_x[511::512])))

        return loss_r + loss_ic + loss_bc

    def train(self, max_epochs):
        optimizer = tf.keras.optimizers.Adam()

        for epoch in range(max_epochs):
            print(f"Epoch {epoch + 1}")
            with tf.GradientTape(persistent=True) as x_tape, tf.GradientTape(persistent=True) as t_tape:
                x_tape.reset()
                t_tape.reset()

                t_tape.watch(self.t_train)
                x_tape.watch(self.x_train)

                inputs = tf.stack([self.t_train,self.x_train],axis=1)

                print(f"inputs: {inputs.shape}")

                u = self.model(inputs)

                with tf.GradientTape() as loss_tape:

                    loss_val = self.loss_function(u, t_tape, x_tape)

                print("Updating Weights")

                grads = loss_tape.gradient(loss_val, self.model.trainable_variables)

                optimizer.apply_gradients(zip(grads, self.model.trainable_variables))