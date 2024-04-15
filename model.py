import numpy as np
import tensorflow as tf

class ConventionalAllenCahnPINN():
    def __init__(self, t, x, usol):
        print("Shape of Data")
        print(f"t: {t.shape}")
        print(f"x: {x.shape}")
        print(f"usol: {usol.shape}")

        # initial condition inputs
        self.x_initial = x
        self.t_initial = np.array([0.]*len(x))
        self.usol_initial = usol[0]

        self.t_train = []
        self.x_train = []
        self.usol_train = []

        # restructure input data
        # shape of data (t_train, x_train, usol_train): (102912,0)
        for i in range(len(t)):
            self.t_train += [t[i]]*(len(x))
            self.x_train.append(x)
            self.usol_train.append(usol[i])

        self.t_train = tf.convert_to_tensor(np.array(self.t_train))
        self.x_train = tf.convert_to_tensor(np.array(self.x_train).flatten())
        self.usol_train = tf.convert_to_tensor(np.array(self.usol_train).flatten())
        
        print(self.x_train.shape)
        print(self.t_train.shape)
        print(self.usol_train.shape)

        # boundary condition inputs ((t,-1) and (t,1))
        self.negative_bound = np.stack([t, np.array([-1]*len(t))]).T
        self.positive_bound = np.stack([t, np.array([1]*len(t))]).T

        print(self.negative_bound.shape)
        print(self.positive_bound.shape)

        # initialize model
        self.model = tf.keras.models.Sequential([
            tf.keras.layers.Dense(2),
            tf.keras.layers.Dense(200, activation='tanh'),
            tf.keras.layers.Dense(200, activation='tanh'),
            tf.keras.layers.Dense(200, activation='tanh'),
            tf.keras.layers.Dense(200, activation='tanh'),
            tf.keras.layers.Dense(1, name='output'),
        ])


    def loss_function(self, u, t_tape, x_tape):
        u_t = t_tape.gradient(u, self.t_train)
        u_x = x_tape.gradient(u, self.x_train)
        u_xx = x_tape.gradient(u_x, self.x_train)

        # residue loss
        loss_r = tf.math.square(tf.math.reduce_mean(u_t - 0.0001*u_xx + 5*u*u*u -5*u))

        # initial condition loss
        loss_ic = tf.keras.losses.MSE(self.usol_initial, self.model(np.stack([self.t_initial, self.x_initial])))

        # boundary condition loss
        loss_bc = tf.keras.losses.MSE(self.model(tf.concat([self.t_initial, self.x_initial], axis=1))) + tf.keras.losses.MSE()

        return loss_r + loss_ic + loss_bc

    def train(self, max_epochs):
        for epoch in max_epochs:
            with tf.GradientTape(persistent=True) as x_tape, tf.GradientTape(persistent=True) as t_tape:
                x_tape.reset()
                t_tape.reset()

                t_tape.watch(self.t_train)
                x_tape.watch(self.x_train)

                inputs = tf.concat([self.t_train,self.x_train],axis=0)
                inputs = tf.transpose(inputs)

                u = self.model(inputs)

                with tf.GradientTape() as loss_tape:

                    loss_val = self.loss_function(u, t_tape, x_tape)

                grads = loss_tape.gradient(loss_val, self.model)

                self.model.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))