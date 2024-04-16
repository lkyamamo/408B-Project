import numpy as np
import tensorflow as tf

class ConventionalAllenCahnPINN():
    def __init__(self, t, x, usol, batch_size=64, f=1000, alpha=0.9, gamma=1.0, epsilon=1.0):

        # hyperparameters
        self.f = f
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.batch_size = batch_size

        # global weights
        self.lambda_ic = 1
        self.lambda_bc = 1
        self.lambda_r = 1

        print("Shape of Data")
        print(f"t: {t.shape}")
        print(f"x: {x.shape}")
        print(f"usol: {usol.shape}")

        # initial condition inputs
        self.x_initial = tf.cast(tf.convert_to_tensor(x.T), tf.float32)
        self.t_initial = tf.cast(tf.convert_to_tensor(np.array([0.]*len(x)).T), tf.float32)
        self.usol_initial = tf.cast(tf.convert_to_tensor(usol[0].T), tf.float32)

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
        self.x_negative_bound = tf.cast(tf.convert_to_tensor(np.array([-1]*len(t)).T), tf.float32)
        self.x_positive_bound = tf.cast(tf.convert_to_tensor(np.array([1]*len(t)).T), tf.float32)
        self.t_bc = tf.cast(tf.convert_to_tensor(np.array(t).T), tf.float32)

        print(f"x_negative_bound: {self.x_negative_bound.shape}")
        print(f"x_positive_bound: {self.x_positive_bound.shape}")
        print(f"t_bc: {self.t_bc.shape}")

        # initialize model
        self.model = tf.keras.models.Sequential([
            #tf.keras.layers.Dense(2, dtype=tf.float32),
            tf.keras.layers.Dense(200, activation='tanh', dtype=tf.float32, kernel_initializer='glorot_uniform'),
            tf.keras.layers.Dense(200, activation='tanh', dtype=tf.float32, kernel_initializer='glorot_uniform'),
            tf.keras.layers.Dense(200, activation='tanh', dtype=tf.float32, kernel_initializer='glorot_uniform'),
            tf.keras.layers.Dense(200, activation='tanh', dtype=tf.float32, kernel_initializer='glorot_uniform'),
            tf.keras.layers.Dense(1, name='output', dtype=tf.float32),
        ])
        self.model.build((None,2))

        # logs
        self.log = {'total_loss':[], 'boundary_loss':[], 'residual_loss':[], 'initial_loss':[], 
                    'ic_loss_grad':[], 'bc_loss_grad':[], 'r_loss_grad':[]}


    def loss_function(self):
        print("Calculating Loss Value")

        # initial condition loss

        u = self.model(tf.stack([self.t_initial, self.x_initial], axis=1))
        loss_ic = tf.math.square(tf.math.reduce_mean((self.usol_initial - u)))

        # boundary condition loss
        with tf.GradientTape(persistent=True) as x_tape, tf.GradientTape(persistent=True) as t_tape:
            t_tape.reset()
            x_tape.reset()
            x_tape.watch(self.x_positive_bound)
            x_tape.watch(self.x_negative_bound)

            u_pos = self.model(tf.stack([self.t_bc, self.x_positive_bound], axis=1))
            u_neg = self.model(tf.stack([self.t_bc, self.x_negative_bound], axis=1))

        u_pos_x = tf.expand_dims(x_tape.gradient(u, self.x_positive_bound, unconnected_gradients=tf.UnconnectedGradients.ZERO), axis=-1)
        u_neg_x = tf.expand_dims(x_tape.gradient(u, self.x_negative_bound, unconnected_gradients=tf.UnconnectedGradients.ZERO), axis=-1)

        loss_bc = tf.math.square(tf.math.reduce_mean((u_pos - u_neg))) + tf.math.square(tf.math.reduce_mean((u_pos_x - u_neg_x)))

        # residual loss 
        with tf.GradientTape(persistent=True) as x_tape, tf.GradientTape(persistent=True) as t_tape:
            t_tape.reset()
            x_tape.reset()
            t_tape.watch(self.t_train)
            x_tape.watch(self.x_train)

            u = self.model(tf.stack([self.t_train, self.x_train], axis=1))

        u_t = tf.expand_dims(t_tape.gradient(u, self.t_train, unconnected_gradients=tf.UnconnectedGradients.ZERO), axis=-1)
        u_x = tf.expand_dims(x_tape.gradient(u, self.x_train, unconnected_gradients=tf.UnconnectedGradients.ZERO), axis=-1)
        u_xx = tf.expand_dims(x_tape.gradient(u_x, self.x_train, unconnected_gradients=tf.UnconnectedGradients.ZERO), axis=-1)
        
        loss_r = tf.math.square(tf.math.reduce_mean(u_t - 0.0001 * u_xx + 5 * tf.math.pow(u_xx,[3]) - 5*u))

        total_loss = self.lambda_r*loss_r + self.lambda_ic*loss_ic + self.lambda_bc*loss_bc

        self.log['total_loss'].append(total_loss.numpy().item())
        self.log['residual_loss'].append(loss_r.numpy().item())
        self.log['initial_loss'].append(loss_ic.numpy().item())
        self.log['boundary_loss'].append(loss_bc.numpy().item())

        return total_loss, loss_r, loss_ic, loss_bc
    
    def residual_loss(self, t, x):
        with tf.GradientTape(persistent=True) as x_tape, tf.GradientTape(persistent=True) as t_tape:
            t_tape.reset()
            x_tape.reset()
            t_tape.watch(t)
            x_tape.watch(x)

            u = self.model(tf.stack([t, x], axis=1))

        u_t = tf.expand_dims(t_tape.gradient(u, t, unconnected_gradients=tf.UnconnectedGradients.ZERO), axis=-1)
        u_x = tf.expand_dims(x_tape.gradient(u, x, unconnected_gradients=tf.UnconnectedGradients.ZERO), axis=-1)
        u_xx = tf.expand_dims(x_tape.gradient(u_x, x, unconnected_gradients=tf.UnconnectedGradients.ZERO), axis=-1)
        
        loss_r = tf.math.square(tf.math.reduce_mean(u_t - 0.0001 * u_xx + 5 * tf.math.pow(u_xx,[3]) - 5*u))

        return(loss_r)
        

    def train(self, max_epochs):
        optimizer = tf.keras.optimizers.Adam()

        for epoch in range(max_epochs):
            print(f"Epoch {epoch + 1}")

            with tf.GradientTape(persistent=True) as loss_tape:
                loss_tape.reset()
                loss_tape.watch(self.model.trainable_variables)
                
                total_loss, loss_r, loss_ic, loss_bc = self.loss_function()

            print("Updating Weights")

            # global weights
            if((epoch + 1) % self.f == 0):
                print("Updateing Global Weights")
                ic_grad = loss_tape.gradient(loss_ic, self.model.trainable_variables, unconnected_gradients=tf.UnconnectedGradients.ZERO)
                bc_grad = loss_tape.gradient(loss_bc, self.model.trainable_variables, unconnected_gradients=tf.UnconnectedGradients.ZERO)
                r_grad = loss_tape.gradient(loss_r, self.model.trainable_variables, unconnected_gradients=tf.UnconnectedGradients.ZERO)
                
                ic_grad_norm = self.compute_L2_norm(ic_grad)
                bc_grad_norm = self.compute_L2_norm(bc_grad)
                r_grad_norm = self.compute_L2_norm(r_grad)

                self.log['ic_loss_grad'].append(ic_grad_norm)
                self.log['bc_loss_grad'].append(bc_grad_norm)
                self.log['r_loss_grad'].append(r_grad_norm)

                self.lambda_ic = self.alpha*self.lambda_ic + (1-self.alpha)*(ic_grad_norm + bc_grad_norm + r_grad_norm)/ic_grad_norm
                self.lambda_bc = self.alpha*self.lambda_bc + (1-self.alpha)*(ic_grad_norm + bc_grad_norm + r_grad_norm)/bc_grad_norm
                self.lambda_r = self.alpha*self.lambda_r + (1-self.alpha)*(ic_grad_norm + bc_grad_norm + r_grad_norm)/r_grad_norm

            total_grad = loss_tape.gradient(total_loss, self.model.trainable_variables, unconnected_gradients=tf.UnconnectedGradients.ZERO)

            optimizer.apply_gradients(zip(total_grad, self.model.trainable_variables))

    def evaluate(self):
        inputs = tf.stack([self.t_train,self.x_train],axis=1)
        output = self.model(inputs)
        
        return tf.reshape(output,(201,512))
    
    def compute_L2_norm(self, tensor_list):
        output = 0
        for tensor in tensor_list: 
            output += tf.math.sqrt(tf.math.reduce_sum(tf.math.square(tensor)))

        return output.numpy()
