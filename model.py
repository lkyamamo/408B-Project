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
        self.x_initial = x.T
        self.t_initial = np.array([0.]*len(x)).T
        self.usol_initial = usol[0].T

        print(f"t_initial: {self.t_initial.shape}")
        print(f"x_initial: {self.x_initial.shape}")
        print(f"usol_initial: {self.usol_initial.shape}")

        self.t_mesh = []
        self.x_mesh = []
        self.usol_train = []

        # restructure input data
        # shape of data (t_train, x_train, usol_train): (102912,0)
        for i in range(len(t)):
            self.t_mesh += [t[i]]*(len(x))
            self.x_mesh.append(x)
            self.usol_train.append(usol[i])

        self.t_mesh = tf.cast(tf.convert_to_tensor(np.array(self.t_mesh)), tf.float32)
        self.x_mesh = tf.cast(tf.convert_to_tensor(np.array(self.x_mesh).flatten()), tf.float32)
        self.usol_train = tf.cast(tf.convert_to_tensor(np.array(self.usol_train).flatten()), tf.float32)

        self.x_train = []
        self.t_train = []

        #np.save(file="t_train", arr=self.t_train)
        #np.save(file="x_train", arr=self.x_train)
        
        print(f"x_mesh: {self.x_mesh.shape}")
        print(f"t_mesh: {self.t_mesh.shape}")
        print(f"usol_train: {self.usol_train.shape}")

        # boundary condition inputs ((t,-1) and (t,1))
        self.negative_bound = np.stack([t, np.array([-1]*len(t))]).T
        self.positive_bound = np.stack([t, np.array([1]*len(t))]).T

        print(f"negative_bound: {self.negative_bound.shape}")
        print(f"positive_bound: {self.positive_bound.shape}")

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
        self.losses = {'total_loss':[], 'boundary_loss':[], 'residual_loss':[], 'initial_loss':[]}


    def loss_function(self, u, t_tape, x_tape):
        print("Calculating Loss Value")
        u_t = tf.expand_dims(t_tape.gradient(u, self.t_train, unconnected_gradients=tf.UnconnectedGradients.ZERO), axis=-1)
        u_x = tf.expand_dims(x_tape.gradient(u, self.x_train, unconnected_gradients=tf.UnconnectedGradients.ZERO), axis=-1)
        u_xx = tf.expand_dims(x_tape.gradient(u_x, self.x_train, unconnected_gradients=tf.UnconnectedGradients.ZERO), axis=-1)

        # residual loss 
        loss_r = tf.math.square(tf.math.reduce_mean(u_t - 0.0001 * u_xx + 5 * tf.math.pow(u_xx,[3]) - 5*u))

        # initial condition loss
        loss_ic = tf.math.square(tf.math.reduce_mean((self.usol_initial - self.model(tf.stack(self.t_initial, self.x_intial)))))

        # boundary condition loss
        loss_bc = tf.math.square(tf.math.reduce_mean((u[0::512] - u[511:512]))) + tf.math.square(tf.math.reduce_mean((u_x[0::512] - u_x[511::512])))

        total_loss = self.lambda_r*loss_r + self.lambda_ic*loss_ic + self.lambda_bc*loss_bc

        print(total_loss)
        print(loss_r)
        print(loss_ic)
        print(loss_bc)

        self.losses['total_loss'].append(total_loss.numpy().item())
        self.losses['residual_loss'].append(loss_r.numpy().item())
        self.losses['initial_loss'].append(loss_ic.numpy().item())
        self.losses['boundary_loss'].append(loss_bc.numpy().item())

        return total_loss, loss_r, loss_ic, loss_bc

    def train(self, max_epochs):
        optimizer = tf.keras.optimizers.Adam()

        for epoch in range(max_epochs):
            print(f"Epoch {epoch + 1}")

            self.x_train = tf.random.shuffle(self.x_mesh)[:self.batch_size]
            self.t_train = tf.random.shuffle(self.t_mesh)[:self.batch_size]

            with tf.GradientTape(persistent=True) as loss_tape:
                loss_tape.reset()
                loss_tape.watch(self.model.trainable_variables)
                with tf.GradientTape(persistent=True) as x_tape, tf.GradientTape(persistent=True) as t_tape:
                    x_tape.reset()
                    t_tape.reset()

                    t_tape.watch(self.t_train)
                    x_tape.watch(self.x_train)

                    inputs = tf.stack([self.t_train,self.x_train],axis=1)
                
                    u = self.model(inputs)

                total_loss, loss_r, loss_ic, loss_bc = self.loss_function(u, t_tape, x_tape)

            print("Updating Weights")

            # global weights
            if(epoch + 1 % self.f == 0):
                ic_grad = loss_tape.gradient(loss_ic, self.model.trainable_variables, unconnected_gradients=tf.UnconnectedGradients.ZERO)
                bc_grad = loss_tape.gradient(loss_bc, self.model.trainable_variables, unconnected_gradients=tf.UnconnectedGradients.ZERO)
                r_grad = loss_tape.gradient(loss_r, self.model.trainable_variables, unconnected_gradients=tf.UnconnectedGradients.ZERO)

                ic_grad_norm = self.compute_L2_norm(ic_grad)
                bc_grad_norm = self.compute_L2_norm(bc_grad)
                r_grad_norm = self.compute_L2_norm(r_grad)

                self.lambda_ic = (ic_grad_norm + bc_grad_norm + r_grad_norm)/ic_grad_norm
                self.lambda_bc = (ic_grad_norm + bc_grad_norm + r_grad_norm)/bc_grad_norm
                self.lambda_r = (ic_grad_norm + bc_grad_norm + r_grad_norm)/r_grad_norm

            total_grad = loss_tape.gradient(total_loss, self.model.trainable_variables, unconnected_gradients=tf.UnconnectedGradients.ZERO)

            optimizer.apply_gradients(zip(total_grad, self.model.trainable_variables))

    def evaluate(self):
        inputs = tf.stack([self.t_mesh,self.x_mesh],axis=1)
        output = self.model(inputs)
        
        return tf.reshape(output,(201,512))
    
    def compute_L2_norm(tensor_list):
        output = 0
        for tensor in tensor_list: 
            output += tf.math.sqrt(tf.math.reduce_sum(tf.math.square(tensor)))

        return output.numpy()
