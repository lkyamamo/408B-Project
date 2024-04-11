import tensorflow as tf
import os
import scipy.io

working_dir = os.getcwd()

file_path = os.path.join(working_dir, 'jaxpi', 'examples', 'allen_cahn','data', 'allen_cahn.mat')

data = scipy.io.loadmat(file_path)
print(f"tt: {data['t']}")
print(f"uu: {data['usol']}")
print(f"x: {data['x']}")

# get data for model
x = data['x']
usol = data['usol'][0]

# create basic model with some initialization
initializer = tf.keras.initializers.GlorotUniform()

inputs = tf.keras.Input(shape=(), name="Conventional_PINN")
x0 = tf.keras.layers.Flatten()(inputs)
x1 = tf.keras.layers.Dense(5, activation='tanh', kernel_initializer=initializer)(x0)
x2 = tf.keras.layers.Dense(5, activation='tanh', kernel_initializer=initializer)(x1)
outputs = tf.keras.layers.Dense(usol.shape, name="predictions")(x2)

model = tf.keras.Model(inputs=inputs, outputs=outputs, name='Model')

model.compile(
    optimizer=tf.keras.optimizers.Adam(),
    loss=tf.keras.losses.CategoricalCrossentropy(),
    metrics=['accuracy']
)

# forward pass
forward = model.evaluate(x)

# calculate gradient
with tf.GradientTape() as tape:
    grad = tape.gradient(forward, x)

print(grad)