import tensorflow as tf

#from Tetsuto Nagashima's implementation based off of AME 508 course work
class AllenCahn():
  def __init__(self, t, x, u0):

    #initial condition
    self.u0 = u0

    #time and spacial locations
    self.t = t
    self.x = x

    self.t0 = t[0]
    self.tfinal = t[-1]
    
def get_model():

  initializer = tf.keras.initalizers.GlorotUniform()

  inputs = tf.keras.Input(shape=(), name="Conventional_PINN")
  x0 = tf.keras.layers.Flatten()(inputs)
  x1 = tf.keras.layers.Dense(200, activation='tanh', kernel_initializer=initializer)(x0)
  x2 = tf.keras.layers.Dense(200, activation='tanh', kernel_initializer=initializer)(x1)
  x3 = tf.keras.layers.Dense(200, activation='tanh', kernel_initializer=initializer)(x2)
  x4 = tf.keras.layers.Dense(200, activation='tanh', kernel_initializer=initializer)(x3)
  outputs = tf.keras.layers.Dense(1, name="predictions")(x4)

  model = tf.keras.Model(inputs=inputs, outputs=outputs, name='Model')
   
  model.compile(
      optimizer=tf.keras.optimizers.Adam(),
      loss=AC_loss_function,
      metrics=['accuracy']
  )

def AC_loss_function():
  uic = 
  Lic = tf.math.reduce_sum(tf.math.squared_difference(u(0,x), g))

def get_matlab_data():
