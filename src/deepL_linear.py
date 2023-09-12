NUM_EXAMPLES = 1000
BATCH_SIZE = 20
EPOCHS = 100 # actually steps
LR = 0.1


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import animation, rc
import sympy as sp
rc('animation', html='html5')

class Model(object):
  def __init__(self, w_init=-1.0, b_init=-1.0):
    self.W = tf.Variable(w_init)
    self.b = tf.Variable(b_init)

  def __call__(self, x):
    return self.W * x + self.b

def loss(target_y, predicted_y):
  return tf.reduce_mean(tf.square(target_y - predicted_y))

def train(model, inputs, outputs, learning_rate):
  with tf.GradientTape() as t:
    current_loss = loss(outputs, model(inputs))
  dW, db = t.gradient(current_loss, [model.W, model.b])
  model.W.assign_sub(learning_rate * dW)
  model.b.assign_sub(learning_rate * db)


TRUE_W = 3.0
TRUE_b = 2.0
SEED = 3141
XBOUND_MIN = -30
XBOUND_MAX = 30

equation_str = input("Enter a linear equation (e.g.,3*x - 5): ")
# Define a symbolic variable
x = sp.symbols('x')
# Parse the input string to create a function
try:
    equation = sp.sympify(equation_str)
    fun = sp.lambdify(x, equation, 'numpy')
    slope = equation.as_poly().all_coeffs()[0]
    intercept = equation.as_poly().all_coeffs()[1]
    
except sp.SympifyError:
    print("Invalid input. Please enter a valid polynomial equation.")
    exit(1)

if abs(slope) >= abs(intercept):
    bound = float(abs(slope))
else:
    bound = float(abs(intercept))
print(bound)
inputs  = tf.random.normal(shape=[NUM_EXAMPLES], seed=SEED)
noise   = tf.random.normal(shape=[NUM_EXAMPLES], seed=SEED+1)
outputs = fun(inputs.numpy()) + noise

ds = (tf.data.Dataset
      .from_tensor_slices((inputs, outputs))
      .shuffle(1000, seed=SEED)
      .batch(BATCH_SIZE)
      .repeat())
ds = iter(ds)

model = Model()

# Collect the history of W-values and b-values to plot later
Ws, bs, xs, ys, ls = [], [], [], [], []

fig = plt.figure(dpi=100, figsize=(8, 3))

# Regression Line
ax1 = fig.add_subplot(131)
ax1.set_title("Fitted Line")
ax1.set_xlabel("x")
ax1.set_ylabel("y")
# ax1.set_xlim(-3, 2.5)
# ax1.set_ylim(-8, 11)
p10, = ax1.plot(inputs, outputs, 'r.', alpha=0.1) # full dataset
p11, = ax1.plot([], [], 'C3.') # batch, color Red
p12, = ax1.plot([], [], 'k') # fitted line, color Black

# Loss
ax2 = fig.add_subplot(132)
ax2.set_title("Training Loss")
ax2.set_xlabel("Batches Seen")
ax2.set_xlim(0, EPOCHS)
ax2.set_ylim(0, 40)
p20, = ax2.plot([], [], 'C0') # color Blue

# Weights
ax3 = fig.add_subplot(133)
ax3.set_title("Weights")
ax3.set_xlabel("Batches Seen")
ax3.set_xlim(0, EPOCHS)
ax3.set_ylim(-(bound+1),bound+1)
ax3.plot(range(EPOCHS), [float(slope) for _ in range(EPOCHS)], 'C5--')
ax3.plot(range(EPOCHS), [float(intercept) for _ in range(EPOCHS)], 'C8--')
p30, = ax3.plot([], [], 'C5') # W color Brown
p30.set_label('Slope')
p31, = ax3.plot([], [], 'C8') # b color Green
p31.set_label('bIntercept')
ax3.legend()

fig.tight_layout()

def init():
    return [p10]

def update(epoch):
  x, y = next(ds)
  y_pred = model(x)
  current_loss = loss(y, y_pred)

  Ws.append(model.W.numpy())
  bs.append(model.b.numpy())
  xs.append(x.numpy())
  ys.append(y_pred.numpy())
  ls.append(current_loss.numpy())
  p11.set_data(x.numpy(), y.numpy())
  inputs = tf.linspace(XBOUND_MIN, XBOUND_MAX, 30)
  p12.set_data(inputs, Ws[-1]*inputs + bs[-1])
  p20.set_data(range(epoch), ls)
  p30.set_data(range(epoch), Ws)
  p31.set_data(range(epoch), bs)
  
    
  train(model, x, y, learning_rate=LR)


  return p11, p12, p20

ani = animation.FuncAnimation(fig, update, frames=range(1, EPOCHS), init_func=init, blit=True, interval=100)
plt.close()
ani
