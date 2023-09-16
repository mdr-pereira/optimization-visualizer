import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import animation, rc
import sympy as sp
rc('animation', html='html5')

#SGD FOR QUADRATIC FUCNTIONS
#parameters
NUM_EXAMPLES = 1024
BATCH_SIZE = 64
EPOCHS = 150 # actually steps
LR = 0.01
SEED = 3141

equation_str = input("Enter a quadratic equation (e.g: 2*x**2 - 5*x + 8): ")
# Define a symbolic variable
x = sp.symbols('x')
# Parse the input string to create a function
try:
    equation = sp.sympify(equation_str)
    fun = sp.lambdify(x, equation, 'numpy')
except sp.SympifyError:
    print("Invalid input. Please enter a valid polynomial equation.")
    exit(1)

# Define Model
model = keras.Sequential([
    layers.Input(shape=(1,)),
    layers.Dense(16, activation='relu'),
    layers.Dense(8, activation='relu'),
    layers.Dense(1)
])
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=LR), 
    loss='mse')

#Data for the model
inputs  = tf.random.normal(shape=[NUM_EXAMPLES], seed=SEED)
noise   = tf.random.normal(shape=[NUM_EXAMPLES], seed=SEED+1)
outputs = fun(inputs.numpy()) + noise
outputs = tf.squeeze(outputs)

ds = (tf.data.Dataset
      .from_tensor_slices((inputs, outputs))
      .repeat()
      .shuffle(1000, seed=SEED)
      .batch(BATCH_SIZE))
ds = iter(ds)
# Collect the history of W-values and b-values to plot later
Ws, bs, xs, ys, ls = [], [], [], [], []
# Create Figure
fig = plt.figure(dpi=150, figsize=(8, 3))

# Regression Curve
ax1 = fig.add_subplot(121)
ax1.set_title("Fitted Curve")
ax1.set_xlabel("x")
ax1.set_ylabel("y")
p10, = ax1.plot(inputs, outputs, 'r.', alpha=0.1) # full dataset
p11, = ax1.plot([], [], 'C3.') # batch
p12, = ax1.plot([], [], 'k') # fitted line

# Loss
ax2 = fig.add_subplot(122)
ax2.set_title("Training Loss")
ax2.set_xlabel("Batches Seen")
ax2.set_xlim(0, EPOCHS)
ax2.set_ylim(0, 40)
p20, = ax2.plot([], [], 'C0')

fig.tight_layout()

def init():
    return [p10]

def update(epoch):
    x, y = next(ds)
    y_pred = model(x)
    current_loss = model.evaluate(x, y)
    x = tf.squeeze(x)
    y = tf.squeeze(y)
    y_pred = tf.squeeze(y_pred)
    
    xs.append(x.numpy())
    ys.append(y_pred.numpy())
    ls.append(current_loss)
    p11.set_data(x.numpy(), y.numpy())
    inputs = tf.linspace(-3.0, 3.0, 30)
    p12.set_data(inputs, model.predict(inputs))
    p20.set_data(range(epoch), ls)

    model.train_on_batch(x, y)

    return p11, p12, p20


ani = animation.FuncAnimation(fig, update, frames=range(1, EPOCHS), init_func=init, blit=True, interval=100)
plt.close()
ani
    