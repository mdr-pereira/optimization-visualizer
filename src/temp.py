import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation, rc
import sympy as sp
import pandas as pd

rc('animation', html='html5')

class Model(object):
    def __init__(self, w_init=-1.0, b_init=-1.0):
        self.W = tf.Variable(w_init, dtype=tf.float32)
        self.b = tf.Variable(b_init, dtype=tf.float32)

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

def plot_regression(X, y, SEED=3141, LR=0.1, EPOCHS=100):

    model = Model()

    # Collect the history of W-values and b-values to plot later
    Ws, bs, xs, ys, ls = [], [], [], [], []

    fig = plt.figure(dpi=100, figsize=(12, 5))

    # Regression Line
    ax1 = fig.add_subplot(131)
    ax1.set_title("Fitted Line")
    ax1.set_xlabel("Features")
    ax1.set_ylabel("Rent")
    p10, = ax1.plot(X, y, 'r.', alpha=0.1)  # full dataset
    p11, = ax1.plot([], [], 'C3.')  # batch, color Red
    p12, = ax1.plot([], [], 'k')  # fitted line, color Black

    # Loss
    ax2 = fig.add_subplot(132)
    ax2.set_title("Training Loss")
    ax2.set_xlabel("Batches Seen")
    ax2.set_xlim(0, EPOCHS)
    ax2.set_ylim(0, 100)
    p20, = ax2.plot([], [], 'C0')  # color Blue

    # Weights
    ax3 = fig.add_subplot(133)
    ax3.set_title("Weights")
    ax3.set_xlabel("Batches Seen")
    ax3.set_xlim(0, EPOCHS)
    ax3.set_ylim(-5000, 5000)
    p30, = ax3.plot([], [], 'C5')  # W color Brown
    p30.set_label('W')
    p31, = ax3.plot([], [], 'C8')  # b color Green
    p31.set_label('b')
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
        inputs = tf.linspace(X.min(), X.max(), 30)
        p12.set_data(inputs, Ws[-1] * inputs + bs[-1])
        p20.set_data(range(epoch), ls)
        p30.set_data(range(epoch), Ws)
        p31.set_data(range(epoch), bs)

        train(model, x, y, learning_rate=LR)

        return p11, p12, p20

    ds = (tf.data.Dataset
          .from_tensor_slices((X, y))
          .shuffle(1000, seed=SEED)
          .batch(BATCH_SIZE)
          .repeat())
    ds = iter(ds)

    anim = animation.FuncAnimation(fig, update, frames=range(1, EPOCHS), init_func=init, blit=True, interval=100)
    plt.close()
    return anim

# Load your dataset
df=pd.read_csv('../House_Rent_Dataset.csv')

# Extract features and target
X = df[['BHK', 'Size', 'Bathroom']].values
y = df['Rent'].values

# Scale the features if necessary
# X = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))

# Plot the regression
plot_regression(X, y, LR=0.1, EPOCHS=100)
