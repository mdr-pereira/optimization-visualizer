"""
BHK: Number of Bedrooms, Hall, Kitchen.
Rent: Price of the Houses/Apartments/Flats.
Size: Size of the Houses/Apartments/Flats in Square Feet.
Floor: Houses/Apartments/Flats situated in which Floor and Total Number of Floors (Example: Ground out of 2, 3 out of 5, etc.)
Area Type: Size of the Houses/Apartments/Flats calculated on either Super Area or Carpet Area or Build Area.
Area Locality: Locality of the Houses/Apartments/Flats.
City: City where the Houses/Apartments/Flats are Located.
Furnishing Status: Furnishing Status of the Houses/Apartments/Flats, either it is Furnished or Semi-Furnished or Unfurnished.
Tenant Preferred: Type of Tenant Preferred by the Owner or Agent.
Bathroom: Number of Bathrooms.
Point of Contact: Whom should you contact for more information regarding the Houses/Apartments/Flats.
"""


import pandas as pd
import numpy as np

# visualization libraries
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from matplotlib.animation import FuncAnimation
from matplotlib import animation, rc
rc('animation', html='html5')

# Regression libraries
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_squared_error
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers






df=pd.read_csv(r"C:\Users\studi\OneDrive\Desktop\Applied computer science\Semester_2\viz\optimization-visualizer\House_Rent_Dataset.csv")



print(df.head())
print(df.shape)

# Pre-Processing
df.isnull().sum()

df.duplicated().sum()

print(df.describe())

print(df.dtypes)
print(df.info())




# Number of House in Each City which is Available for Rent
city_df = df['City'].value_counts()
city_df
city_df = df['City'].value_counts()
city_df = df['City'].value_counts()
city_df
fig = px.bar(city_df, x=city_df.index, y=city_df.values, color=city_df.index, 
       title='Number of Houses in Each City which is Available for Rent', text=city_df.values)
fig.update_traces(width=0.3)
fig. update_layout(showlegend=False)
fig.update_layout(
    xaxis_title="City",
    yaxis_title="Count",
    font = dict(size=17))
fig.show()


# Types of Tenant Preferred
tenant_df = df['Tenant Preferred'].value_counts()
tenant_df
tenant_df
plt.figure(figsize = (20, 8))
explode = (0, 0, 0.1)
colors = sns.color_palette('pastel')[0:5]
tenant_df.plot(kind = 'pie',
            colors = colors,
            explode = explode,
            autopct = '%1.1f%%')
plt.axis('equal')
plt.legend(labels = tenant_df.index, loc = "best")
plt.show()

# Different Types of Furnishing Status
furnishing_df = df['Furnishing Status'].value_counts()
furnishing_df

plt.figure(figsize = (20, 8))
explode = (0, 0, 0.1)
colors = sns.color_palette('pastel')[0:5]
furnishing_df.plot(kind = 'pie',
            colors = colors,
            explode = explode,
            autopct = '%1.1f%%')
plt.axis('equal')
plt.legend(labels = furnishing_df.index, loc = "best")
plt.show()


# Plot of each feature to show normal distribution
sns.pairplot(df, hue ='Rent')
plt.show()






# ---------------------------------------Training the Model---------------------------------------------------

X = df[['BHK', 'Size', 'Bathroom']]
y = df['Rent']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"TrainingSet Shape: {X_train}")

#parameters
NUM_EXAMPLES = 1024
BATCH_SIZE = 64
EPOCHS = 150 # actually steps
LR = 0.01
SEED = 3141
Ws, bs, xs, ys, ls = [], [], [], [], []

ds = (tf.data.Dataset
      .from_tensor_slices((X_train, y_train))
      .repeat()
      .shuffle(1000, seed=SEED)
      .batch(BATCH_SIZE))
ds = iter(ds)

model = keras.Sequential([
    layers.Input(shape=(3796,3)),
    layers.Dense(16, activation='relu'),
    layers.Dense(8, activation='relu'),
    layers.Dense(1)
])
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=LR), 
    loss='mse')


fig = plt.figure(dpi=150, figsize=(8, 3))
# Regression Curve
ax1 = fig.add_subplot(121)
ax1.set_title("Fitted Curve")
ax1.set_xlabel("x")
ax1.set_ylabel("y")
p10 = ax1.plot(X_train, y_train, 'r.', alpha=0.1) # full dataset
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


# Initialize the SGDRegressor model
# sgd_reg = SGDRegressor(loss='squared_loss', penalty='l2', alpha=0.01, max_iter=1, random_state=42)

# Fit model

# Obtain weights and intersection values for plots


# Plot regression and optimization process

# There are code snippets i tried but didnt work as expected so you can try also

# Initialize the scatter plot (empty for now)
# sc = ax.scatter(X_scaled[:, 1], y, c='b', label='Actual Rent')
# line, = ax.plot([], [], 'r', lw=2, label='Line of Best Fit')
# ax.set_xlabel('Size (Standardized)')
# ax.set_ylabel('Rent')
# ax.set_title('Linear Regression Animation')
# ax.legend()

# # Function to initialize the plot
# def init():
#     line.set_data([], [])
#     return line,

# # Function to update the plot for each epoch
# def update(epoch):
#     sgd_reg.partial_fit(X_scaled, y)
#     y_pred = sgd_reg.predict(X_scaled)
#     mse = mean_squared_error(y, y_pred)
#     line.set_data(X_scaled[:, 1], y_pred)
#     ax.set_title(f'Epoch {epoch + 1}, MSE: {mse:.2f}')
#     return line,

# # Create the animation
# animation = FuncAnimation(fig, update, init_func=init, frames=1000, repeat=False, blit=True)

# # Display the animation
# plt.show()
# print("DOne")


# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# scaler = StandardScaler()
# X_scaled = scaler.fit_transform(X_train)
# X_test_scaled = scaler.transform(X_test)
# # Initialize the SGDRegressor model
# sgd_reg = SGDRegressor(loss='squared_loss', penalty='l2', alpha=0.01, max_iter=1, random_state=42)
# fig, ax = plt.subplots()
# plt.close()  # Close the initial empty plot window


# # Initialize the scatter plot (empty for now)
# sc = ax.scatter(X_scaled[:, 1], y, c='b', label='Actual Rent')
# line, = ax.plot([], [], 'r', lw=2, label='Line of Best Fit')
# ax.set_xlabel('Size (Standardized)')
# ax.set_ylabel('Rent')
# ax.set_title('Linear Regression Animation')
# ax.legend()

# # Function to initialize the plot
# def init():
#     line.set_data([], [])
#     return line,

# # Function to update the plot for each epoch
# def update(epoch):
#     sgd_reg.partial_fit(X_scaled, y)
#     y_pred = sgd_reg.predict(X_scaled)
#     mse = mean_squared_error(y, y_pred)
#     line.set_data(X_scaled[:, 1], y_pred)
#     ax.set_title(f'Epoch {epoch + 1}, MSE: {mse:.2f}')
#     return line,

# # Create the animation
# animation = FuncAnimation(fig, update, init_func=init, frames=1000, repeat=False, blit=True)

# # Display the animation
# plt.show()

# # Build and train the SGDRegressor model
# sgd_reg = SGDRegressor(loss='squared_loss', penalty='l2', alpha=0.01, max_iter=1000, random_state=42)
# mse_history = []  # To store MSE values during training

# # Training loop
# for epoch in range(10):  # You can adjust the number of epochs as needed
#     sgd_reg.partial_fit(X_train_scaled, y_train)
#     y_pred = sgd_reg.predict(X_train_scaled)
#     mse = mean_squared_error(y_train, y_pred)
#     mse_history.append(mse)

# # Visualize the training process (MSE vs. epoch)
# epochs = range(1, len(mse_history) + 1)
# plt.plot(epochs, mse_history, marker='o')
# plt.xlabel('Epoch')
# plt.ylabel('Mean Squared Error (MSE)')
# plt.title('Training Loss (MSE) vs. Epoch')
# plt.grid(True)
# plt.show()

# # Make predictions on the test data
# y_pred = sgd_reg.predict(X_test_scaled)

# # Evaluate the model
# mse = mean_squared_error(y_test, y_pred)
# print(f'Mean Squared Error on Test Data: {mse}')

# # sns.scatterplot(data=X_train)
# # LR = 0.01
# # model = Model()
# # # train(model,X_train,y_train,LR)
# # # y_pred = model.predict(X_test)
# # # model = deepL_linear.Model()
# # # deepL_linear.train(model, X_train, y_train, learning_rate=LR)
# # fig = plt.figure(dpi=100, figsize=(8, 3))

# #   # Regression Line
# # ax1 = fig.add_subplot(131)
# # ax1.set_title("Fitted Line")
# # ax1.set_xlabel("x")
# # ax1.set_ylabel("y")
# #   # ax1.set_xlim(-3, 2.5)
# #   # ax1.set_ylim(-8, 11)
# # p10, = ax1.plot(X_train, y_train, 'r.', alpha=0.1) # full dataset
# # p11, = ax1.plot([], [], 'C3.') # batch, color Red
# # p12, = ax1.plot([], [], 'k') # fitted line, color Black

