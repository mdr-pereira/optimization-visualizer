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




df=pd.read_csv('../House_Rent_Dataset.csv')



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
# to show
plt.show()

# Scatter Plot on House Rents vs House Sizes
fig = px.scatter(df, x='Size', y='Rent', color='BHK', size='Size', hover_data=['Rent'])
fig.update_layout(title='House Rents vs House Sizes',
                  yaxis_zeroline=False, xaxis_zeroline=False)
fig.show()




# # Training the Model

# hrp = df[['BHK', 'Rent', 'Size', 'Bathroom']]
# X = hrp.drop('Rent',axis=1)
# X.head()

# y = hrp["Rent"]
# y.head()


# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# sns.scatterplot(data=X_train)
# LR = 0.01
# model = Model()
# # train(model,X_train,y_train,LR)
# # y_pred = model.predict(X_test)
# # model = deepL_linear.Model()
# # deepL_linear.train(model, X_train, y_train, learning_rate=LR)
# fig = plt.figure(dpi=100, figsize=(8, 3))

#   # Regression Line
# ax1 = fig.add_subplot(131)
# ax1.set_title("Fitted Line")
# ax1.set_xlabel("x")
# ax1.set_ylabel("y")
#   # ax1.set_xlim(-3, 2.5)
#   # ax1.set_ylim(-8, 11)
# p10, = ax1.plot(X_train, y_train, 'r.', alpha=0.1) # full dataset
# p11, = ax1.plot([], [], 'C3.') # batch, color Red
# p12, = ax1.plot([], [], 'k') # fitted line, color Black

