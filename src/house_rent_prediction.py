import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,r2_score
from sklearn.model_selection import train_test_split,cross_val_score,cross_val_predict
from sklearn.preprocessing import scale
from sklearn.linear_model import Ridge,Lasso
from sklearn import model_selection
from sklearn.linear_model import RidgeCV,LassoCV
from deepL_linear import Model


df=pd.read_csv('..\House_Rent_Dataset.csv')
print(df.head())

# Pre-Processing
df.isnull().sum()

df.duplicated().sum()

print(df.describe())

print(df.dtypes)

df.shape

# Perform one-hot encoding using pandas' get_dummies
df_encoded = pd.get_dummies(df, columns=["Area Locality", "Furnishing Status"], prefix=["Locality", "Furnishing"])

print(df_encoded.head)


# Plot each attribute against Rent
# attributes = df.columns.difference(["Rent"])
# print(attributes.dtype)
# numeric_columns = attributes.select_dtypes(exclude=['object'])

numeric_columns = df.select_dtypes(exclude=['object'])
numeric_columns = numeric_columns.columns.difference(["Rent"]) 


# Create subplots for each numeric column
# fig, axes = plt.subplots(len(numeric_columns), 1, figsize=(7, 6 * len(numeric_columns)))

# for i, attribute in enumerate(numeric_columns):
#     axes[i].scatter(df[attribute], df["Rent"])
#     axes[i].set_xlabel(attribute)
#     axes[i].set_ylabel("Rent")
#     axes[i].set_title(f"{attribute} vs. Rent")
#     axes[i].grid(True)

# plt.tight_layout(rect=[1, 1, 1, 0.97])  # Adjust the rect parameter to control title spacing
# plt.show()

for attribute in numeric_columns:
    plt.figure(figsize=(8, 6))
    plt.scatter(df[attribute], df["Rent"])
    plt.xlabel(attribute)
    plt.ylabel("Rent")
    plt.title(f"{attribute} vs. Rent")
    plt.grid(True)
    plt.show()

#Add title
plt.title("Rent in Different Cities According to Area Type")

sns.barplot(x=df["City"], y=df["Rent"], hue=df["BHK"], ci=None)
plt.show()


# Training the Model

hrp = df[['BHK', 'Rent', 'Size', 'Bathroom']]
X = hrp.drop('Rent',axis=1)
X.head()

y = hrp["Rent"]
y.head()


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
sns.scatterplot(data=X_train)
LR = 0.01
model = Model()
model.fit(X_train,y_train)
y_pred = model.predict(X_test)
# model = deepL_linear.Model()
# deepL_linear.train(model, X_train, y_train, learning_rate=LR)
