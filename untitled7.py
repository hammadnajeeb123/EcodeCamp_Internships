# -*- coding: utf-8 -*-
"""Untitled7.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1Bd0PeQ3KlWLwCwPnE2970NQfj6kNFbiX
"""

import pandas as pd
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
lb=LabelEncoder()

# Replace 'your_dataset.csv' with the path to your dataset file
df = pd.read_csv('breast cancer pridiction data set.csv')

# Now you can work with the DataFrame 'df'
df.info()
df.isnull().sum()
df.describe()
df.dropna(axis=1)
df["diagnosis"].value_counts()
sns.countplot(df["diagnosis"],label="count")

lb.fit_transform(df.iloc[:,1].values)

df.head(10)

import pandas as pd
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt


# Initialize the LabelEncoder
lb = LabelEncoder()

# Replace 'breast cancer prediction data set.csv' with the path to your dataset file
df = pd.read_csv('breast cancer pridiction data set.csv')

# Convert the 'diagnosis' column to numerical values
df['diagnosis'] = lb.fit_transform(df['diagnosis'])

# Now you can work with the DataFrame 'df'
print(df.info())
print(df.isnull().sum())
print(df.describe())
print(df.dropna(axis=1))
print(df["diagnosis"].value_counts())
sns.countplot(df["diagnosis"], label="count")

# Display the first 10 rows of the DataFrame
print(df.head(10))

df["diagnosis"].value_counts()

df.corr()

import pandas as pd
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

# Initialize the LabelEncoder
lb = LabelEncoder()

# Replace 'breast cancer prediction data set.csv' with the path to your dataset file
df = pd.read_csv('breast cancer pridiction data set.csv')

# Convert the 'diagnosis' column to numerical values
df['diagnosis'] = lb.fit_transform(df['diagnosis'])

# Now you can work with the DataFrame 'df'
print(df.info())
print(df.isnull().sum())
print(df.describe())
print(df.dropna(axis=1))
print(df["diagnosis"].value_counts())
sns.countplot(df["diagnosis"], label="count")

# Display the first 10 rows of the DataFrame
print(df.head(10))

# Display the value counts of the 'diagnosis' column again
print(df["diagnosis"].value_counts())

# Create a heatmap of the correlation matrix for all features except 'diagnosis'
plt.figure(figsize=(30,30))
sns.heatmap(df.drop(columns=['diagnosis']).corr(), annot=True, fmt=".2f")
plt.show()

# Display the correlation matrix for all features except 'diagnosis'
print(df.drop(columns=['diagnosis']).corr())
sns.heatmap(df.drop(columns=['diagnosis']).corr(), annot=True)
plt.show()
# Create a pairplot for the first 10 features, colored by 'diagnosis'
sns.pairplot(df.iloc[:,1:5], hue="diagnosis")

x=df.iloc[:,2:32].values
y=df.iloc[:,1].values

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
x_train=sc.fit_transform(x_train)
x_test=sc.fit_transform(x_test)

x_train
from sklearn.linear_model import LogisticRegression
log=LogisticRegression(random_state=0)
log.fit(x_train,y_train)
log.score(x_train,y_train)
log.score(x_test,y_test)

from sklearn.metrics import confusion_matrix,classification_report
y_pred=log.predict(x_test)
cm=confusion_matrix(y_test,y_pred)
sns.heatmap(cm,annot=True)

print(classification_report(y_test,y_pred))