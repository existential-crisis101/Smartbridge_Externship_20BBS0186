#!/usr/bin/env python
# coding: utf-8

# # Aryaa Reddy - Assignment 2 - Applied Data Science 

# In[11]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[12]:


#Loading the Dataset 
df=pd.read_csv('titanic.csv')
df.head()


# In[13]:


df.tail()


# In[15]:


#Univariate Analysis 
# Frequency distribution of 'Sex'
sex_counts = df['sex'].value_counts()
print(sex_counts)

# Bar plot for 'Sex' distribution
sns.countplot(x='sex', data=df)
plt.title('Distribution of Passengers by Sex')
plt.show()


# In[17]:


#Bivariate analysis
# Box plot of 'Age' by 'Survived'
sns.boxplot(x='survived', y='age', data=df)
plt.title('Age Distribution by Survival')
plt.show()


# In[23]:


# Multivariate Analysis

# Correlation matrix
correlation_matrix = df.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()

# Pair plot of selected variables
selected_vars = ['survived', 'age', 'fare', 'pclass']
sns.pairplot(data=df[selected_vars].dropna(), hue='survived')
plt.title('Pair Plot of Selected Variables')
plt.show()


# In[22]:


# descriptive statistics on the dataset 

descriptive_stat = df.describe()
print(descriptive_stat)


# In[25]:


# Check for missing values
missing_values = df.isnull().sum()
print(missing_values)

# Handling missing values

# Drop rows with missing values
df_dropna = df.dropna()
print("After dropping rows with missing values:")
print(df_dropna.isnull().sum())

# Fill missing values with mean age
mean_age = df['age'].mean()
df_fillna_mean = df.fillna({'age': mean_age})
print("After filling missing values with mean age:")
print(df_fillna_mean.isnull().sum())

# Fill missing values with mode embarked
mode_embarked = df['embarked'].mode()[0]
df_fillna_mode = df.fillna({'embarked': mode_embarked})
print("After filling missing values with mode embarked:")
print(df_fillna_mode.isnull().sum())


# In[35]:


# Find the outliers and replace the outliers

# Outlier Detection
sns.boxplot(df.fare)


# In[36]:


perc99=df.fare.quantile(0.99)
perc99


# In[37]:


df=df[df.<=perc99]
sns.boxplot(df.fare)


# In[38]:


# Check for categorical columns
categorical_cols = df.select_dtypes(include='object').columns
print("Categorical Columns:")
print(categorical_cols)

# Perform one-hot encoding
df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
print("Encoded DataFrame:")
print(df_encoded.head())


# In[72]:


#Split the data into dependent and independent variables. 

X = df.drop("survived", axis=1)  # Independent variables (features)
y = df["survived"]               # Dependent variable (target)

# Print the shape of the datasets
print("Shape of X:",X.shape)
print("Shape of y:",y.shape)


# In[83]:


# Scale the independent variables 

from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

# Split the dataset into dependent and independent variables
X = df.drop("survived", axis=1)  # Independent variables (features)

# Preprocess the features

# Identify numerical and categorical columns
numeric_cols = X.select_dtypes(include='number').columns
categorical_cols = X.select_dtypes(include='object').columns

# Create a column transformer for preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ('numeric', MinMaxScaler(), numeric_cols),
        ('categorical', OneHotEncoder(), categorical_cols)
    ])

# Fit and transform the data
X_scaled = preprocessor.fit_transform(X)

# Convert the scaled array back to a DataFrame
X_scaled_df = pd.DataFrame(X_scaled)

# Print the first few rows of the scaled DataFrame
print(X_scaled_df.head())


# In[84]:


# Split data into training and testing 

from sklearn.model_selection import train_test_split


# In[96]:


# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Print the shapes of the resulting datasets
print("Training set shape:", X_train.shape, y_train.shape)
print("Testing set shape:", X_test.shape, y_test.shape)


# In[97]:


y_test


# In[98]:


y_train


# In[100]:


X_train.head()


# In[101]:


X_test.head()


# In[ ]:




