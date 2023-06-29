#!/usr/bin/env python
# coding: utf-8

# In[47]:


#Load the dataset into the tool.
import pandas as pd

data = pd.read_csv('housing.csv')


# In[48]:


# Univariate analysis
import matplotlib.pyplot as plt

# Histogram 
plt.hist(data['price'], bins=20)
plt.xlabel('Price')
plt.ylabel('Area')
plt.title('Price VS Area')
plt.show()

# Bar plot 
data['bedrooms'].value_counts().plot(kind='bar')
plt.xlabel('Bedrooms')
plt.ylabel('Stories')
plt.title('Bedrooms VS Stories')
plt.show()


# In[11]:


# Bivariate analysis
import seaborn as sns

# Scatter plot 
sns.scatterplot(x='area', y='price', data=data)
plt.xlabel('Area')
plt.ylabel('Price')
plt.title('Price vs. Area')
plt.show()

# Box plot 
sns.boxplot(x='parking', y='price', data=data)
plt.xlabel('Parking')
plt.ylabel('Price')
plt.title('Parking vs Price')
plt.show()


# In[12]:


# Multivariate analysis
# Correlation matrix
corr_matrix = data.corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()

# Pair plot
sns.pairplot(data)
plt.show()


# In[13]:


# Descriptive statistics
statistics = data.describe()
print(statistics)


# In[15]:


# Check for missing values
missing_values = data.isnull().sum()
print(missing_values)
data = data.dropna()


# In[17]:


#detecting outliers
import numpy as np
def detect_outliers_zscore(data, threshold=3):
    z_scores = np.abs((data - data.mean()) / data.std())
    outliers = data[z_scores > threshold]
    return outliers
numerical_columns = ['price', 'area', 'bedrooms', 'bathrooms','stories','parking']
for column in numerical_columns:
    outliers = detect_outliers_zscore(data[column])
    print(f"Outliers in {column}:")
    print(outliers)


# In[18]:


#replacing outliers
def replace_outliers(data, column, method='median', threshold=3):
    z_scores = np.abs((data[column] - data[column].mean()) / data[column].std())
    outliers = data[z_scores > threshold]
    
    if method == 'median':
        replacement_value = data[column].median()
    elif method == 'mean':
        replacement_value = data[column].mean()
    else:
        raise ValueError("Invalid replacement method. Choose 'median' or 'mean'.")
    
    data.loc[z_scores > threshold, column] = replacement_value
    
    return data

numerical_columns = ['price', 'area', 'bedrooms', 'bathrooms','stories','parking']
for column in numerical_columns:
    data = replace_outliers(data, column, method='median')
data.to_csv('housing_without_outliers.csv', index=False)


# In[35]:


# Check for categorical columns
categorical_columns = data.select_dtypes(include=['object']).columns

# Perform one-hot encoding
data_encoded = pd.get_dummies(data, columns=categorical_columns)


# In[36]:


# Split into dependent and independent variables
X = data.drop('price', axis=1)  # Independent variables
y = data['price']  # Dependent variable


# In[37]:


from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# In[46]:


# Split the data into training and testing
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)


# In[45]:


# Build the model
from sklearn.linear_model import LinearRegression

model = LinearRegression()


# In[44]:


# Train the model
model.fit(X_train, y_train)


# In[41]:


# Test the model
y_pred = model.predict(X_test)


# In[43]:


# Measure performance
from sklearn.metrics import mean_squared_error, r2_score

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("RMSE:", rmse)
print("R-squared:", r2)

