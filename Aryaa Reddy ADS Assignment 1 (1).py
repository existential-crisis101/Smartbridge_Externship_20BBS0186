#!/usr/bin/env python
# coding: utf-8

# In[2]:


#1. Assign your Name to variable name and Age to variable age. Make a Python program that prints your name and age
name = "Aryaa Reddy"
age = 19

print("Name:", name)
print("Age:", age)


# In[3]:


#2. X="Datascience is used to extract meaningful insights." Split the string
X = "Datascience is used to extract meaningful insights."
split_words = X.split()

print(split_words)


# In[6]:


#3. Make a function that gives multiplication of two numbers
def multiply_numbers(a, b):
    return a * b

result = multiply_numbers(129, -23)
print(result)


# In[7]:


#4. Create a Dictionary of 5 States with their capitals. also print the keys and values.
indian_states = {
    "Karnataka": "Bengaluru",
    "Maharashtra": "Mumbai",
    "Tamil Nadu": "Chennai",
    "Uttar Pradesh": "Lucknow",
    "Gujarat": "Gandhinagar"
}

print("Keys:")
for state in indian_states:
    print(state)

print("\nValues:")
for capital in indian_states.values():
    print(capital)

print("\nKeys and Values:")
for state, capital in indian_states.items():
    print(state, "<-->", capital)


# In[9]:


#5. Create a list of 1000 numbers using range function.
numbers = list(range(1, 1001))
print(numbers)


# In[10]:


#6. Create an identity matrix of dimension 4 by 4
n = 4  

identity_matrix = [[1 if i == j else 0 for j in range(n)] for i in range(n)]

for row in identity_matrix:
    print(row)


# In[13]:


#7. Create a 3x3 matrix with values ranging from 1 to 9
matrix = [[j + 1 + (i * 3) for j in range(3)] for i in range(3)]

for row in matrix:
    print(row)


# In[15]:


#8. Create 2 similar dimensional array and perform sum on them.
import numpy as np

array1 = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
array2 = np.array([[9, 8, 7], [6, 5, 4], [3, 2, 1]])

result = array1 + array2

print(result)


# In[16]:


#9. Generate the series of dates from 1st Feb, 2023 to 1st March, 2023 (both inclusive)
from datetime import datetime, timedelta

start_date = datetime(2023, 2, 1)
end_date = datetime(2023, 3, 1)

current_date = start_date
while current_date <= end_date:
    print(current_date.strftime('%Y-%m-%d'))
    current_date += timedelta(days=1)


# In[18]:


#10. Given a dictionary, convert it into corresponding dataframe and display it dictionary = {'Brand': ['Maruti', 'Renault', 'Hyndai'], 'Sales' : [250, 200, 240]}
import pandas as pd

dictionary = {'Brand': ['Maruti', 'Renault', 'Hyundai'], 'Sales': [250, 200, 240]}
df = pd.DataFrame(dictionary)

print(df)

