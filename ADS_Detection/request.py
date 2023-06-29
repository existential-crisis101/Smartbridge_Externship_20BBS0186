# -*- coding: utf-8 -*-
"""
Created on Mon Jun 26 18:07:28 2023

@author: Sheet gupta
"""

import requests

url = 'http://localhost:5000/results'
r = requests.post(url,json={'A1_Score':1, 'A2_Score':1, 'A3_Score':1, 'A4_Score':1, 
'A5_Score':0,'A6_Score':0, 'A7_Score':1, 'A8_Score':1, 'A9_Score':0, 'A10_Score':0,
'age':26, 'gender':'f', 'ethnicity': 'White-European','jaundice':'no','autism':'no',
'country_of_res':'United States','used_app_before':'no','relation':'Self'})

print(r.json())