# -*- coding: utf-8 -*-
"""
Created on Fri Jul 19 16:42:20 2013

@author: Bit Shift Boys
"""

import pandas as pd
# import rows 1-4501 and add column titles
train = pd.read_csv('./input00.txt', sep="\s", skiprows=1,nrows=4500, names=range(25))
# import rows 4502-6002 and add column titles
test = pd.read_csv('./input00.txt', sep="\s", skiprows=4502, names=range(24))
# for columns slice out col id and :
# training data - first 9 features
for col in range(2,11):
   train[col] = train[col].map(lambda val: float(str(val)[2:]))
# training data - first 9 features
for col in range(11,25):
   train[col] = train[col].map(lambda val: float(str(val)[3:]))
# test data - first 9 features
for col in range(1,10):
   test[col] = test[col].map(lambda val: float(str(val)[2:]))
# test data - first 9 features
for col in range(10,24):
   test[col] = test[col].map(lambda val: float(str(val)[3:]))

# scale data
# turn -1 to 0 and leave 1 as 1 - THIS WILL TURN ALL NON -1 VALS TO 1!!! check this
train[1] = train[1].map(lambda val: 0 if val == -1 else 1)
test[1] = test[1].map(lambda val: 0 if val == -1 else 1)
# find max and divide each number by max

