# -*- coding: utf-8 -*-
"""
Created on Fri Jul 19 16:42:20 2013

@author: gregory
"""

import pandas as pd
train = pd.read_csv('./input00.txt', sep="\s", skiprows=1,nrows=4500, names=["id","rank","col1","col2","col3","col4","col5","col6","col7","col8","col9","col10","col11","col12","col13","col14","col15","col16","col17","col18","col19","col20","col21","co22","col23"])
test = pd.read_csv('./input00.txt', sep="\s", skiprows=4502, names=["id","col1","col2","col3","col4","col5","col6","col7","col8","col9","col10","col11","col12","col13","col14","col15","col16","col17","col18","col19","col20","col21","co22","col23"])
