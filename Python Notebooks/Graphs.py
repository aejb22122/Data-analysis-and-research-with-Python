#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  1 19:01:20 2018

@author: annick-eudes
"""

# ------------------------------- PRELIMINARIES  --------------------------------

# First thing set the working directory - it is done by setting it in the folder
# icon to the right;

# Next step is to import all the library we will need

# Libraries

import pandas as pd
import numpy as np
import seaborn as sns  # for plots
import matplotlib.pyplot as plt  # as plt is sort of a nickname for the library because
# it is too long.
import statsmodels.formula.api as smf  # statsmodels
import statsmodels.stats.multicomp as multi  # statsmodels and posthoc test
import statsmodels.api as sm  # Statsmodel for the qqplots
import scipy.stats  # For the Chi-Square test of independance

# Machine learning
# Libraries for decision trees

from pandas import Series, DataFrame
import os 

from sklearn.cross_validation import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
import sklearn.metrics

# import graphviz

df = pd.read_csv("iris-data-clean.csv", low_memory=False)

len(df)

