#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  4 16:55:31 2017

@author: annick-eudes
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: annick-eudes
"""
# --------------------------------------------------------------------------------------------
#                 PART 1. DATA ANALYSIS AND VISUALISATION
# --------------------------------------------------------------------------------------------

# ------------------------------- PRELIMINARIES  --------------------------------

# First thing set the working directory - it is done by setting it in the folder
# icon to the right;

# Next step is to import all the library we will need
#%%

import pandas as pd
import numpy as np
import seaborn as sns                 # for plots
import matplotlib.pyplot as plt # as plt is sort of a nickname for the library because
                                # it is too long.
import statsmodels.formula.api as smf # statsmodels
import statsmodels.stats.multicomp as multi # statsmodels and posthoc test
import scipy.stats              # For the Chi-Square test of independance


#%%
# Importing the data set:

df = pd.read_csv("ool_pds.csv", low_memory = False)

# Because Python is treating the variables has string instead of numeric variables
# we will convert them as numeric with the following function

""" setting variables you will be working with to numeric
10/29/15 note that the code is different from what you see in the videos
 A new version of pandas was released that is phasing out the convert_objects(convert_numeric=True)
It still works for now, but it is recommended that the pandas.to_numeric function be
used instead """

""" These where the old codes :
df["W1_G2"] = df["W1_G2"].convert_objects(convert_numeric = True)
df["W1_P20"] = df["W1_P20"].convert_objects(convert_numeric = True)
df["W1_F1"] = df["W1_F1"].convert_objects(convert_numeric = True)"""

# New codes
df["W1_G2"] = pd.to_numeric(df["W1_G2"], errors = "coerce")
df["W1_P20"] = pd.to_numeric(df["W1_P20"], errors = "coerce")
df["W1_F1"] = pd.to_numeric(df["W1_F1"], errors = "coerce")

# The research question is :
# To what extent is the perception of the US situation (W1_G2) associated with the level of income (W1_P20)?

# The variables of interest in our research question
print("W1_P20 is the Personnal Annual income")
print("W1_G2 is the US economy's situation")
print("W1_F1 is the Percentage of how the respondants think about the future")

# Determining the number of rows and columns in the dataset
print("This is the number of observations in the dataset:")
print(len(df))            # Number of observations = rows

print("This is the number of variables in the dataset:")
print(len(df.columns))    # Number of variables = columns

#%%

# -------------------------- Section # 1 / Basis descriptive data analysis ----------------------------


# ---------------------------- Examination of frequency tables  ------------------------------------

# Explatory data analysis ~ starting with one variable
# Univariate analysis
# The 'dropna = False' argument will display the missing values
# Making simple frequency tables [counts and frequencies].

# Counts :

print("Counts of the us economic situation 1 = Better, 2 = About the same, 3 = Worse, -1 = Refused to answer:")
c1 = df["W1_G2"].value_counts(sort=False, dropna = False)
print (c1)

print("Count of personal annual income :")
c2 = df["W1_P20"].value_counts(sort=False, dropna = False)
print (c2)

print("Counts of when you think about your future, are you generally optimistic, pessimistic, or neither optimistic nor pessimistic?")
c3 = df["W1_F1"].value_counts(sort=False, dropna = False)
print(c3)

# frequencies

print("Percentage of the us economic situation : 1 = Better, 2 = About the same, 3 = Worse, -1 = Refused to answer")
p1 = df["W1_G2"].value_counts(sort=False, normalize = True)
print (p1)

print("Percentage of personal annual income :")
p2 = df["W1_P20"].value_counts(sort=False, normalize = True)
print (p2)

print("Percentage of how the respondants think about the future")
p3 = df["W1_F1"].value_counts(sort=False, normalize = True)
print(p3)


# There are otherways to do this to have the same results, by using the .groupby function
print("Counts of the us economic situation 1 = Better, 2 = About the same, 3 = Worse, -1 = Refused to answer:")
ct1 = df.groupby("W1_G2").size()
print(ct1)

print("Count of personal annual income :")
ct2 = df.groupby("W1_P20").size()
print(ct2)

print("Counts of when you think about your future, are you generally optimistic, pessimistic, or neither optimistic nor pessimistic?")
ct3 = df.groupby("W1_F1").size()
print(ct3)

# To have the frequency, the code is simmilar, we just need to had the *100/len(Data)
print("Percentage of the us economic situation : 1 = Better, 2 = About the same, 3 = Worse, -1 = Refused to answer")
ct4 = df.groupby("W1_G2").size()*100/len(df)
print(ct4)

print("Percentage of personal annual income :")
ct5 = df.groupby("W1_P20").size()*100/len(df)
print(ct5)

print("Percentage of when you think about your future, are you generally optimistic, pessimistic, or neither optimistic nor pessimistic?")
ct6 = df.groupby("W1_F1").size()*100/len(df)
print(ct6)

#%%
# -------------- Section # 2 / regarding our research ~ Making data management decisions --------------

# Data Management

# Data management ~ Making decions about the data
# 1rst decide to code or not the missing values
# 2ed Creating or not new variables

# We will not be working with a subset of the data, because our research question, and
# our hypothesis, needs to have a view of the optimisum in regards to the income level.
# 1rst we will code the missing values
# we are going to set responses of (-1 ~ Refused)  for these variables to missing,
# so that Python disregards these values. We will code the missing values (nan).


# ------------------ Coding or recoding missing values ----------------------------------------

print("Let's start the Data Management ~ decision about the data, missing values and creating secondary variables")
df["W1_P20"]=df["W1_P20"].replace(-1, np.nan)
df["W1_G2"]=df["W1_G2"].replace(-1, np.nan)
df["W1_F1"]=df["W1_F1"].replace(-1, np.nan)

# Let's have a look at the variables with the new managed variables compared to the original variables
# The 'dropna = False' argument will display the missing values

print("Count of personal annual income (with the recoding of the missing values): ")
c2 = df["W1_P20"].value_counts(sort=False, dropna = False)
print(c2)

print("Counts of When you think about your future, are you generally 1 = optimistic, 2 = neither, or 3 = pessimistic? (with the recoding of the missing values)")
opt = df["W1_F1"].value_counts(sort=False, dropna = False)
print(opt)

print("Percentage of the us economic situation : 1 = Better, 2 = About the same, 3 = Worse, -1 = Refused to answer (with the recoding of the missing values)")
p1 = df["W1_G2"].value_counts(sort=False, dropna = True)
print(p1)


#%%
# We chose to group values within individual variables for the W1_P20 variable representing
# income level.
# categorize quantitative variable based on customized splits are done by using cut function
# we split the variable into 4 groups (1-7, 8-11, 12-15, 16-19)
# remember that Python starts counting from 0, not 1

# --------------------------- Grouping values within individual variables --------------------------

print("The income level is divided into 4 groups : 1-7 (5k- 24k), 8-11(25k-49k) 12-15(50k-99k), 16-19 (100k-175k or more))")
df["W1_P20"] = pd.cut(df.W1_P20, [0, 7, 11, 15, 19])
c10 = df["W1_P20"].value_counts(sort = False, dropna = True)
print(c10)

#%%

# --------------------------- Counts of the variables ------------------------------
# For verification purposes

print("Counts of the us economic situation 1 = Better, 2 = About the same, 3 = Worse, -1 = Refused to answer:")
c1 = df["W1_G2"].value_counts(sort=False)
print (c1)

print("Count of personal annual income :")
c2 = df["W1_P20"].value_counts(sort=False)
print (c2)

print("Counts of When you think about your future, are you generally 1 = optimistic, 2 = neither, or 3 = pessimistic?")
opt = df["W1_F1"].value_counts(sort=False)
print(opt)

# These variables have been managed

#%%

#  ---------------------------- Section # 3 / Visualising data ~ Graphs -------------------------------

# Visualizing categorical variables
# in order for categorical variables to be ordered properly on the horizontal, or X axis, of
# the univariate graph, we should convert your categorical variables, which are often formatted
# as numeric variables, into a format that Python recognizes as categorical.

# In our research question, we have W1_F1 (view on economic situation) has a categorical variable
# and W1_F1, view of the respondants about the future, and W1_P20 witch is a ordinal variable (Still)
# a type of categorical variable
#df["W1_G2"] = df["W1_G2"].astype('category')
#df["W1_F1"] = df["W1_F1"].astype('category')
#df["W1_P20"] = df["W1_P20"].astype('category')

# We will convert the variables to categoric to provide the descriptive statistics
#df["W1_G2"] = df["W1_G2"].astype('category')
#df["W1_F1"] = df["W1_F1"].astype('category')
#df["W1_P20"] = df["W1_P20"].astype("category")


# Standard deviation and other descriptive statistics for quantitative variables
#print("Describe the views of the economy's outcome")
#desc1 = df["W1_G2"].describe()
#print(desc1)

#print("Describe the views on the future by respondants")
#desc2 = df["W1_F1"].describe()
#print(desc2)

#print("Describe the personnal annual income for the respondants")
#desc3 = df["W1_P20"].describe()
#print(desc3)

#%%

# UNIVARIATE GRAPH

# Let's plot our categorical variables :
sns.countplot(x = "W1_G2", data = df)
plt.xlabel("-1 = refused, 1 = better, 2 = about the same, or 3 = worse")
plt.title("Respondants views on the nation's economy compared to one year ago")

#%%

# UNIVARIATE GRAPH
# Graphique pour une variable : "Respondants views regardgin their future :

sns.countplot(x = "W1_F1", data = df)
plt.xlabel("-1 = refused, 1 = optimistic, 2 = neither optimistic nor pessimistic, 3 = pessimistic")
plt.title("Respondants views regarding their future")

#%%
# UNIVARIATE GRAPH
# Graohique pour une variable : Income groups reported by respondents :

sns.countplot(x = "W1_P20", data = df)
plt.xlabel("Interval of annual income :1-7 (5k- 24k), 8-11(25k-49k) 12-15(50k-99k), 16-19 (100k-175k or more)")
plt.title("Income groups reported by respondents")

#%%
# Now let's display the graphics for the managed variables
# Graphing a quatitative variable
# The W1_P20 is not a ordinal variable, this is of example only
# seaborn.distplot(Data["W1_P20"].dropna(), kde = False)
# plt.xlabel("Group of personal annual income")
# plt.title("Income groups reported by respondents")

# Standard deviation and other descriptive statistics for quantitative variables
print("Describe the views of the economy's outcome")
desc1 = df["W1_G2"].describe()
print(desc1)

print("Describe the views on the future by respondants")
desc2 = df["W1_F1"].describe()
print(desc2)

print("Describe the personnal annual income for the respondants")
desc3 = df["W1_P20"].describe()
print(desc3)

#%%

# ------------------ Make a decision about the role that each variable will play -----
#
# The explanatory variable is the  income level (W1_P20) the perception of the and the response
# variable nation’s economic situation (W1_G2 and/or W1_F1). Thus, using the graphing decisions
# flow chart we will use a Categorical to Categorical bar chart to plot the associations between
# our explanatory and response variables.
# We have to convert the categorical variables to numeric to do a C -> C bar chart.

# Setting variables you will be working with to numeric

# Ancienne facon de faire :
#df["W1_P20"] = df["W1_P20"].convert_objects(convert_numeric=True)
#df["W1_G2"] = df["W1_G2"].convert_objects(convert_numeric=True)
# Pandas a été mis à jour, maintenant, il faut utiliser :

df["W1_F1"] = pd.to_numeric(df["W1_F1"], errors="coerce")
df["W1_P20"] = pd.to_numeric(df["W1_P20"], errors="coerce")
df["W1_G2"] = pd.to_numeric(df["W1_G2"], errors="coerce")




#%%
# BI VARIATE GRAPH :

# Personnal income versus the US economy's situation

# When doing the visualisations, the variables have to be in numeric (* sinon cela ne va pas marcher)
# Pour éviter ce problème, il est mieux de reconvertir les variables qui étaient en catégorie
# En des variables numériques.

print("This is the Categorical -> Categorical graph of US economy's situtation vs Personnal annual income")
sns.factorplot(x = "W1_P20", y = "W1_G2", data = df, kind = "bar", ci = None)
plt.xlabel("Personnal annual income")
plt.ylabel("The US economy's situation")

#%%
# BI VARIATE GRAPH :

# Personnal income versus how respondants think about the future

sns.factorplot(x = "W1_P20", y = "W1_F1", data = df, kind = "bar", ci = None)
plt.xlabel("Personnal annual income")
plt.ylabel("How the respondants think about the future")

#%%
# --------------------------------------------------------------------------------------------
#                               PART 2. DATA ANALYSIS TOOLS
# --------------------------------------------------------------------------------------------

# Now that we have a research question, selected the data set and managed our variables
# of interest and visualized their relationship graphically, we are ready to
# test those relationships statistically.

# A partir d'ici, toutes les variables doivent etre numériques.


# ---------------------------- To calulate the ANOVA F-Statistics ----------------------------------

# If we have a bi-variate statistical analysis tools for two variables i.e. y = ax + b + et
# Analysis of variace Quantiative response variable (y) and Explanatory Categorical variable (x)
# Using ols function for the computing of the F-statistic and associated p value.

# As a reminder :

print("W1_D1 is the variable is How would you rate the president at the time (Barack Obama)")
print("W1_P20 is the Personnal Annual income")
print("The income level is divided into 4 groups : 1-7 (5k- 24k), 8-11(25k-49k), 12-15(50k-99k), 16-19 (100k-175k or more))")

model1 = smf.ols(formula='W1_D1 ~ C(W1_P20)', data=df)
results1 = model1.fit()
print (results1.summary())


#%%

# To interpret this finding fully, we need to examine the actual means for the variables
# We will create a new data frame with the quatitative response variable and the categorical
# explanatory variable
sub3 = df[['W1_D1', 'W1_P20']].dropna()


# Means and standard deviations in the new dataframe :
print ("Means for income level by the president's rating")
m1= sub3.groupby('W1_P20').mean()
print (m1)


print ("standard deviations for income level by the president's rating")
sd1 = sub3.groupby('W1_P20').std()
print (sd1)

#%%
# POST HOC TEST :

# In order to conduct post hoc paired comparisons in the context of my ANOVA, examining the association
# between ethnicity and number of cigarettes smoked per month, I'm going to use
# the Tukey HSDT, or Honestly Significant Difference Test.


# mc1 is the object that will store the mutiple comparisons test
# then, I include the quantitative response variable and the categorical explanatory variable
mc1 = multi.MultiComparison(sub3['W1_D1'], sub3['W1_P20'])
res1 = mc1.tukeyhsd()       # result (mc1) x the tukey post hoc test
print(res1.summary())

#%%
# ---------------------------- Chi square test of independance ------------------------------------

# Is the perception of the us economic situation dependent or indedendent of 
# the income levels?

# In reference to our research question, the explanatory variable is the income 
# level (W1_P20) the perception of the and the response variable nation’s 
# economic situation (measured by W1_G2 and/or W1_F1). 

# Explanatory variable, idependant (x) variable
# W1_P20: is the Personnal Annual income

# The response variables:
# W1_G2: is the US economy's situation
# W1_F1: is the Percentage of how the respondants think about the future

# 1) request the contengency table of observerved counts
print("Contengency table for the US economy's situation and Personnal Annual income")
print("The first results include the table of counts of the response variable by the explanatory variable.")
count1 = pd.crosstab(df["W1_G2"], df["W1_P20"])
print(count1)

# 2) Now we have to generate the column % wich will show the % of in
print("Column percentage")
colsum = count1.sum(axis=0)
colpercent = count1/colsum
print(colpercent)

# Chi-square :
print("Chi-Square value, p-value, expected counts:")
chi_sq1 = scipy.stats.chi2_contingency(count1)
print(chi_sq1)

# If I what to graph the percent of personnal annual income that have a positive
# view of the us economic outcoume

# To plot this we need to :
# First setting out explanatory variable to categorical and a response variable to numeric.  
df["W1_P20"] = df["W1_P20"].astype("category")
df["W1_G2"] = pd.to_numeric(df["W1_G2"], errors="coerce")

# Plot
# X the categorical variable and Y the numeric variable
sns.factorplot(x = "W1_P20", y = "W1_G2", data=df, kind = "bar", ci=None)
plt.xlabel("The Personnal Annual income")
plt.ylabel("The US economy's situation")
