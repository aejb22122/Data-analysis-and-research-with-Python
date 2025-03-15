
# ----------------------------------------------------------------------------------------
#                           Importing required packages and the dataset
# ----------------------------------------------------------------------------------------



# First thing set the working directory - it is done by setting it in the folder
# icon to the right in spyder IDE

# Next step is to import all the library we will need for data analysis :

import pandas
import numpy
import seaborn                  # for plots
import matplotlib.pyplot as plt # as plt is sort of a nickname for the library because
                                # it is too long.
import statsmodels.formula.api as smf # statsmodels
import statsmodels.stats.multicomp as multi # statsmodels and posthoc test
import scipy.stats              # For the Chi-Square test of independance





# We will import these libraries for data visualisation and graphs :
import seaborn
import matplotlib.pyplot as plt # as plt is sort of a nicknme for the matplotlib.pyplot because
                                # it is too long.


# Importing the data set
Data = pandas.read_csv("ool_pds.csv", low_memory = False)

print("W1_P20 is the Personnal Annual income (this variable is an interval variable)")
print("W1_G2 is the US economy's situation (this variable is an ordinal variable)")
print("W1_F1 is the Percentage of how the respondants think about the future", end ='')
print("(this variable is an ordinal variable)")

# If you whant you can see the head of the dataset :
Data.head(5)

# Determining the number of rows and columns in the dataset
print("This is the number of observations in the dataset:")
print(len(Data))            # Number of observations = rows

print("This is the number of variables in the dataset:")
print(len(Data.columns))    # Number of variables = columns

# Because Python is treating the variables has string instead of numeric variables
# we will convert them as numeric with the following to avoid order problems (like sorting and calculations)
Data["W1_G2"] = Data["W1_G2"].convert_objects(convert_numeric = True)
Data["W1_P20"] = Data["W1_P20"].convert_objects(convert_numeric = True)
Data["W1_F1"] = Data["W1_F1"].convert_objects(convert_numeric = True)


# ----------------------------------------------------------------------------------------
#                           Making frequency distributions
# ----------------------------------------------------------------------------------------

# Explatory data analysis ~ starting with one variable

# Univariate analysis :
# Python does not display missing values for numeric variables .
# If we'd like to include a count of our missing data in the frequency distribution
# we also need to add the option drop NA equals false argument

# Counts
print("Counts of the us economic situation : 1 = Better, 2 = About the same, ",end='')
print("3 = Worse, -1 = Refused to answer:")
c1 = Data["W1_G2"].value_counts(sort=False, dropna = False)
print (c1)


print("Count of personal annual income :")
c2 = Data["W1_P20"].value_counts(sort=False, dropna = False)
print (c2)
print("Counts of When you think about your future, are you generally 1 = optimistic, ",end='')
print(" 2 = neither, or 3 = pessimistic?")
opt = Data["W1_F1"].value_counts(sort=False, dropna = False)
print(opt)

# Percentages
# The 'dropna = False' argument will display the missing values
print("Percentage of the us economic situation : 1 = Better, 2 = About the same, ",end='')
print(" 3 = Worse, -1 = Refused to answer")
p1 = Data["W1_G2"].value_counts(sort=False, normalize = True)
print (p1)

print("Percentage of personal annual income :")
p2 = Data["W1_P20"].value_counts(sort=False, normalize = True)
print (p2)

print("Percentage of how the respondants think about the future")
opt2 = Data["W1_F1"].value_counts(sort=False, normalize = True)
print(opt2)


# There are otherways to do this to have the same results, by using the .groupby function
print("Counts of the us economic situation 1 = Better, 2 = About the same, ",end='')
print("3 = Worse, -1 = Refused to answer:")
ct1 = Data.groupby("W1_G2").size()
print(ct1)

print("Count of personal annual income :")
ct2 = Data.groupby("W1_P20").size()
print(ct2)

# To have the frequency, the code is simmilar, we just need to had the *100/len(Data)
print("Percentage of the us economic situation : 1 = Better, 2 = About the same,",end='')
print(" 3 = Worse, -1 = Refused to answer")
ct3 = Data.groupby("W1_G2").size()*100/len(Data)
print(ct3)

print("Percentage of personal annual income :")
ct4 = Data.groupby("W1_P20").size()*100/len(Data)
print(ct4)



# ----------------------------------------------------------------------------------------
#                             Working with a subset of the data
# ----------------------------------------------------------------------------------------


# We can redefine the variables by adding logic statements in the codes.
# We now focus on the economic outcome in the mid level income groups.
# i.e. $20,000 < income < 99,999 $ (For some reason the <= sends to much numbers)
# Let's establish the subset

#print("We now focus on the economic outcome in the mid level income groups.")
#sub1 = Data[(Data["W1_P20"]> 7) & (Data["W1_P20"])< 15 & (Data["W1_G2"] == 1)]
#sub2 = sub1.copy()          # the sub1.copy eliminate a copy error that might occure.

# Just to verify
#print('Counts for the original dataset')
#print("This is the number of observations in the original dataset")
#print(len(Data))            # Number of observations = rows

#print("This is the number of variables in the dataset:")
#print(len(Data.columns))    # Number of variables = columns


#print("Number of rows, number of observations in the sub group of the dataset")
#print(len(sub2))            # number of rows, number of observations
#print("Number of variables")
#print(len(Data.columns))    # Number of columns


# Let's look at the frequency tables for the subset for the data frame
# These are the counts
#print("Counts of the mid-level incomes")
#c3 = sub2["W1_P20"].value_counts(sort = False)
#print(c3)

#print("Counts of repondants that consider that the us economic situation that consider", end='')
#print(" that the country is doing better")
#c4 = sub2["W1_G2"].value_counts(sort= False)
#print(c4)   # The result showed for only the value of 1 for the category with is what was intended.

#print("Counts of respondants and their outlook on the future")
#print("1 = Optimistic, 2 = Neither, 3 = Pessimistic, -1 = Refused")
#c5 = sub2["W1_F1"].value_counts(sort = False)
#print(c5)   # This is for the subset of the income that we have defined

# Frequency distributions
#print("Percentage of mid-level incomes (i.e. $20,000 < income < 99,999 $)")
#c6 = sub2["W1_P20"].value_counts(sort = False, normalize = True)
#print(c6)

#print("Percentage of repondants that consider that the us economic situation", end='')
#print(" is doing better")
#c7 = sub2["W1_G2"].value_counts(sort = False, normalize = True)
#print(c7)

#print("Percentage of respondants and their outlook on the future")
#print("1 = Optimistic, 2 = Neither, 3 = Pessimistic, -1 = Refused")
#c8 = sub2["W1_F1"].value_counts(sort = False, normalize = True)
#print(c8)


# -----------------------------------------------------------------------------
#                                 Data Management
# -----------------------------------------------------------------------------


# This is for practice :
# Data management ~ Making decions about the data
# 1rst decide to code or not the missing values
# we are going to set responses of (-1 ~ Refused)  for these variables to missing,
# so that Python disregards these values. We will code the missing values (nan).

#print("Let's start the Data Management ~ decision about the data, missing values", end = '')
#print(" and creating secondary variables")
#sub2["W1_P20"]=sub2["W1_P20"].replace(-1, numpy.nan)
#sub2["W1_G2"]=sub2["W1_G2"].replace(-1, numpy.nan)
#sub2["W1_F1"]=sub2["W1_F1"].replace(-1, numpy.nan)

# To verify, we are going to print the distribution of the original variable and the other
# With the -1 equal to NAN
# Python does not display missing values for numeric.
# If we'd like to include a count of our missing data in the frequency distribution
# we also need to add the option drop NA equals false;
# We don't realy need to do it for the variable W1_G2 because it is the one == 1, there are no
# missing values for this variable.

#print("Counts of repondants that consider that the us economic situation that consider",end='')
#print(" that the country is doing better")
#c4 = sub2["W1_G2"].value_counts(sort= False)
#print(c4)

#print("Counts of repondants that consider that the us economic situation that consider",end='')
#print(" that the country is doing better")
#c9 = sub2["W1_G2"].value_counts(sort= False, dropna = False)
#print(c9)

#print("Counts of the mid-level incomes")
#c3 = sub2["W1_P20"].value_counts(sort = False)
#print(c3)

#print("Count of the mid-level incomes (with the recoding of the missing values)")
#c33 = sub2["W1_P20"].value_counts(sort = False, dropna = False)
#print(c33)

#print("Counts of respondants and their outlook on the future")
#print("1 = Optimistic, 2 = Neither, 3 = Pessimistic, -1 = Refused")
#c5 = sub2["W1_F1"].value_counts(sort = False)
#print(c5)

#print("Counts of respondants and their outlook on the future (with the recoding of", end = '')
#print("the missing values")
#c55 = sub2["W1_F1"].value_counts(sort = False, dropna = False)
#print(c55)
# These variables have been managed in regards to the missing value.

# There are other ways to manage data = Recoding valid data and creating secondary variables
# If we whant to see only the variables that we are working with, we need to create a new subset
# With the unique identifier and the variables we are working with :
#print("If we whant to see only the variables that we are working with, we need to create", end = '')
#print(" a new subset with the unique identifier and the variables we are working with")
#sub3=sub2[["W1_CASEID", "W1_P20", "W1_G2", "W1_F1"]]
#sub3.head(5)        # 5 first rows of the subset
#print(len(sub3))

# --------------------------------------------------------------------------------------------
#               For the assignment # 2 / our research ~ Making data management decisions
# --------------------------------------------------------------------------------------------

# We will not be working with a subset of the data, because our research question, and
# our hypothesis, needs to have a view of the optimisum in regards to the income level.
# 1rst we will code the missing values
# we are going to set responses of (-1 ~ Refused)  for these variables to missing,
# so that Python disregards these values. We will code the missing values (nan).

# ------------------ Coding or recoding missing values ----------------------------------------

print("Let's start the Data Management ~ decision about the data, missing values", end = '')
print(" and creating secondary variables")
Data["W1_P20"]=Data["W1_P20"].replace(-1, numpy.nan)
Data["W1_G2"]=Data["W1_G2"].replace(-1, numpy.nan)
Data["W1_F1"]=Data["W1_F1"].replace(-1, numpy.nan)

# Let's have a look at the variables with the new managed variables compared to the original variables
# The 'dropna = False' argument will display the missing values

print("Count of personal annual income (with the recoding of the missing values): ")
c2 = Data["W1_P20"].value_counts(sort=False, dropna = False)
print(c2)

print("Counts of When you think about your future, are you generally 1 = optimistic, ",end='')
print(" 2 = neither, or 3 = pessimistic? (with the recoding of the missing values)")
opt = Data["W1_F1"].value_counts(sort=False, dropna = False)
print(opt)

print("Percentage of the us economic situation : 1 = Better, 2 = About the same, ",end='')
print(" 3 = Worse, -1 = Refused to answer (with the recoding of the missing values)")
p1 = Data["W1_G2"].value_counts(sort=False, dropna = True)
print(p1)



# We chose to group values within individual variables for the W1_P20 variable representing
# income level.
# categorize quantitative variable based on customized splits are done by using cut function
# we split the variable into 4 groups (1-7, 8-11, 12-15, 16-19)
# remember that Python starts counting from 0, not 1

# ------------------ Grouping values within individual variables --------------------------

print("The income level is divided into 4 groups : 1-7 (5k- 24k), 8-11(25k-49k)", end = '')
print(" 12-15(50k-99k), 16-19 (100k-175k or more))")
Data["W1_P20"] = pandas.cut(Data.W1_P20, [0, 7, 11, 15, 19])
c10 = Data["W1_P20"].value_counts(sort = False, dropna = True)
print(c10)

# --------------------------------------------------------------------------------------------
#                       Visualising data ~ Graphs
# --------------------------------------------------------------------------------------------
# Visualizing categorical variables
# in order for categorical variables to be ordered properly on the horizontal, or X axis, of
# the univariate graph, we should convert your categorical variables, which are often formatted
# as numeric variables, into a format that Python recognizes as categorical.

# In our research question, we have W1_F1 (view on economic situation) has a categorical variable
# and W1_F1, view of the respondants about the future, and W1_P20 witch is a ordinal variable (Still)
# a type of categorical variable
Data["W1_G2"] = Data["W1_G2"].astype('category')
Data["W1_F1"] = Data["W1_F1"].astype('category')
Data["W1_P20"] = Data["W1_P20"].astype("category")

# Let's plot our categorical variables :
seaborn.countplot(x="W1_G2", data = Data)
plt.xlabel("-1 = refused, 1 = better, 2 = about the same, or 3 = worse")
plt.title("Respondants views on the nation's economy compared to one year ago")

seaborn.countplot(x = "W1_F1", data = Data)
plt.xlabel("-1 = refused, 1 = optimistic, 2 = neither optimistic nor pessimistic, 3 = pessimistic")
plt.title("Respondants views regarding their future")

seaborn.countplot(x = "W1_P20", data = Data)
plt.xlabel("Interval of annual income :1-7 (5k- 24k), 8-11(25k-49k) 12-15(50k-99k), 16-19 (100k-175k or more)")
plt.title("Income groups reported by respondents")

# Now let's display the graphics for the managed variables
# Graphing a quatitative variable
# The W1_P20 is not a ordinal variable, this is of example only
# seaborn.distplot(Data["W1_P20"].dropna(), kde = False)
# plt.xlabel("Group of personal annual income")
# plt.title("Income groups reported by respondents")

# Standard deviation and other descriptive statistics for quantitative variables
print("Describe the views of the economy's outcome")
desc1 = Data["W1_G2"].describe()
print(desc1)

print("Describe the views on the future by respondants")
desc2 = Data["W1_F1"].describe()
print(desc2)

print("Describe the personnal annual income for the respondants")
desc3 = Data["W1_P20"].describe()
print(desc3)

# ------------------ Make a decision about the role that each variable will play -----
#
# The explanatory variable is the  income level (W1_P20) the perception of the and the response
# variable nation’s economic situation (W1_G2 and/or W1_F1). Thus, using the graphing decisions
# flow chart we will use a Categorical to Categorical bar chart to plot the associations between
# our explanatory and response variables.
# We have to convert the categorical variables to numeric to do a C -> C bar chart.
# Setting variables you will be working with to numeric

#%%

#Data["W1_P20"] = Data["W1_P20"].convert_objects(convert_numeric=True)
#Data["W1_G2"] = Data["W1_G2"].convert_objects(convert_numeric=True)

Data["W1_P20"] = Data.apply(pandas.to_numeric, errors = 'coerce')
Data["W1_G2"] = Data.apply(pandas.to_numeric, errors = "coerce")

#%%
print("This is the C -> C graph US economy's situtation vs Personnal annual income")
seaborn.factorplot(x = "W1_P20", y = "W1_G2", data = Data, kind = "bar", ci = None)
plt.xlabel("Personnal annual income")
plt.ylabel("The US economy's situation")

seaborn.factorplot(x = "W1_P20", y = "W1_F1", data = Data, kind = "bar", ci = None)
plt.xlabel("Personnal annual income")
plt.ylabel("How the respondants think about the future ")
