# -*- coding: utf-8 -*-
"""
Created on Sun Sep 22 17:21:48 2019

@author: lucky yadav
"""
#============================================================================================
#CLASSIFYING PERSONNEL INCOME
#============================================================================================
####################################REQUIRE PACKAGES ##########################################
#to change the directory
import os
#to perform numercal operation
import numpy as np
#TO work with dataframe
import pandas as pd
#to visualize the data
import seaborn as sns
#to partion the data
from sklearn.model_selection import train_test_split
#Importing library for the LogisticRegression
from sklearn.linear_model import LogisticRegression
#Importing performance metrics-accuracy score ,confusion_matrix
from sklearn.metrics import accuracy_score,confusion_matrix
os.chdir("D:\python programs")
###############################################################################################
#=========================================================================================
#importing data
#==================================================================================================
data_income=pd.read_csv("income.csv")
#==========================================================================================
#creating a copy of original data 
data=data_income.copy()
"""
#Exploratory data analysis:
#1.getting to know the data
#2.Data preprocessing(Missing values)
#3.Cross tables and data visulization
"""
#============================================================================
#Getting the know the data
#============================================================================
#*******To check the variable data type
print(data.info())


#******Check for missing values!
data.isnull()

print("Data columns with null values:\n",data.isnull().sum())
#******* No missinvsluesg  
#*** summary of numerical variables
summary_num=data.describe()
print(summary_num)

#*** summary of categeorical  variables
summary_cate=data.describe(include="O")
print(summary_cate)
#*****Frequency of eaach categeries
data["JobType"].value_counts()
data["occupation"].value_counts()
#*****There is exist "?" instead of nan
"""
GO back and read the data by including "na_values[" ?"] to distinct values
"""
data=pd.read_csv("income.csv",na_values=[" ?"])
#=====================================================================================
#Data pre-processing
#===================================================================================
data.isnull().sum()

missing =data[data.isnull().any(axis=1)]

#axis=1=>to consider atleast one column value is missing
"""
Points to note:
1.Missing values in jobtype  1809
2.Missing values in occpation=1816
3.There are 1809 rows where two specific columns i.e occupation & JobType have missing values
4.(1816-1809)=7=>you still have occupation unfilled for these 7 rows.Because ,jobType is never worked
    
"""
data2=data.dropna(axis=0)
#Relationship between independent variables
correlation=data2.corr()
#========================================================================================
#Crosstable & data visulizaton
#=============================================================================================
#Extracting column names
data2.columns
#=====================================================================================
gender= pd.crosstab(index=data2["gender"],columns='counts',normalize=True)
print(gender)

#========================================================================================
#Gender vs salary status:
#==============================================================================================================
gender_salsat=pd.crosstab(index=data2["gender"],columns=data2["SalStat"],margins=True,normalize="index")
print(gender_salsat)
#========================================================================================
#Frequency distribution of the salary distribution
#========================================================================================
SalStat=sns.countplot(data["SalStat"])
"""75% of the people'salary status is<=50,000
&25% of the people's salary status is >50,000
"""
######################Histogram of the age##################################
sns.distplot(data2["age"],bins=10,kde=False)
#people with age 20-45 age are the high in frequency
######################Boxplot age are high in frequency######################
sns.boxplot("SalStat","age",data=data2)
data2.groupby("SalStat").median()
#People with 35-50 age are more Likely to earn >50000
sns.countplot(y="JobType",data=data2,)
job_vs_salary=pd.crosstab(index=data2["JobType"],columns=data2["SalStat"],margins=True,normalize="index")
print(job_vs_salary)
sns.countplot(y="EdType",data=data2)
edu_vs_salary=pd.crosstab(index=data2["EdType"],columns=data2["SalStat"],margins=True,normalize="index")
print(edu_vs_salary)
sns.countplot(y="occupation",data=data2)
occ_vs_salary=pd.crosstab(index=data2["occupation"],columns=data2["SalStat"],margins=True,normalize="index")
print(occ_vs_salary)
sns.distplot(data2["capitalgain"],bins=10,kde =False)
sns.distplot(data2["capitalloss"],bins=10,kde =False)
sns.boxplot(x=data2["SalStat"],y=data2["hoursperweek"])
#===============================================================================
#Logistic Regression 
#=============================================================================
#Reindexing the salary names to 0,1
data2["SalStat"]=data2["SalStat"].map({" less than or equal to 50,000":0," greater than 50,000":1})
print(data2["SalStat"])


#storing the column names
new_data=pd.get_dummies(data2,drop_first=True)
columns_list=list(new_data.columns)
print(columns_list)
#seperating the input names fro m the data
features=list(set(columns_list)-set(["SalStat"]))
print(features)
#storing the output values in y
y= new_data["SalStat"].values
print(y)
#storing the values from input featchers
x=new_data[features].values
print(x)
#Spliting the daata into train and text
train_x,test_x,train_y,test_y=train_test_split(x,y,test_size=0.3)
#make instance of the model
logistic =LogisticRegression()
##fitting the values for x and y
logistic.fit(train_x,train_y)
logistic.coef_
logistic.intercept_
##prediction of test data
prediction= logistic.predict(test_x)
print(prediction)
##confusion matrix
confusion_matrix=confusion_matrix(test_y,prediction)
print(confusion_matrix)
##Calculate the accurcy
accuracy_score=accuracy_score(test_y,prediction)
print(accuracy_score)
##Printing the misclassified values from prediction
print("Misclassified samples :",(test_y!=prediction).sum())
#Logistic Regression -Removing  in-significant variable
data2["SalStat"]=data2["SalStat"].map({" less than or equal to 50,000":0," greater than 50,000":1})
print(data2["SalStat"])
cols=["gender","nativecountry","race","JobType","age","hoursperweek"]
new_data=data2.drop(cols,axis=1)
new_data2=pd.get_dummies(new_data,drop_first=True)
#storing columns names
columns_list=list(new_data.columns)
print(columns_list)
#seprating input features from data
features=list(set(columns_list)-set(["SalStat"]))
print(features)
# Storing output values in y
y=new_data["SalStat"].values
print(y)
#storing the values from input features
x=new_data[features].values
#spliting the values into train and test
train_x,test_x,train_y,test_y=train_test_split(x,y,test_size=0.3)
#Make instance of the model
logistic =LogisticRegression()
logistic.fit(train_x,train_y)
#prediction of test data
prediction= logistic.predict(test_x)
#Calculate the accurcy
accuracy_score=accuracy_score(test_y,prediction)
print(accuracy_score)
#Printing the misclassified values from prediction
print("Misclassified samples :",(test_y!=prediction).sum())

















  



