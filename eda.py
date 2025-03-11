#Importing libraries
import numpy as np
import pandas as pd 
import statsmodels.api as sm
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import confusion_matrix, accuracy_score, mean_squared_error, r2_score, roc_auc_score, roc_curve, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from lightgbm import LGBMClassifier
from sklearn.model_selection import KFold
import warnings
warnings.simplefilter(action = "ignore")

df=pd.read_csv("diabetes.csv",encoding="ISO-8859-1")
df.head()
#Feature information
df.info()
# Displaying the current columns.
df.columns
# Descriptive statistics of the data set accessed.
df.describe()
# The histagram of the Age variable was reached.
df["Age"].hist(edgecolor = "black");
print("Max Age: " + str(df["Age"].max()) + " Min Age: " + str(df["Age"].min()))


# Histogram and density graphs of all variables were accessed.
fig, ax = plt.subplots(4,2, figsize=(16,16))
sns.distplot(df.Age, bins = 20, ax=ax[0,0]) 
sns.distplot(df.Pregnancies, bins = 20, ax=ax[0,1]) 
sns.distplot(df.Glucose, bins = 20, ax=ax[1,0]) 
sns.distplot(df.BloodPressure, bins = 20, ax=ax[1,1]) 
sns.distplot(df.SkinThickness, bins = 20, ax=ax[2,0])
sns.distplot(df.Insulin, bins = 20, ax=ax[2,1])
sns.distplot(df.DiabetesPedigreeFunction, bins = 20, ax=ax[3,0]) 
sns.distplot(df.BMI, bins = 20, ax=ax[3,1])


# The distribution of the outcome variable in the data was examined and visualized.
f,ax=plt.subplots(1,2,figsize=(18,8))
df['Outcome'].value_counts().plot.pie(explode=[0,0.1],autopct='%1.1f%%',ax=ax[0],shadow=True)
plt.show()


# Correlation Matrix
df.corr()

# Correlation matrix graph of the data set
f, ax = plt.subplots(figsize= [20,15])
sns.heatmap(df.corr(), annot=True, fmt=".2f", ax=ax, cmap = "magma" )
ax.set_title("Correlation Matrix", fontsize=20)
plt.show()


plt.figure(figsize=(15, 30))
plt.subplot(4, 2, 2)
sns.boxplot(x=df['Age'], y=df['Glucose'], palette="pastel")
plt.title("Glucose Distribution by Age")
plt.xlabel("Age")
plt.ylabel("Glucose (%)")


plt.subplot(4, 2, 3)
sns.violinplot(x=df['Age'], y=df['BloodPressure'], palette="coolwarm")
plt.title("BloodPressure Score Distribution by Age")
plt.xlabel("Age")
plt.ylabel("BloodPressure")


plt.subplot(4, 2, 4)
sns.boxplot(x=df['Age'], y=df['SkinThickness'], palette="Set2")
plt.title("SkinThickness by Age")
plt.xlabel("Age")
plt.ylabel("SkinThickness")


plt.subplot(4, 2, 5)
sns.countplot(x="Insulin", hue="Age", data=df, palette="pastel")
plt.title("Insulin by Age")
plt.xlabel("Insulin")
plt.ylabel("Count")


plt.subplot(4, 2, 6)
sns.boxplot(x=df['Age'], y=df['Outcome'], palette="coolwarm")
plt.title("Outcome by Age")
plt.xlabel("Age")
plt.ylabel("Outcome")

plt.show()


#Distribution of Age by Glucose
plt.figure(figsize=(12,8))
sns.boxplot(x=df['Age'], y=df['Glucose'], palette="pastel")
plt.title("Glucose Distribution by Age")
plt.xlabel("Age")
plt.ylabel("Glucose (%)")
plt.show()

# Distribution of Blood Pressure by Age
plt.figure(figsize=(12,8))
sns.violinplot(x=df['Age'], y=df['BloodPressure'], palette="coolwarm")
plt.title("BloodPressure Score Distribution by Age")
plt.xlabel("Age")
plt.ylabel("BloodPressure")
plt.show()