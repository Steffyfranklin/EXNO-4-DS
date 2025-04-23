# EXNO:4-DS
# AIM:
To read the given data and perform Feature Scaling and Feature Selection process and save the
data to a file.

# ALGORITHM:
STEP 1:Read the given Data.
STEP 2:Clean the Data Set using Data Cleaning Process.
STEP 3:Apply Feature Scaling for the feature in the data set.
STEP 4:Apply Feature Selection for the feature in the data set.
STEP 5:Save the data to the file.

# FEATURE SCALING:
1. Standard Scaler: It is also called Z-score normalization. It calculates the z-score of each value and replaces the value with the calculated Z-score. The features are then rescaled with x̄ =0 and σ=1
2. MinMaxScaler: It is also referred to as Normalization. The features are scaled between 0 and 1. Here, the mean value remains same as in Standardization, that is,0.
3. Maximum absolute scaling: Maximum absolute scaling scales the data to its maximum value; that is,it divides every observation by the maximum value of the variable.The result of the preceding transformation is a distribution in which the values vary approximately within the range of -1 to 1.
4. RobustScaler: RobustScaler transforms the feature vector by subtracting the median and then dividing by the interquartile range (75% value — 25% value).

# FEATURE SELECTION:
Feature selection is to find the best set of features that allows one to build useful models. Selecting the best features helps the model to perform well.
The feature selection techniques used are:
1.Filter Method
2.Wrapper Method
3.Embedded Method

# CODING AND OUTPUT:
```
import pandas as pd
from scipy import stats
import numpy as np

df=pd.read_csv("/content/bmi.csv")
df.head()
```
![image](https://github.com/user-attachments/assets/d2a68016-91ad-4aa5-854d-7ae3505821d5)
```
df.dropna()
```
![image](https://github.com/user-attachments/assets/27b7360a-6a8e-47d9-93a6-3f51f2cb7713)
```
max_val=np.max(np.abs(df[['Height','Weight']]))
max_val
```
![image](https://github.com/user-attachments/assets/28dcf121-5d16-49c4-80de-72bf3cf72bb4)
```
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()

df[['Height','Weight']]=sc.fit_transform(df[['Height','Weight']])
df.head(10)
```
![image](https://github.com/user-attachments/assets/a3c2e8a7-5db8-4424-b368-639a1323f5d6)
```
from sklearn.preprocessing import Normalizer
nm=Normalizer()

df[['Height','Weight']]=nm.fit_transform(df[['Height','Weight']])
df
```
![image](https://github.com/user-attachments/assets/cab27239-7791-4778-ab11-6d3edd2f095a)
```
from sklearn.preprocessing import MaxAbsScaler
mas=MaxAbsScaler()

df[['Height','Weight']]=mas.fit_transform(df[['Height','Weight']])
df
```
![image](https://github.com/user-attachments/assets/b72a0317-3111-4e00-bf8f-afae9c1bc5f4)
```
from sklearn.preprocessing import RobustScaler
rs=RobustScaler()

df[['Height','Weight']]=rs.fit_transform(df[['Height','Weight']])
df.head(5)
```
![image](https://github.com/user-attachments/assets/0bf3e81c-5fa6-442b-b6fd-15725f9f1e6c)
```
from scipy.stats import chi2_contingency
import seaborn as sns
tips=sns.load_dataset('tips')
tips.head()
```
![image](https://github.com/user-attachments/assets/f71554fd-0582-42f1-bfdb-365283a520b3)
```
contingency_table=pd.crosstab(tips['sex'],tips['time'])
contingency_table
```
![image](https://github.com/user-attachments/assets/42820113-7e04-489e-8fd1-b3fde46765dc)
```
chi2,p,_,_=chi2_contingency(contingency_table)
print('chi-square statistic:',chi2)
print('p-value:',p)
```
![image](https://github.com/user-attachments/assets/f72b7b12-1942-4b68-9fcc-f3e85ce943a5)
```
from sklearn.feature_selection import SelectKBest,mutual_info_classif,f_classif
data={'Feature1':[1,2,3,4,5],'Feature2':['A','B','C','A','B'],'Feature3':[0,1,1,0,1],'Target':[0,1,1,0,1]}
df=pd.DataFrame(data)
df
```
![image](https://github.com/user-attachments/assets/5646edaa-8918-4375-870c-2d5ca56a9d1d)
```
x=df[['Feature1','Feature3']]
y=df['Target']
selector=SelectKBest(score_func=mutual_info_classif,k=1)
x_new=selector.fit_transform(x,y)
print('Selected features:',x_new)
```
![image](https://github.com/user-attachments/assets/5a18b14f-7aac-41dc-af94-4498b1ae34de)
```
selectedFeatureIndices=selector.get_support(indices=True)
selectedFeatures=x.columns[selectedFeatureIndices]
print('Selected features:',selectedFeatures)
```
![image](https://github.com/user-attachments/assets/c43f1030-b884-41af-aa45-e5f83593ce42)

# RESULT:
      Successfully  read the given data and perform Feature Scaling and Feature Selection process and save the data to a file. 
