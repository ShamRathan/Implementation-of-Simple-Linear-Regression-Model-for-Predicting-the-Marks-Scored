# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the required libraries and read the dataframe.
2. Assign hours to X and scores to Y.
3. Implement training set and test set of the dataframe.
4. Plot the required graph both for test data and training data.
5. Find the values of MSE , MAE and RMSE.

## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: S.Sham Rathan
RegisterNumber: 212221230093
*/

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error , mean_squared_error
df=pd.read_csv('student_scores.csv')
df.head()
df.tail()
X=df.iloc[;,1].values
X
Y=df.iloc[:,1].values
Y
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=1/3,random_state=0)
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train,Y_train)
Y_pred=regressor.predict(X_test)
Y_pred
Y_test
plt.scatter(X_train,Y_train,color='blue')
plt.plot(X_train,regressor.predict(X_train),color="red")
plt.title("Hours vs scores(Training set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
plt.scatter(X_test,Y_test,color='black')
plt.plot(X_train,regressor.predict(X_train),color="red")
plt.title("Hours vs scores(Test set)")
plt.xlabel("Hours")
plt.ylabel("Scores")

plt.show()mse=mean_squared_error(Y_test,Y_pred)
print('MSE= ',mse)

mae=mean_absolute_error(Y_test,Y_pred)
print('MAE= ',mae)

rmse=np.sqrt(mse)
print("RMSE =",rmse)
```

## Output:
![1](https://user-images.githubusercontent.com/93587823/228481030-e4a19c86-6b19-4634-84d0-4ea57a6d235e.png)
![2](https://user-images.githubusercontent.com/93587823/228481099-92035a7a-a702-46f1-a928-b4739ebbdd23.png)
![3](https://user-images.githubusercontent.com/93587823/228481141-bfa59176-5fdc-475c-9124-a5213c6be920.png)
![4](https://user-images.githubusercontent.com/93587823/228481204-6cb9e274-8222-4751-93be-dc67b3556f07.png)
![5](https://user-images.githubusercontent.com/93587823/228481277-fe1e17be-5cbe-459c-b5a2-987f51eef63e.png)
![6](https://user-images.githubusercontent.com/93587823/228481239-d29ae62b-024f-4d94-94f8-23082daa1313.png)
![7](https://user-images.githubusercontent.com/93587823/228481264-ab994fca-b22e-4694-88a0-8ab593737aa5.png)
![8](https://user-images.githubusercontent.com/93587823/228481330-8d1567f4-f6a8-4af0-8851-3db414effed4.png)
![9](https://user-images.githubusercontent.com/93587823/228481341-caae1a78-add3-4a1f-837c-440caf4c1d4f.png)


## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
