# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Gather data consisting of two variables. Input- a factor that affects the marks and Output - the marks scored by students
2.Plot the data points on a graph where x-axis represents the input variable and y-axis represents the marks scored
3.Define and initialize the parameters for regression model: slope controls the steepness and intercept represents where the line crsses the y-axis
4.Use the linear equation to predict marks based on the input Predicted Marks = m.(hours studied) + b
5.for each data point calculate the difference between the actual and predicted marks
6.Adjust the values of m and b to reduce the overall error. The gradient descent algorithm helps update these parameters based on the calculated error
7.Once the model parameters are optimized, use the final equation to predict marks for any new input data
## Program:
```
/*
/*
Program to implement the simple linear regression model for predicting the marks scored.

Developed by: HARISH.S

RegisterNumber: 212224240052  
*/


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error
df=pd.read_csv(r"C:\Users\admin\Downloads\student_scores.csv")
print(df)
df.head(0)
df.tail(0)
print(df.head())
print(df.tail())
x = df.iloc[:,:-1].values
print(x)
y = df.iloc[:,1].values
print(y)
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)
y_pred = regressor.predict(x_test)
print(y_pred)
print(y_test)
#Graph plot for training data
plt.scatter(x_train,y_train,color='black')
plt.plot(x_train,regressor.predict(x_train),color='blue')
plt.title("Hours vs Scores(Training set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
#Graph plot for test data
plt.scatter(x_test,y_test,color='black')
plt.plot(x_train,regressor.predict(x_train),color='red')
plt.title("Hours vs Scores(Testing set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
mse=mean_absolute_error(y_test,y_pred)
print('MSE = ',mse)
mae=mean_absolute_error(y_test,y_pred)
print('MAE = ',mae)
rmse=np.sqrt(mse)
print("RMSE= ",rmse)


*/
```

## Output:
## DATASET:
![image](https://github.com/user-attachments/assets/4c2ed000-03aa-4d49-8cfb-17082703e9fd)

## HEAD VALUES:
![image](https://github.com/user-attachments/assets/76ad756b-1f40-4473-a9d1-002949888ef0)
## TAIL VALUES:
![image](https://github.com/user-attachments/assets/18954e9f-cb1d-40d0-9e69-2bae8202a8c8)
## X AND Y VALUES:
![image](https://github.com/user-attachments/assets/14e001ee-7866-492f-9e49-f23876382849)
## PREDICTION OF X AND Y:
![image](https://github.com/user-attachments/assets/83c6e477-faaa-417e-a4b1-a3987eaaa559)
## MSE,MAE & RMSE
![image](https://github.com/user-attachments/assets/176a3311-4f74-41ee-af63-1dcf48be74cb)
## TRAINING DATA:
![image](https://github.com/user-attachments/assets/50631a5b-48b5-46f4-8d9a-c0c7df1231b0)
## TESTING DATA:
![image](https://github.com/user-attachments/assets/8e8636f8-b731-477c-8844-8a486a9dfff2)







## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
