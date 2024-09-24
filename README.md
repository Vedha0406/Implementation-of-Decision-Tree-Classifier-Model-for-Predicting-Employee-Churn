# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.  Start the program
2.  Attach the given data file
3.  Now find the satisfaction level of employee data
4.  Find the accuracy and new predict value
5.  End the program

## Program:

/*
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: VEDHASHREE.G
RegisterNumber: 212223240171
*/
```
import pandas as pd
data=pd.read_csv("C:/Users/admin/Desktop/INTR MACH/Employee.csv")

data.head()

data.info()

data.isnull().sum()

data["left"].value_counts()

from sklearn.preprocessing import LabelEncoder
le= LabelEncoder()

data["salary"]= le.fit_transform(data["salary"])
data.head()

x = data[["satisfaction_level" , "last_evaluation" , "number_project" ,"average_montly_hours" , "time_spend_company" ,"Work_accident" , "promotion_last_5years" , "salary"]]
x.head()

from sklearn.model_selection import train_test_split
x_train , x_test , y_train , y_test = train_test_split(x,y,test_size=0.2 , random_state= 100)

from sklearn.tree  import DecisionTreeClassifier
dt = DecisionTreeClassifier(criterion= "entropy")
dt.fit(x_train, y_train)
y_pred= dt.predict(x_test)

from sklearn import metrics
accuracy= metrics.accuracy_score(y_test , y_pred)
accuracy

dt.predict([[0.5,0.8,9,260,6,0,1,2]])
```
## Output:
![369765840-a48dd485-4cd1-4b7a-bf84-235dc7bca53c](https://github.com/user-attachments/assets/72c21806-3816-4400-8f8e-13adab55c0bf)


![369761483-51ca152d-93db-4630-8bb0-1ce89b7935aa](https://github.com/user-attachments/assets/df515836-798c-4fce-a21c-6af2e930f955)

## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
