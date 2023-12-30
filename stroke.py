from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score , confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import pandas as pd
import numpy as np

from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier

df= pd.read_csv('healthcare-dataset-stroke-data.csv')
df.drop(['id'], axis=1, inplace=True)
df ["bmi"] = df["bmi"].replace(np.NaN, df["bmi"].mean())
df.dropna(inplace=True)
df.drop_duplicates(inplace=True)

gender_mapping = {'Male': 0, 'Female': 1, np.nan : 2}
ever_married_mapping = {'No': 0, 'Yes': 1, np.nan : 2}
work_type_mapping = {'Never_worked': 0, 'Govt_job': 1, 'Self-employed': 2, 'children' : 3, 'Private' : 4, np.nan : 5}
Residence_type_mapping = {'Rural': 0, 'Urban': 1, np.nan : 2}
smoking_status_mapping = {'smokes': 0, 'formerly smoked': 1, 'unknown': 2, 'never smoked' : 3, np.nan : 4}

df['gender'] = df['gender'].map(gender_mapping)
df['ever_married'] = df['ever_married'].map(ever_married_mapping)
df['work_type'] = df['work_type'].map(work_type_mapping)
df['Residence_type'] = df['Residence_type'].map(Residence_type_mapping)
df['smoking_status'] = df['smoking_status'].map(smoking_status_mapping)

df.dropna(inplace=True)
#Model

features = ["age",'bmi','hypertension', 'heart_disease', 'ever_married', 'work_type', 'avg_glucose_level', 'Residence_type', 'smoking_status','gender']

X = df[features]
Y = df['stroke']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3 , random_state=1255)
train_x,test_x,train_y,test_y=X_train, X_test, Y_train, Y_test 
# smote=SMOTE()
# train_x,train_y=smote.fit_resample(X_train,Y_train)
# test_x,test_y=smote.fit_resample(X_test,Y_test)

xgc=XGBClassifier(objective='binary:logistic',n_estimators=100000,max_depth=5,learning_rate=0.001,n_jobs=-1)
xgc.fit(train_x,train_y)
Y_pred=xgc.predict(test_x)
print(Y_pred.tolist())
accuracy = accuracy_score(test_y, Y_pred)
confusion = confusion_matrix(Y_test, Y_pred)
print("Accuracy --> ",accuracy)
print("Confusion Matrix:\n", confusion)


#train 

# rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
# model = LogisticRegression()

# rf_classifier.fit(X_train, Y_train)
# model.fit(X_train, Y_train)

# Y_pred = rf_classifier.predict(X_test)
# y_pred = model.predict(X_test)

# accuracy = accuracy_score(Y_test, y_pred)
# confusion = confusion_matrix(Y_test, y_pred)

# print(Y_test.tolist())
# print(Y_pred.tolist())

# print("Accuracy:", accuracy)
# print("Confusion Matrix:\n", confusion)

print("================================== Enter the following details to check if you are likely to have a stroke ==================================\n")

age=int(input("Enter age: "))
bmi=float(input("Enter bmi: "))
hypertension=int(input("Enter hypertension: "))
heart_disease=int(input("Enter heart_disease: "))
ever_married=int(input("Enter ever_married: 'No': 0, 'Yes': 1 : "))
work_type=int(input("Enter work_type: Never_worked': 0, 'Govt_job': 1, 'Self-employed': 2, 'children' : 3, 'Private' : 4 : "))
avg_glucose_level=float(input("Enter avg_glucose_level: "))
Residence_type=int(input("Enter Residence_type: Rural': 0, 'Urban': 1 : "))
smoking_status=int(input("Enter smoking_status: smokes': 0, 'formerly smoked': 1, 'unknown': 2, 'never smoked' : 3 : "))
gender=int(input("Enter Gender: 0 for Male and 1 for Female: "))

k=xgc.predict([[age,bmi,hypertension,heart_disease,ever_married,work_type,avg_glucose_level,Residence_type,smoking_status,gender]])

print("\n ==================================================== Result ==========================================================\n")
if k==0:
    print("You are safe")
else:
    print("You will most likely have a stroke")
