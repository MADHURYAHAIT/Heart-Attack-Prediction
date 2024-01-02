from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score , confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import pandas as pd
import numpy as np
import pickle
import warnings
warnings.filterwarnings("ignore")

from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier

df= pd.read_csv('Heart.csv')


df.dropna(inplace=True)
df.drop_duplicates(inplace=True)

features = ["age","impluse","pressurehight","pressurelow","glucose","kcm","troponin"]
X = df[features]
Y = df['class']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3 , random_state=42)

rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)

rf_classifier.fit(X_train, Y_train)

pickle.dump(rf_classifier, open('model.pkl','wb'))
model=pickle.load(open('model.pkl','rb'))

# Y_pred = rf_classifier.predict(X_test)

# print(Y_pred.tolist())
# accuracy = accuracy_score(Y_test, Y_pred)
# confusion = confusion_matrix(Y_test, Y_pred)

# print("Accuracy:", accuracy)
# print("Confusion Matrix:\n", confusion)

# print("================================== Enter the following details to check if you are likely to have a stroke ==================================\n")

# age=int(input("Enter your age: "))
# gender=input("Enter Gender :")
# if gender.upper()=="MALE":
#     gender=1
# else:
#     gender=0
# impluce=int(input("Enter your impulse: "))
# pressurehight=int(input("Enter your pressurehight: "))
# pressurelow=int(input("Enter your pressurelow: "))
# glucose=float(input("Enter your glucose: "))
# kcm=float(input("Enter your kcm: "))
# troponin=float(input("Enter your troponin: "))

# k=rf_classifier.predict([[age,gender,impluce,pressurehight,pressurelow,glucose,kcm,troponin]])

# print("\n ================================================================ Result ====================================================================\n")
# if k=='negative':
#     print("You are safe")
# else:
#     print("You will most likely have Heart Attack")