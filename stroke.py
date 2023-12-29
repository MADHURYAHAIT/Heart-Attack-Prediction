from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
import pandas as pd

df= pd.read_csv('healthcare-dataset-stroke-data.csv')
print(df)