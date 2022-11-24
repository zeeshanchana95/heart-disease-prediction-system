# importing required libraries
import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier


# Loading and Reading Dataset of Heart Disease
heart = pd.read_csv("heart.csv")

# making copy of dataset as a backup
heart_df = heart.copy()

# Renaming the names of some of the columns 
heart_df = heart_df.rename(columns={'condition':'target'})
print(heart_df.head())

# Start Building the Model
#Separating dataset columns for X and Y. Y contains target columns
x= heart_df.drop(columns= 'target')
y= heart_df.target

# Splitting dataset into Portions of Training and Testing
x_train, x_test, y_train, y_test= train_test_split(x, y, test_size= 0.25, random_state=42)

#feature scaling
scaler= StandardScaler()
x_train_scaler= scaler.fit_transform(x_train)
x_test_scaler= scaler.fit_transform(x_test)

# creating K-Nearest-Neighbor classifier
model=RandomForestClassifier(n_estimators=20)
model.fit(x_train_scaler, y_train)
y_pred= model.predict(x_test_scaler)
p = model.score(x_test_scaler,y_test)
print(p)

print('Classification Report\n', classification_report(y_test, y_pred))
print('Accuracy: {}%\n'.format(round((accuracy_score(y_test, y_pred)*100),2)))

cm = confusion_matrix(y_test, y_pred)
print(cm)

# Creating a pickle file for the classifier
filename = 'model.pkl'
pickle.dump(model, open(filename, 'wb'))

