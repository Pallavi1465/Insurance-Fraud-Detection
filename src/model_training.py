import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
data=pd.read_csv('data/processed/cleaned_insurance_data.csv')
data.drop(columns=["_c39"], inplace=True, errors='ignore')
print(data.info())
print(data.isnull().sum())

X = data.drop('fraud_reported', axis=1)
y = data['fraud_reported']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Decision Tree
def dtc():
    dt_model = DecisionTreeClassifier(max_depth=5, criterion='gini', min_samples_split=2,min_samples_leaf=5, 
                                      class_weight='balanced', random_state=42)
    dt_model.fit(X_train, y_train)
    dt_predictions = dt_model.predict(X_test)
    
    print("Decision Tree Classification Report:")
    print(classification_report(y_test, dt_predictions))
    print("Train Accuracy:", dt_model.score(X_train, y_train))
    print("Test Accuracy:", dt_model.score(X_test, y_test))
dtc()

    # K-Nearest Neighbors
def knn():
    knn_model = KNeighborsClassifier(n_neighbors=30)
    knn_model.fit(X_train, y_train)
    knn_predictions = knn_model.predict(X_test)
    
    print("KNN Classification Report:")
    print(classification_report(y_test, knn_predictions))
knn()
    # Random Forest
def rfc():
    rf_model = RandomForestClassifier(class_weight='balanced', n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    rf_predictions = rf_model.predict(X_test)
    
    print("Random Forest Classification Report:")
    print(classification_report(y_test, rf_predictions))
rfc()
    # Logistic Regression
def lr():    
    lr_model = LogisticRegression(max_iter=1000, random_state=42)
    lr_model.fit(X_train, y_train)
    lr_predictions = lr_model.predict(X_test)
    
    print("Logistic Regression Classification Report:")
    print(classification_report(y_test, lr_predictions))
lr()
def svc():
    svc_model = SVC(kernel='rbf', random_state=42)
    svc_model.fit(X_train, y_train)
    svc_predictions = svc_model.predict(X_test)
    
    print("SVM Classification Report:")
    print(classification_report(y_test, svc_predictions))
svc()
import pickle
# Save the best model (decision tree) to a file
best_model = DecisionTreeClassifier(random_state=42)
best_model.fit(X_train, y_train)    
with open('best_model.pkl', 'wb') as f:
    pickle.dump(best_model, f)
