#using Random Forest

#imports
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

#Loading the data
df = pd.read_csv('sonar data.csv', header=None)

#to see all columns
pd.set_option('display.max_columns', 70)
print(df.head())
print(df.shape)

#more info abt dataset
print(df.describe())
# df.info()

print(df.isnull().sum())

#splitting data intp features and labels
y = df[60]
X = df.drop(columns=[60])

#train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=777) 
#stratify->split the dataset in such a way that both the train_set and test_set have correct proportions of R and M

#Logistic Regression model

rfc = RandomForestClassifier()
rfc.fit(X_train, y_train)
rfc_prediction = rfc.predict(X_test)

rfc_accuracy = accuracy_score(y_test, rfc_prediction)
print(rfc_accuracy) #0.8571428571428571


# Plotting the confusion matrix
conf_matrix = confusion_matrix(y_test, rfc_prediction)
plt.figure(figsize=(10,7))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['M', 'R'], yticklabels=['M', 'R'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# Classification report
print(classification_report(y_test, rfc_prediction))


