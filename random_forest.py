import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix


dataset = pd.read_csv('Titanic/train.csv')

dataset.describe()
dataset.info()
dataset['Embarked'].unique()

dataset['Age'] = dataset['Age'].fillna(dataset['Age'].mean())
dataset = dataset[dataset['Embarked'].notnull()].reset_index(drop=True)

X = dataset.iloc[:, [2, 4, 5, 6, 7, 9, 11]].values
y = dataset.iloc[:, 1].values

# le_0 = LabelEncoder()
# X[:, 0] = le_0.fit_transform(X[:, 0])
# le_1 = LabelEncoder()
# X[:, 1] = le_1.fit_transform(X[:, 1])
# le_6 = LabelEncoder()
# X[:, 6] = le_6.fit_transform(X[:, 6])
# oneHotEncoder = OneHotEncoder(categorical_features=[0, 1, 6])
# X = oneHotEncoder.fit_transform(X).toarray()
# X = X[:, [0,1,3,5,6,8,9,10,11]]
le_1 = LabelEncoder()
X[:, 1] = le_1.fit_transform(X[:, 1])
le_6 = LabelEncoder()
X[:, 6] = le_6.fit_transform(X[:, 6])
oneHotEncoder = OneHotEncoder(categorical_features=[0])
X = oneHotEncoder.fit_transform(X).toarray()[:, 1:]
oneHotEncoder = OneHotEncoder(categorical_features=[7])
X = oneHotEncoder.fit_transform(X).toarray()[:, 1:]

tests = []
for i in range(10):

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15)

    sc = StandardScaler()
    # X_train[:, 5:] = sc.fit_transform(X_train[:, 5:])
    X_train = sc.fit_transform(X_train)
    # X_test[:, 5:] = sc.transform(X_test[:, 5:])
    X_test = sc.transform(X_test)
    classifier = RandomForestClassifier(n_estimators=50, criterion='gini')
    classifier.fit(X_train, y_train)

    y_pred = classifier.predict(X_test)

    cm = confusion_matrix(y_test, y_pred)
    tests.append((cm[0][1] + cm[1][0]) * 100 / (cm[0][0] + cm[0][1] + cm[1][1] + cm[1][0]))

for i, test in enumerate(tests):
    print('error[' + str(i) + '] = ' + str(test) )

print('MIN tests = ' + str(np.min(tests)))
print('MAX tests = ' + str(np.max(tests)))
print('average tests = ' + str(np.mean(tests)))
