import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, LabelEncoder
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense

dataset = pd.read_csv('train.csv')

dataset['Age'] = dataset['Age'].fillna(dataset['Age'].mean())
dataset = dataset[dataset['Embarked'].notnull()].reset_index(drop=True)

X = dataset.iloc[:, [2, 4, 5, 6, 7, 9, 11]].values
y = dataset.iloc[:, 1].values

le_1 = LabelEncoder()
X[:, 1] = le_1.fit_transform(X[:, 1])
le_6 = LabelEncoder()
X[:, 6] = le_6.fit_transform(X[:, 6])
oneHotEncoder_0 = OneHotEncoder(categorical_features=[0])
X = oneHotEncoder_0.fit_transform(X).toarray()[:, 1:]
oneHotEncoder_7 = OneHotEncoder(categorical_features=[7])
X = oneHotEncoder_7.fit_transform(X).toarray()[:, 1:]

epochs = 150
batch = 20
test = 0.15

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test)
sc = MinMaxScaler()
# X_train[:, 5:] = sc.fit_transform(X_train[:, 5:])  # 755, 9
X_train = sc.fit_transform(X_train)  # 755, 9
# X_test[:, 5:] = sc.transform(X_test[:, 5:])
X_test = sc.transform(X_test)

accuracies = []
for i in range(10):

    # Initialising the ANN
    classifier = Sequential()

    # Adding input layer and the first hidden layer
    classifier.add(Dense(5, input_shape=(9,), kernel_initializer='uniform', activation='relu'))

    # Adding a second layer
    classifier.add(Dense(5, kernel_initializer='uniform', activation='relu'))

    # Adding the output layer
    classifier.add(Dense(1, kernel_initializer='uniform', activation='sigmoid'))

    # Compiling the ANN
    classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Fitting the ANN to the training set
    classifier.fit(X_train, y_train, batch_size=batch, epochs=epochs)


    y_pred = classifier.predict(X_test)
    y_pred = (y_pred > 0.5)
    y_pred = np.reshape(y_pred, y_test.shape)
    cm = mean_squared_error(y_test, y_pred)
    accuracies.append(cm)

for acc in accuracies:
    print(str((1-acc)*100) + '%')

# if (cm < 0.15):
#     with pd.read_csv('') as dataset_test:
#         with open('results_' + str(cm), 'w') as f:
#             pass


# with open('Titanic/result_data.csv', 'w') as f:
#     f.write("epoch,batch,test,min,max,mean\n")
#     for datium in data:
#         f.write("{},{},{},{},{},{}\n".format(datium['epoch'], datium['batch'], datium['test'], datium['min'], datium['max'], datium['mean']))
