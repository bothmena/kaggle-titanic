import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, LabelEncoder
from sklearn.metrics import mean_squared_error
import itertools
import time
from keras.models import Sequential
from keras.layers import Dense

start = time.time()

dataset = pd.read_csv('Titanic/train.csv')

dataset.describe()
dataset.info()
dataset['Embarked'].unique()
tu = dataset['Ticket'].unique()
len(tu)

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
# le_0 = LabelEncoder()
# X[:, 0] = le_0.fit_transform(X[:, 0])
# oneHotEncoder = OneHotEncoder(categorical_features=[0, 1, 6])
# X = oneHotEncoder.fit_transform(X).toarray()
# X = X[:, [1, 2, 3, 5, 6, 8, 9, 10, 11]]

epochs = [50, 100, 150, 200]
batch = [5, 10, 15, 20]
test = [0.1, 0.15, 0.2, 0.25]
combinations = list(itertools.product(*[epochs, batch, test]))
data = []

for combo in combinations:

    accuracies = []
    for i in range(10):

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=combo[2])
        sc = MinMaxScaler()
        # X_train[:, 5:] = sc.fit_transform(X_train[:, 5:])  # 755, 9
        X_train = sc.fit_transform(X_train)  # 755, 9
        # X_test[:, 5:] = sc.transform(X_test[:, 5:])
        X_test = sc.transform(X_test)

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
        classifier.fit(X_train, y_train, batch_size=combo[1], epochs=combo[0])

        y_pred = classifier.predict(X_test)
        y_pred = (y_pred > 0.5)
        y_pred = np.reshape(y_pred, y_test.shape)
        cm = mean_squared_error(y_test, y_pred)
        accuracies.append(cm)

    data.append({
        'epoch': combo[0],
        'batch': combo[1],
        'test': combo[2]*100,
        'min': np.round(np.min(accuracies)*100, 2),
        'max': np.round(np.max(accuracies)*100, 2),
        'mean': np.round(np.mean(accuracies)*100, 2),
    })

    # if (cm < 0.15):
    #     with pd.read_csv('') as dataset_test:
    #         with open('results_' + str(cm), 'w') as f:
    #             pass
end = time.time()
elapsed_time = round(end-start, 2)

print("")
print("time of execution: " + str(elapsed_time//60) + ' mins, ' + str(round(elapsed_time%60, 2)) + ' s.')

with open('Titanic/result_data.csv', 'w') as f:
    f.write("epoch,batch,test,min,max,mean\n")
    for datium in data:
        f.write("{},{},{},{},{},{}\n".format(datium['epoch'], datium['batch'], datium['test'], datium['min'], datium['max'], datium['mean']))
