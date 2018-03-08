import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, LabelEncoder
from keras.wrappers.scikit_learn import KerasClassifier
from keras.models import Sequential
from keras.layers import Dense, Dropout


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


# Evaluation the ANN

# Improving the ANN
# Dropout Regularization to reduce overfitting if needed
# if you have overfitting you should apply Dropout to all the layers.

def build_classifier():
    optimizer = 'rmsprop'
    classifier = Sequential()
    classifier.add(Dense(100, input_shape=(9,), kernel_initializer='uniform', activation='relu'))
    # p = fraction of the input units to drop
    # start with p=0.1 , if you still have overfitting you try p+=0.1 until you reach 0.5 or you solve the overfitting problem
    # if you go over 0.5 you risk underfitting !
    classifier.add(Dropout(rate=0.2))
    classifier.add(Dense(100, kernel_initializer='uniform', activation='relu'))
    classifier.add(Dropout(rate=0.2))
    classifier.add(Dense(100, kernel_initializer='uniform', activation='relu'))
    classifier.add(Dropout(rate=0.2))
    classifier.add(Dense(100, kernel_initializer='uniform', activation='relu'))
    classifier.add(Dropout(rate=0.2))
    classifier.add(Dense(100, kernel_initializer='uniform', activation='relu'))
    classifier.add(Dropout(rate=0.2))
    classifier.add(Dense(1, kernel_initializer='uniform', activation='sigmoid'))
    classifier.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

    return classifier

# remove hyper params: batch_size (20) and epochs (150)
k_cls = KerasClassifier(build_fn=build_classifier, batch_size=25, epochs=100)
accuracies = cross_val_score(estimator=k_cls, X = X_train, y = y_train, cv = 10) #  , n_jobs=-1 (don't use it if you use tf with gpu)

mean = accuracies.mean()
variance = accuracies.std()

print("# mean = {}%".format(round(mean*100, 2)))
print("# variance = {}%".format(round(variance*100, 2)))
print("# accuracies: {}% -> {}%".format(round(accuracies.min()*100, 2), round(accuracies.max()*100, 2)))

# hyper_params = {
#     'batch_size': [25, 32],
#     'epochs': [100, 200],
#     'optimizer': ['adam', 'rmsprop'],
# }
#
# gs = GridSearchCV(estimator=k_cls, param_grid=hyper_params, scoring='accuracy', cv=10)
# gs = gs.fit(X_train, y=y_train)
#
# best_params = gs.best_score_
# best_accuracy = gs.best_score_




# mean = 0.8015789453397717
# accuracies: 0.63 -> 0.87
# variance = 6.84%

# mean = 0.801421052844901
# accuracies: 0.67 -> 0.84
# variance = 4.89%

# mean = 77.09%
# variance = 8.28%
# accuracies: 53.95% -> 85.53%

# parameter tuning
# Hyper parameters: batch_size, epochs, optimisers, nb of neurones in the net

# video from youtube:
# dataset (1316): train/validation/test = 80%/10%/10%
# Model : ANN:
#     Input: (None, 7)
#     Dense_1: (None, 100)
#     Dropout_1: rate: 0.3
#     Dense_2: (None, 2)
#     Output: (None, 100)

#     nb of epochs: 25

#  => Loss (tr/validation): 0.35/0.47
#  => Accuracy (tr/validation): 0.75/0.79



