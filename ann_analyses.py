import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv('result_data.csv')

dataset.info()

dataset['max_accuracy'] = dataset['min'].apply(lambda x: 100-x)
dataset['min_accuracy'] = dataset['max'].apply(lambda x: 100-x)
dataset['mean_accuracy'] = dataset['mean'].apply(lambda x: 100-x)

top = dataset[dataset['max_accuracy'] > 90]
top

dataset.loc[:, ['min', 'max', 'mean']].describe()
dataset.loc[:, ['max_accuracy', 'min_accuracy', 'mean_accuracy']].describe()

dataset['min'].plot(kind='hist', color='blue')
# dataset['max'].plot(color='blue')
# dataset['mean'].plot(color='blue')
plt.hist()
plt.show()

# 0      50      5  10.0  8.99  24.72  18.43         91.01         75.28
# 45    150     20  15.0  8.96  41.04  24.55         91.04         58.96
