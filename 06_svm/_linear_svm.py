import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.io import loadmat
from sklearn.svm import LinearSVC


def load_data(path='../datasets/06_svm/data1.mat'):
    data_origin = loadmat(path)
    x_data = data_origin.get('X')
    y_data = data_origin.get('y')
    data_frame = pd.DataFrame(x_data, columns=['x1', 'x2'])
    data_frame['y'] = y_data
    return data_frame


def plot_origin_data(data_frame=load_data()):
    plt.figure(figsize=(10, 8))
    sns.set(font_scale=1.25)
    positive = data_frame[data_frame['y'] == 1]
    negative = data_frame[data_frame['y'] == 0]
    plt.scatter(positive['x1'], positive['x2'], s=30, c='b', marker='o', label='Positive')
    plt.scatter(negative['x1'], negative['x2'], s=50, c='r', marker='x', label='Negative')
    plt.legend(loc='best')


data = load_data()
plot_origin_data(data)

# svc1 = LinearSVC(loss='hinge', C=1, max_iter=3000)
# svc1.fit(data[['x1', 'x2']], data['y'])
# print(svc1.score(data[['x1', 'x2']], data['y']))
# data['SVM1 Confidence'] = svc1.decision_function(data[['x1', 'x2']])
# plt.figure(figsize=(10, 8))
# plt.scatter(data['x1'], data['x2'], s=30, c=data['SVM1 Confidence'], cmap='RdBu')
# plt.title('SVM (C=1) Decision Confidence')

svc100 = LinearSVC(C=100, loss='hinge')
svc100.fit(data[['x1', 'x2']], data['y'])
print(svc100.score(data[['x1', 'x2']], data['y']))
data['SVM100 Confidence'] = svc100.decision_function(data[['x1', 'x2']])
plt.figure(figsize=(10, 8))
plt.scatter(data['x1'], data['x2'], s=30, c=data['SVM100 Confidence'], cmap='RdBu')
plt.title('SVM (C=100) Decision Confidence')


print(data.head())
plt.show()

