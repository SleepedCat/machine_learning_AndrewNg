import pandas as pd
from sklearn import svm
from scipy.io import loadmat
import matplotlib.pyplot as plt
import seaborn as sns


def load_data(path='../datasets/06_svm/data2.mat'):
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
    plt.scatter(positive['x1'], positive['x2'], s=20, c='b', marker='o', label='Positive')
    plt.scatter(negative['x1'], negative['x2'], s=30, c='r', marker='x', label='Negative')
    plt.legend(loc='best')
    plt.xlabel('X1')
    plt.ylabel('X2')


data = load_data()
plot_origin_data()

svc = svm.SVC(C=100, gamma=10, probability=True)
svc.fit(data[['x1', 'x2']], data['y'])
print(svc.score(data[['x1', 'x2']], data['y']))

pp = svc.predict_proba(data[['x1', 'x2']])[:, 0]
fig, ax = plt.subplots(figsize=(8,6))
ax.scatter(data['x1'], data['x2'], s=30, c=pp, cmap='Blues', linewidth=0.1)
plt.show()
