from scipy.io import loadmat
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import svm
import numpy as np
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV


def load_data(path='../datasets/06_svm/data3.mat'):
    data_origin = loadmat(path)
    x_data = data_origin.get('X')
    y_data = data_origin.get('y')
    data_frame_train = pd.DataFrame(x_data, columns=['x1', 'x2'])
    data_frame_train['y'] = y_data

    x_val = data_origin.get('Xval')
    y_val = data_origin.get('yval')
    data_frame_cv = pd.DataFrame(x_val, columns=['x1', 'x2'])
    data_frame_cv['y'] = y_val
    return data_frame_train, data_frame_cv


def plot_origin_data(data_frame):
    plt.figure(figsize=(10, 8))
    sns.set(font_scale=1.25)
    positive = data_frame[data_frame['y'] == 1]
    negative = data_frame[data_frame['y'] == 0]
    plt.scatter(positive['x1'], positive['x2'], s=20, c='b', marker='o', label='Positive')
    plt.scatter(negative['x1'], negative['x2'], s=30, c='r', marker='x', label='Negative')
    plt.legend(loc='best')
    plt.xlabel('X1')
    plt.ylabel('X2')


data_train, data_cv = load_data()
print(data_train.head(), data_cv.head(), sep='\n\n')
print('')
plot_origin_data(data_train)
plot_origin_data(data_cv)

candidate = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100]
combination = [(C, gamma) for C in candidate for gamma in candidate]
search = []
for c, gamma in combination:
    svc = svm.SVC(C=c, gamma=gamma)
    svc.fit(data_train[['x1', 'x2']], data_train['y'])
    search.append(svc.score(data_cv[['x1', 'x2']], data_cv['y']))
best_index = int(np.argmax(search))
c, gamma = combination[best_index]
best_svc = svm.SVC(C=c, gamma=gamma)
best_svc.fit(data_train[['x1', 'x2']], data_train['y'])
y_pred = best_svc.predict(data_cv[['x1', 'x2']])
print(classification_report(y_pred, data_cv['y']))

params = {'C': candidate, 'gamma': candidate}
svc = svm.SVC()
clf = GridSearchCV(svc, params, n_jobs=-1, cv=3)
clf.fit(data_train[['x1', 'x2']], data_train['y'])
best_param = clf.best_params_
c, gamma = best_param.get('C'), best_param.get('gamma')
y_pred = clf.predict(data_cv[['x1', 'x2']])
print(classification_report(y_pred, data_cv['y']))
plt.show()
