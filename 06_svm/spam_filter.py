from scipy.io import loadmat
from sklearn import svm
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression


def load_data(path='../datasets/06_svm/spamTrain.mat', key1='X', key2='y'):
    data_origin = loadmat(path)
    x_data, y_data = data_origin.get(key1), data_origin.get(key2)
    return x_data, y_data


x_train, y_train = load_data()
x_test, y_test = load_data(path='../datasets/06_svm/spamTest.mat', key1='Xtest', key2='ytest')
print('x_train {}\ny_train {}\nx_test {}\ny_test {}\n'.format(x_train.shape, y_train.shape, x_test.shape, y_test.shape))

svc = svm.SVC()
svc.fit(x_train, y_train.ravel())
y_pred = svc.predict(x_test)
print(classification_report(y_test, y_pred))

lr = LogisticRegression()
lr.fit(x_train, y_train.ravel())
y_pred = lr.predict(x_test)
print(classification_report(y_test, y_pred))
