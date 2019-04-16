from scipy.io import loadmat
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.optimize as opt


def load_data(path='../datasets/05_bias_and_variance/data1.mat'):
    data = loadmat(path)
    x_data, y_data, x_test, y_test, x_val, y_val = data.get('X'), data.get('y'), data.get('Xtest'), data.get('ytest'), data.get('Xval'), data.get('yval')
    print('X {}\ny {}\nX_test {}\ny_test {}\nX_val {}\ny_val {}\n'.format(x_data.shape, y_data.shape, x_test.shape, y_test.shape, x_val.shape, y_val.shape))
    return x_data, y_data, x_test, y_test, x_val, y_val


def cost(theta, x_data, y_data):
    theta = np.matrix(theta)
    error = x_data @ theta.T - y_data
    return (error.T @ error)[0, 0] / (2 * x_data.shape[0])


def regularize_cost(theta, x_data, y_data, λ=1):
    theta = np.matrix(theta)
    reg_term = λ / (2 * x_data.shape[0]) * np.sum(np.power(theta[:, 1:], 2))
    return cost(theta, x_data, y_data) + reg_term


def grad(theta, x_data, y_data):
    theta = np.matrix(theta)
    gradient = (x_data @ theta.T - y_data).T @ x_data / x_data.shape[0]
    return gradient


def regularize_grad(theta, x_data, y_data, λ=1):
    theta = np.matrix(theta)
    gradient = grad(theta, x_data, y_data)
    reg_term = (λ / x_data.shape[0]) * theta[:, 1:]
    gradient[:, 1:] = gradient[:, 1:] + reg_term
    return gradient


def linear_regression(x_data, y_data, λ=1):
    theta = np.ones(x_data.shape[1])
    # train it
    res = opt.minimize(fun=regularize_cost,
                       x0=theta,
                       args=(x_data, y_data, λ),
                       method='TNC',
                       jac=regularize_grad)
    return res


def plot_learning_curve(x_data, y_data, x_v, y_v, λ=0):
    training_cost, cv_cost = [], []
    for i in np.arange(1, 1 + x_data.shape[0]):
        x_i = np.matrix(x_data[:i, :])
        y_i = np.matrix(y_data[:i, :])
        theta_final = linear_regression(x_i, y_i, λ).x
        tc = regularize_cost(theta_final, x_i, y_i, λ)
        cv = regularize_cost(theta_final, x_v, y_v, λ)
        training_cost.append(tc)
        cv_cost.append(cv)
    plt.figure(figsize=(10, 8))
    sns.set(font_scale=1.25)
    plt.plot(np.arange(1, 1 + x_data.shape[0]), training_cost, label='training_cost')
    plt.plot(np.arange(1, 1 + x_data.shape[0]), cv_cost, label='cv_cost')
    plt.legend(loc='best')


def mapping_feature(x, power):
    data = {'f{}'.format(i): np.power(x, i) for i in range(1, 1 + power)}
    return pd.DataFrame(data)


def normalize_feature(df):
    return df.apply(lambda feature: (feature - feature.mean()) / feature.std())


def prepare_poly_feature(*args, power):
    def prepare(x):
        df = normalize_feature(mapping_feature(x, power))
        ones = pd.DataFrame(data={'ones': np.ones(len(df))})
        df = pd.concat((ones, df), axis=1)
        return df.values
    return [prepare(x) for x in args]


# --------------------- START ---------------------
# === 加载数据 START ===
X, y, X_test, y_test, X_val, y_val = load_data()
data_frame = pd.DataFrame(data={'water_level': X.flatten(), 'flow': y.flatten()})
plt.figure()
sns.set(font_scale=1.25)
sns.lmplot('water_level', 'flow', data_frame, height=7, fit_reg=False)
X, X_test, X_val = [np.insert(x, 0, np.ones(x.shape[0]), axis=1) for x in (X, X_test, X_val)]
# === 加载数据 END ===

# === 代价函数 START ===
theta_origin = np.ones(X.shape[1])
cost_origin = cost(theta_origin, X, y)
print('cost_origin: {}'.format(cost_origin))
# === 代价函数 END ===

# === 获得参数并进行可视化 START ===
theta_min = linear_regression(X, y, 1).x
_gradient = grad(theta_origin, X, y)
_gradient_reg = regularize_grad(theta_origin, X, y, 1)
print('gradient: {}\nregularized gradient: {}\ntheta_min: {}\n'.format(_gradient, _gradient_reg, theta_min))
plt.figure(figsize=(9, 7))
sns.set(font_scale=1.25)
sns.lmplot('water_level', 'flow', data_frame, height=7, fit_reg=False,)
plt.plot(X[:, 1:], theta_min[0] + theta_min[1] * X[:, 1:], c='r', linewidth=2)
# === 获得参数并进行可视化 END ===

# === 学习曲线 START ===
# === 原始状态 START ===
plot_learning_curve(X, y, X_val, y_val)
# === 原始状态 END ===

# === 加入多项式特征 START ===
# === 加入多项式特征后的X START ===
print(mapping_feature(X[:, 1:].flatten(), 3).head())
print('')
# === 加入多项式特征后的X END ===

X_poly, Xtest_poly, Xval_poly = prepare_poly_feature(X[:, 1:].flatten(), X_test[:, 1:].flatten(), X_val[:, 1:].flatten(), power=8)
print(X_poly[:3, :])
plot_learning_curve(X_poly, y, Xval_poly, y_val, 0)
plot_learning_curve(X_poly, y, Xval_poly, y_val, 100)
# === 加入多项式特征 END ===
# === 学习曲线 END ===

# === 不同λ对应的数值 START ===
l_candidate = [0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10]
t_cost, c_cost = [], []
for l in l_candidate:
    theta_m = linear_regression(X_poly, y, l).x
    tc = cost(theta_m, X_poly, y)
    cv = cost(theta_m, Xval_poly, y_val)
    t_cost.append(tc)
    c_cost.append(cv)
plt.figure(figsize=(10, 8))
sns.set(font_scale=1.25)
plt.plot(l_candidate, t_cost, label='training')
plt.plot(l_candidate, c_cost, label='cross validation')
plt.legend(loc=9)
plt.xlabel('lambda')
plt.ylabel('cost')
# === 不同λ对应的数值 END ===
# --------------------- END ---------------------
plt.show()

