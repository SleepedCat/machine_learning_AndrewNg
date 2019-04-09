import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.optimize as opt
from sklearn.metrics import classification_report
pd.set_option('display.max_columns', None)


def get_x(data_frame):
    ones = pd.DataFrame(data={'ones': np.ones(len(data_frame))})
    data_frame = pd.concat([ones, data_frame], axis=1)
    return data_frame.iloc[:, :-1].values


def get_y(data_frame):
    return data_frame.iloc[:, -1:].values


# 标准化处理
def normalize_feature(data_frame):
    return data_frame.apply(lambda feature: (feature - np.mean(feature)) / np.std(feature))


# 获取x_data, y_data的矩阵形式
def to_matrix(*args):
    params = []
    for param in args:
        params.append(np.matrix(param))
    return tuple(params)


# sigmoid函数(激活函数)
def sigmoid(z):
    return 1 / (1 + np.exp(-z))


# 代价函数
def cost(theta, x_data, y_data):
    theta, x_data, y_data = to_matrix(theta, x_data, y_data)
    h_theta_x = sigmoid(x_data @ theta.T)
    return -np.mean(np.multiply(y, np.log(h_theta_x)) + np.multiply(1 - y, np.log(1 - h_theta_x)))


# 加入正则化的代价函数
def regularize_cost(theta, x_data, y_data, λ=0):
    return cost(theta, x_data, y_data) + np.sum(np.power(theta[1:], 2)) * (λ / (2 * len(x_data)))


# 梯度，method1
def grad_loop(theta, x_data, y_data):
    theta, x_data, y_data = to_matrix(theta, x_data, y_data)
    error = sigmoid(x_data @ theta.T) - y_data
    gradient = np.matrix(np.zeros(theta.shape[1]))
    for j in range(theta.shape[1]):
        gradient[0, j] = np.mean(np.multiply(error, x_data[:, j]))
    return gradient


# 梯度, method2
def grad(theta, x_data, y_data):
    theta, x_data, y_data = to_matrix(theta, x_data, y_data)
    error = sigmoid(x_data @ theta.T) - y_data
    gradient = error.T @ x_data / x_data.shape[0]
    return gradient


# 加入正则化的梯度
def regularize_grad(theta, x_data, y_data, λ=0):
    theta, x_data, y_data = to_matrix(theta, x_data, y_data)
    gradient = grad(theta, x_data, y_data)
    gradient[:, 1:] = gradient[:, 1:] + λ / x_data.shape[0] * theta[:, 1:]
    return gradient


# 预测函数
def predict(theta, x_data):
    return (sigmoid(x_data @ theta.T) >= 0.5).astype(int)


# 特征映射
def feature_mapping(data_frame, x, y, power):
    data = {'f{}{}'.format(i - j, j): np.multiply(np.power(x, i - j), np.power(y, j))
            for i in np.arange(1, power + 1) for j in np.arange(i + 1)}
    data_frame1 = pd.DataFrame(data).sort_index(axis=1)
    data_frame2 = data_frame.iloc[:, -1:]
    return pd.concat([data_frame1, data_frame2], axis=1)


# === 绘制决策边界 START ===
def find_decision_boundary(data_frame, theta, density, power, threshold):
    t1 = np.linspace(data_frame['f10'].min(), data_frame['f10'].max(), density)
    t2 = np.linspace(data_frame['f01'].min(), data_frame['f01'].max(), density)
    cordinates = [(x_cord, y_cord) for x_cord in t1 for y_cord in t2]
    x_cords, y_cords = zip(*cordinates)
    data_frame = feature_mapping(data_frame, x_cords, y_cords, power)
    inner_product = get_x(data_frame) @ theta.T
    decision = data_frame[np.abs(inner_product) <= threshold]
    return decision.f10, decision.f01


def draw_boundary(data_frame, theta, power, λ, density=1000, threshold=2 * 10 ** -3):
    negative = data_frame[data_frame['accepted'] == 0]
    positive = data_frame[data_frame['accepted'] == 1]
    x, y = find_decision_boundary(data_frame, theta, density, power, threshold)
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.set()
    ax.scatter(negative['f10'], negative['f01'], s=50, c='r', marker='x', label='not admitted')
    ax.scatter(positive['f10'], positive['f01'], s=30, c='b', marker='o', label='admitted')
    plt.scatter(x, y, c='R', s=10, label='decision boundary')
    plt.title('Decision boundary')
    plt.xlabel('test1')
    plt.ylabel('test2')
    plt.title('Training set VS. Prediction')
    plt.legend(loc='best')
    plt.show()

# === 绘制决策边界 END ===


# --------------------- 数据集1 START ---------------------
# === 读取数据 START ===
path = '../datasets/02_logistic_regression/data1.txt'
data = pd.read_csv(path, header=None, names=['exam1', 'exam2', 'admitted'])
# === 读取数据 END ===

# === 显示原始数据集 START ===
plt.figure()
sns.set(style='darkgrid', font_scale=1.25, palette=sns.set_palette("Set1", 2))
sns.lmplot('exam1', 'exam2', data, hue='admitted', height=7, fit_reg=False, legend=False, scatter_kws={'s': 30})
# === 显示原始数据集 END ===

# === 获取X, y, theta START ===
X = get_x(data)
y = get_y(data)
theta_origin = np.zeros(X.shape[1])
print('--- 查看X, y, theta_origin的形状 START---')
print('X {}\ny {}\ntheta_origin {}'.format(X.shape, y.shape, theta_origin.shape))
print('--- 查看X, y, theta_origin的形状 END---')
print('')
# === 获取X, y, theta END ===

# === 可视化sigmoid函数 START ===
plt.figure(figsize=(10, 8))
z = np.linspace(-10, 10, 100)
plt.plot(z, sigmoid(z), linewidth=2)
plt.xlabel('z', fontsize=17)
plt.ylabel('g(z)', fontsize=17)
plt.title('sigmoid function')
# 设置X， Y轴
ax = plt.gca()
ax.spines['top'].set_color('None')
ax.spines['right'].set_color('None')
ax.xaxis.set_ticks_position('bottom')
ax.yaxis.set_ticks_position('left')
ax.spines['bottom'].set_position(('data', 0))
ax.spines['left'].set_position(('data', 0))
# === 可视化sigmoid函数 END ===

# === 测试代价函数的代码是否正确实现 START ===
cost_origin = cost(theta_origin, X, y)
print('--- 测试代价函数 START---')
print('cost_origin: {}'.format(cost_origin))
print('--- 测试代价函数 END---')
print('')
# === 测试代价函数的代码是否正确实现 END ===

# === 测试梯度的代码是否正确实现 START ===
gradient1 = grad_loop(theta_origin, X, y)
gradient2 = grad(theta_origin, X, y)
print('--- 测试梯度 START ---')
print('gradient1: {}\ngradient2: {}'.format(gradient1, gradient2))
print('--- 测试梯度 END ---')
print('')
# === 测试梯度的代码是否正确实现 END ===

# === 实现梯度下降算法 START ===
print('--- 梯度下降算法确定最终的参数，并进行可视化 START ---')
theta_min = opt.minimize(fun=regularize_cost, x0=theta_origin, args=(X, y), method='TNC', jac=regularize_grad).x
print('theta_min: {}'.format(theta_min))
y_pred = predict(theta_min, X)
print(classification_report(y, y_pred))
print('--- 梯度下降算法确定最终的参数，并进行可视化 END ---')
print('')
# === 实现梯度下降算法 END ===

# === 绘制决策边界 START ===
negative = data[data['admitted'] == 0]
positive = data[data['admitted'] == 1]
fig, ax = plt.subplots(figsize=(10, 8))
sns.set()
ax.scatter(negative['exam1'], negative['exam2'], s=50, c='r', marker='x', label='not admitted')
ax.scatter(positive['exam1'], positive['exam2'], s=30, c='b', marker='o', label='admitted')
theta_0, theta_1, theta_2 = theta_min[0], theta_min[1], theta_min[2]
x = np.linspace(data['exam1'].min(), data['exam1'].max(), 100)
f = -1 / theta_2 * (theta_0 + theta_1 * x)
plt.plot(x, f, color='grey', linewidth=2)
plt.xlabel('exam1')
plt.ylabel('exam2')
plt.title('Training set VS. Prediction')
plt.legend(loc='best')
# === 绘制决策边界 END ===
# --------------------- 数据集1 END ---------------------

# --------------------- 数据集2 START ---------------------
# === 显示原始数据集 START ===
path = '../datasets/02_logistic_regression/data2.txt'
data = pd.read_csv(path, header=None, names=['test1', 'test2', 'accepted'])
plt.figure()
sns.set(style='darkgrid', font_scale=1.25, palette=sns.set_palette("Set1", 2))
sns.lmplot('test1', 'test2', data, hue='accepted', height=8, fit_reg=False, legend=False, scatter_kws={'s': 30})
# === 显示原始数据集 END ===

# === 特征映射后的数据 START ===
print('--- 特征映射 START ---')
X = np.array(data.test1)
y = np.array(data.test2)
data = feature_mapping(data, X, y, power=6)
print(data.head(7))
X = get_x(data)
y = get_y(data)
theta_origin = np.zeros(X.shape[1])
print('X {}\ny {}\ntheta_origin {}'.format(X.shape, y.shape, theta_origin.shape))
print('--- 特征映射 END ---')
print('')
# === 特征映射后的数据 END ===

# === 测试代价函数的代码是否正确实现 START ===
cost_origin = regularize_cost(theta_origin, X, y, 1)
print('--- 测试代价函数 START---')
print('cost_origin: {}'.format(cost_origin))
print('--- 测试代价函数 END---')
print('')
# === 测试代价函数的代码是否正确实现 END ===

# === 测试梯度的代码是否正确实现 START ===
print('--- 测试梯度 START ---')
gradients = regularize_grad(theta_origin, X, y, λ=1)
print('gradient: {}'.format(gradients))
print('--- 测试梯度 END ---')
print('')
# === 测试梯度的代码是否正确实现 END ===

# === 实现梯度下降算法 START ===
print('--- 梯度下降算法确定最终的参数，并进行可视化 START ---')
theta_min = opt.minimize(fun=regularize_cost, x0=theta_origin, args=(X, y, 1), method='TNC', jac=regularize_grad).x
print('theta_min: {}'.format(theta_min))
y_pred = predict(theta_min, X)
print(classification_report(y, y_pred))
print('--- 梯度下降算法确定最终的参数，并进行可视化 END ---')
print('')
# === 实现梯度下降算法 END ===

# === 绘制决策边界 START ===
draw_boundary(data, theta_min, 6, 1)
# 过拟合
theta_min = opt.minimize(fun=regularize_cost, x0=theta_origin, args=(X, y, 0), method='TNC', jac=regularize_grad).x
draw_boundary(data, theta_min, 6, 0)
# 欠拟合
theta_min = opt.minimize(fun=regularize_cost, x0=theta_origin, args=(X, y, 100), method='TNC', jac=regularize_grad).x
draw_boundary(data, theta_min, 6, 100)
# === 绘制决策边界 END ===
# --------------------- 数据集2 END ---------------------
plt.show()

