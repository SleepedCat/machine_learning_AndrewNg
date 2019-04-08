import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from sklearn import linear_model as lm


# === 处理数据 START ===
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
# === 处理数据 END ===


# 代价函数
def cost(theta, x_data, y_data):
    x_data, y_data = to_matrix(x_data, y_data)
    error = x_data @ theta.T - y_data
    return np.mean(np.power(error, 2)) / 2


# 计算梯度，method1
def grad_loop(theta, x_data, y_data):
    x_data, y_data = to_matrix(x_data, y_data)
    error = x_data @ theta.T - y_data
    gradient = np.matrix(np.zeros(theta.shape[1]))
    for j in range(theta.shape[1]):
        gradient[0, j] = np.mean(np.multiply(error, x_data[:, j]))
    return gradient


# 计算梯度，method2
def grad(theta, x_data, y_data):
    x_data, y_data = to_matrix(x_data, y_data)
    error = x_data @ theta.T - y_data
    gradient = error.T @ X / len(x_data)
    return gradient


# 梯度下降算法
def batch_gradient_descent(theta, x_data, y_data, epoch=500, alpha=0.01):
    cost_data = [cost(theta, x_data, y_data)]
    for i in range(epoch):
        theta = theta - alpha * grad(theta, x_data, y_data)
        cost_data.append(cost(theta, x_data, y_data))
    return theta, np.array(cost_data)


# Normal Equation
def normal_equation(x_data, y_data):
    return np.linalg.inv(x_data.T @ x_data) @ x_data.T @ y_data


# 利用TensorFlow实现线性回归
def linear_regression(x_data, y_data, alpha=0.01, epoch=500, optimizer=tf.train.GradientDescentOptimizer):
    # 可以理解成是函数中的形参
    X = tf.placeholder(dtype=tf.float32, shape=x_data.shape)
    y = tf.placeholder(dtype=tf.float32, shape=y_data.shape)
    # 创建变量
    with tf.variable_scope('linear_regression'):
        Weights = tf.get_variable(name='weights', shape=(X.shape[1], 1), initializer=tf.constant_initializer)
        y_pred = tf.matmul(X, Weights)
        loss = tf.matmul(y_pred - y, y_pred - y, transpose_a=True) / (2 * x_data.shape[0])
    opt_operation = optimizer(learning_rate=alpha).minimize(loss)
    # run Session
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        losses = []
        for i in range(epoch):
            _, loss_val, weights_val = sess.run([opt_operation, loss, Weights], feed_dict={X: x_data, y: y_data})
            losses.append(loss_val[0, 0])
            if len(losses) > 1 and np.abs(losses[-1] - losses[-2]) < 10 ** -9:
                break
    tf.reset_default_graph()
    return {'loss': losses, 'params': weights_val}


# --------------------- 数据集1 START ---------------------
# === 读取数据 START ===
path = '../datasets/01_linear_regression/data1.txt'
data = pd.read_csv(path, header=None, names=['population', 'profit'])
# === 读取数据 END ===

# === 显示原始数据集 START ===
sns.set()
sns.lmplot('population', 'profit', data, height=7, fit_reg=False)
# === 显示原始数据集 END ===

# === 获取X, y, theta START ===
X = get_x(data)
y = get_y(data)
theta_origin = np.matrix(np.zeros(X.shape[1]))
print('--- 查看X, y, theta_origin的形状 START---')
print('X {}\ny {}\ntheta_origin {}'.format(X.shape, y.shape, theta_origin.shape))
print('--- 查看X, y, theta_origin的形状 END---')
print('')
# === 获取X, y, theta END ===

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
epoch = 500
theta_final, costs = batch_gradient_descent(theta_origin, X, y, epoch)
print('theta_final: {}'.format(theta_final))
plt.figure(figsize=(9, 7))
sns.set()
plt.plot(np.arange(epoch + 1), costs, linewidth=2)
plt.xlabel('epochs')
plt.ylabel('costs')
plt.title('visualize cost data')
print('--- 梯度下降算法确定最终的参数，并进行可视化 END ---')
print('')
# === 实现梯度下降算法 END ===

# === 绘制决策边界 START ===
plt.figure(figsize=(9, 7))
plt.scatter(data.population, data.profit, s=20, label='Training set')
theta_0, theta_1 = theta_final[0, 0], theta_final[0, 1]
plt.plot(data.population, data.population * theta_1 + theta_0, linewidth=2, label='Prediction', c='r')
plt.xlabel('population')
plt.ylabel('costs')
plt.title('Training set VS. Prediction')
plt.legend(loc='best')
# === 绘制决策边界 END ===

# === sklearn实现线性回归 START ===
lr_model = lm.LinearRegression()
X = data.iloc[:, :-1].values
lr_model.fit(X, y)
x = X.flatten()
f = lr_model.predict(X).flatten()
fig, ax = plt.subplots(figsize=(12, 8))
ax.plot(x, f, 'r', label='Prediction')
ax.scatter(data.population, data.profit, label='Training set')
ax.legend(loc=2)
ax.set_xlabel('Population')
ax.set_ylabel('Profit')
ax.set_title('Predicted Profit vs. Population Size')
plt.show()
# === sklearn实现线性回归 END ===
# --------------------- 数据集1 END ---------------------

# --------------------- 数据集2 START ---------------------
# === 可视化代价曲线 START ===
path = '../datasets/01_linear_regression/data2.txt'
data = pd.read_csv(path, header=None, names=['square', 'bedrooms', 'price'])
data = normalize_feature(data)
X = get_x(data)
y = get_y(data)
theta_origin = np.matrix(np.zeros(X.shape[1]))
epoch = 500
theta_final, costs = batch_gradient_descent(theta_origin, X, y, epoch)
print('--- 利用梯度下降法确定最终参数 START ---')
print('theta_final: {}'.format(theta_final))
print('--- 利用梯度下降法确定最终参数 END ---')
print('')
plt.figure(figsize=(9, 7))
sns.set()
plt.plot(np.arange(epoch + 1), costs, linewidth=2)
plt.xlabel('epochs')
plt.ylabel('costs')
plt.title('visualize cost data')
# === 可视化代价曲线 END ===

# === 不同学习速率对应的代价曲线 START ===
fig, ax = plt.subplots(figsize=(16, 9))
sns.set(context='talk')
base = np.logspace(-1, -5, num=4)
candidate = np.sort(np.concatenate((base, base * 3)))
epoch = 500
for learning_rate in candidate:
    _, costs = batch_gradient_descent(theta_origin, X, y, epoch, alpha=learning_rate)
    ax.plot(np.arange(epoch+1), costs, label=learning_rate)
ax.set_xlabel('epoch', fontsize=18)
ax.set_ylabel('cost', fontsize=18)
ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
# === 不同学习速率对应的代价曲线 END ===

# === Normal Equation START ===
theta_final = normal_equation(X, y).T
print('--- 正规方程确定最终参数 START ---')
print('theta_final: {}'.format(theta_final))
print('--- 正规方程确定最终参数 END ---')
print('')
# === Normal Equation END ===

# === TensorFlow START ===
optimizer_dict = {'GD': tf.train.GradientDescentOptimizer,
                 'Adagrad': tf.train.AdagradOptimizer,
                  'Adam': tf.train.AdamOptimizer,
                  'Ftrl': tf.train.FtrlOptimizer,
                  'RMS': tf.train.RMSPropOptimizer
                  }
results = []
for name in optimizer_dict:
    res = linear_regression(X, y, epoch=2000, optimizer=optimizer_dict[name])
    res['name'] = name
    results.append(res)
print('--- TensorFlow确定最终参数 START ---')
for index, name in enumerate(optimizer_dict):
    print('{}: {}'.format(optimizer_dict[name].__name__, results[index]['params'].T))
print('--- TensorFlow确定最终参数 END ---')
print('')
fig, ax = plt.subplots(figsize=(16, 9))
sns.set(context='poster')
for res in results:
    loss_data = res['loss']
    ax.plot(np.arange(len(loss_data)), loss_data, label=res['name'])
ax.set_xlabel('epoch', fontsize=18)
ax.set_ylabel('cost', fontsize=18)
ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
ax.set_title('different optimizer', fontsize=18)
# === TensorFlow END ===
# --------------------- 数据集2 END ---------------------
plt.show()

