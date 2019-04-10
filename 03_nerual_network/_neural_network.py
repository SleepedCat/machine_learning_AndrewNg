from scipy.io import loadmat
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt
from sklearn.metrics import classification_report
np.set_printoptions(suppress=True)


def load_data(path='../datasets/03_neural_network/data1.mat', transpose=True):
    data = loadmat(path)
    x_data = data.get('X')
    y_data = data.get('y')
    # 为了画出图像
    if transpose:
        size = int(np.sqrt(x_data.shape[1]))
        x_data = np.array([img.reshape((size, size)).T for img in x_data])
        x_data = np.array([img.reshape(np.square(size)) for img in x_data])
    return x_data, y_data


def plot_100_images(x_data):
    fig, axes = plt.subplots(nrows=10, ncols=10, sharex=True, sharey=True, figsize=(10, 10))
    size = int(np.sqrt(x_data.shape[1]))
    choices = np.random.randint(0, x_data.shape[0], 100)
    for i in range(10):
        for j in range(10):
            choice = choices[i * 10 + j]
            axes[i][j].matshow(x_data[choice, :].reshape((size, size)), cmap=plt.cm.binary)
            plt.xticks(())
            plt.yticks(())


# 训练分类器
def one_vs_all(y_data, classes):
    y_matrix = []
    for i in range(1, classes + 1):
        y_matrix.append((y_data.flatten() == i).astype(int))
    return np.array(y_matrix).T


# 激活函数
def sigmoid(z):
    return 1 / (1 + np.exp(-z))


# 代价函数
def cost(theta, x_data, y_data):
    theta, y_data = np.matrix(theta), np.matrix(y_data)
    h_theta_x = sigmoid(x_data @ theta.T)
    return -np.mean(np.multiply(y_data, np.log(h_theta_x)) + np.multiply(1 - y_data, np.log(1 - h_theta_x)))


# 正则化代价函数
def regularize_cost(theta, x_data, y_data, λ=1):
    theta = np.matrix(theta)
    return cost(theta, x_data, y_data) + λ / (2 * x_data.shape[0]) * np.sum(np.power(theta[:, 1:], 2))


# 梯度
def grad(theta, x_data, y_data):
    theta, y_data = np.matrix(theta), np.matrix(y_data)
    return (sigmoid(x_data @ theta.T) - y_data).T @ x_data / x_data.shape[0]


# 正则化梯度
def regularize_grad(theta, x_data, y_data, λ=1):
    theta = np.matrix(theta)
    gradient = grad(theta, x_data, y_data)
    gradient[:, 1:] = gradient[:, 1:] + λ / x_data.shape[0] * theta[:, 1:]
    return gradient


# 逻辑回归
def logistic_regression(theta, x_data, y_data, λ=1):
    res = opt.minimize(fun=regularize_cost, x0=theta, args=(x_data, y_data, λ), method='TNC', jac=regularize_grad)
    return res.x


def predict(x_data, theta):
    theta = np.matrix(theta)
    prob = sigmoid(x_data @ theta.T)
    return (prob >= 0.5).astype(int)


# --------------------- 训练神经网络 START ---------------------
# === 画出一部分图像 START ===
X, y_answer = load_data()
plot_100_images(X)
# === 划出一部分图像 END ===

# === 训练二分类 START ===
print('--- 训练二分类 START ---')
# === 查看形状 START===
X = np.insert(X, 0, values=np.ones(X.shape[0]), axis=1)
y = one_vs_all(y_answer, 10)
theta_origin = np.zeros(X.shape[1])
y_actual = np.matrix(y[:, -1]).T
print('X {}\ny {}\ntheta_origin {}\ny_actual {}\n'.format(X.shape, y.shape, theta_origin.shape, y_actual.shape))
# === 查看形状 END ===

# === 确定参数 START ===
theta_min = logistic_regression(theta_origin, X, y_actual)
print('theta_min {}'.format(theta_min))
print('')
# === 确定参数 END ===

# === 计算精确率 START ===
y_pred = predict(X, theta_min)
accuracy = np.mean(y_actual == y_pred)
print('Accuracy={}%'.format(accuracy * 100))
# === 计算精确率 END ===
print('--- 训练二分类 END ---')
print('')
# === 训练二分类 END ===

# === 训练多分类 START ===
print('--- 训练多分类 START ---')
# === 确定参数 START ===
theta_min_all = np.array([logistic_regression(theta_origin, X, np.matrix(y[:, col]).T) for col in range(y.shape[1])])
print('theta_min_all {} ---> {}\n'.format(theta_min_all.shape, theta_min_all))
# === 确定参数 END ===

# === 预测 START ===
probability_matrix = sigmoid(X @ theta_min_all.T)
print('probability_matrix {} ---> {}\n'.format(probability_matrix.shape, probability_matrix))
y_pred = np.argmax(probability_matrix, axis=1) + 1      # +1 是因为数组的索引是从0开始的
print('y_pred {} ---> {}'.format(y_pred.shape, y_pred))
print(classification_report(y_answer, y_pred))
# === 预测 END ===
print('--- 训练多分类 END ---')
print('')
# === 训练多分类 END ===

# === 加载已有的参数并进行评估 START===
print('--------------------- 前馈预测 START ---------------------')
params = loadmat('../datasets/03_neural_network/weights.mat')
Theta1, Theta2 = params.get('Theta1'), params.get('Theta2')
print('Theta1 {} ---> {}\n\nTheta2 {} ---> {}'.format(Theta1.shape, Theta1, Theta2.shape, Theta2))
X, y_answer = load_data(transpose=False)
X = np.insert(X, 0, values=np.ones(X.shape[0]), axis=1)
a1 = X
Z2 = a1 @ Theta1.T
a2 = sigmoid(Z2)
a2 = np.insert(a2, 0, values=np.ones(a2.shape[0]), axis=1)
Z3 = a2 @ Theta2.T
a3 = sigmoid(Z3)
print('probability_matrix {} --- > {}\n'.format(a3.shape, a3))
y_pred = np.argmax(a3, axis=1) + 1
print(classification_report(y_pred, y_answer))
# === 加载已有的参数并进行评估 END===
print('--------------------- 前馈预测 END ---------------------')
plt.show()
# --------------------- 训练神经网络 END ---------------------
