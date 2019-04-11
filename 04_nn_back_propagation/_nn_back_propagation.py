from scipy.io import loadmat
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import classification_report
import scipy.optimize as opt
np.set_printoptions(suppress=True)


def load_data(path='../datasets/04_nn_back_propagation/data1.mat', transpose=True):
    data = loadmat(path)
    x_data, y_data = data.get('X'), data.get('y')
    if transpose:
        size = int(np.sqrt(x_data.shape[1]))
        x_data = np.array([img.reshape(size, size).T for img in x_data])
        x_data = np.array([img.reshape(size * size) for img in x_data])
    return x_data, y_data


def plot_100_images(x_data):
    choices = np.random.randint(0, x_data.shape[0], 100)
    size = int(np.sqrt(x_data.shape[1]))
    fig, axes = plt.subplots(nrows=10, ncols=10, sharex=True, sharey=True, figsize=(10, 10))
    for i in range(10):
        for j in range(10):
            choice = choices[i * 10 + j]
            axes[i][j].matshow(x_data[choice].reshape(size, size), cmap=plt.cm.binary)
            plt.xticks(())
            plt.yticks(())


def one_vs_all(y_data, classes):
    y_matrix = []
    for k in range(1, classes + 1):
        y_matrix.append((y_data == k).astype(int).flatten())
    return np.array(y_matrix).T


# OneHotEncoder编码,作用与one_vs_all函数相同
def expand_y(y_data):
    return OneHotEncoder(categories='auto', sparse=False).fit_transform(y_data)


def load_weights(path='../datasets/04_nn_back_propagation/weights.mat'):
    weights = loadmat(path)
    return weights.get('Theta1'), weights.get('Theta2')


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def serialize(v1, v2):
    return np.concatenate((v1.flatten(), v2.flatten()))


def deserialize(v):
    return v[:25 * 401].reshape((25, 401)), v[25 * 401:].reshape((10, 26))


def feed_forward(theta, x_data):
    theta_1, theta_2 = deserialize(theta)
    a1 = x_data
    z2 = a1 @ theta_1.T
    a2 = sigmoid(z2)
    a2 = np.insert(a2, 0, values=np.ones(a2.shape[0]), axis=1)
    z3 = a2 @ theta_2.T
    a3 = sigmoid(z3)
    return a1, z2, a2, z3, a3


def cost(theta, x_data, y_data):
    _, _, _, _, h_theta_x = feed_forward(theta, x_data)
    total_cost = (np.multiply(y_data, np.log(h_theta_x)) + np.multiply(1 - y_data, np.log(1 - h_theta_x))).sum()
    return -total_cost / x_data.shape[0]


def regularize_cost(theta, x_data, y_data, λ=1):
    theta_1, theta_2 = deserialize(theta)
    reg_theta_1 = λ / (2 * x_data.shape[0]) * np.sum(np.power(theta_1[:, 1:], 2))
    reg_theta_2 = λ / (2 * x_data.shape[0]) * np.sum(np.power(theta_2[:, 1:], 2))
    return cost(theta, x_data, y_data) + reg_theta_1 + reg_theta_2


def gradient_sigmoid(z):
    return np.multiply(sigmoid(z), sigmoid(1 - z))


def gradient(theta, x_data, y_data):
    _, theta_2 = deserialize(theta)
    a1, z2, a2, z3, h_theta_x = feed_forward(theta, x_data)
    delta3 = h_theta_x - y_data
    delta2 = np.multiply((delta3 @ theta_2)[:, 1:], gradient_sigmoid(z2))
    grad_1 = delta2.T @ a1 / a1.shape[0]
    grad_2 = delta3.T @ a2 / a1.shape[0]
    return serialize(grad_1, grad_2)


def regularize_gradient(theta, x_data, y_data, λ=1):
    theta_1, theta_2 = deserialize(theta)
    theta_1[:, 0] = 0
    theta_2[:, 0] = 0
    reg_grad_1 = λ / x_data.shape[0] * theta_1
    reg_grad_2 = λ / x_data.shape[0] * theta_2
    reg_grad = serialize(reg_grad_1, reg_grad_2)
    return gradient(theta, x_data, y_data) + reg_grad


def nn_training(x_data, y_data):
    init_theta = np.random.uniform(-0.12, 0.12, 10285)  # 25*401 + 10*26

    _res = opt.minimize(fun=regularize_cost,
                        x0=init_theta,
                        args=(x_data, y_data, 1),
                        method='TNC',
                        jac=regularize_gradient,
                        options={'maxiter': 400})
    return _res


# --------------------- 根据已有的数据集与参数计算代价 START ---------------------
# === 画图 START ===
X, _ = load_data()
plot_100_images(X)
# === 画图 END ===

# === 获取数据 START ===
print('------- 获取数据 START -------')
X, y = load_data(transpose=False)
X = np.insert(X, 0, values=np.ones(X.shape[0]), axis=1)
# y = one_vs_all(y, 10)
y = expand_y(y)
theta1, theta2 = load_weights()
thetas = serialize(theta1, theta2)
print('X {}\ny {}\ntheta1 {}\ntheta2 {}\nthetas {}'.format(X.shape, y.shape, theta1.shape, theta2.shape, thetas.shape))
print('neural network units (not add bias) ---> input_layer: {}, hidden_layer {}, output_layer {}'.format(400, 25, 10))
print('------- 获取数据 END -------')
print('')
# === 获取数据 END ===

# === 计算代价 START ===
print('------- 计算代价 START -------')
_, _, _, _, hx = feed_forward(thetas, X)
print('hx {} ---> {}\n'.format(hx.shape, hx))
cost_all = cost(thetas, X, y)
print('cost: {}'.format(cost_all))
reg_cost_all = regularize_cost(thetas, X, y)
print('reg_cost: {}'.format(reg_cost_all))
print('------- 计算代价 END -------')
print('')
# === 计算代价 END ===
# --------------------- 根据已有的数据集与参数计算代价 END ---------------------

# --------------------- 反向传播算法 START ---------------------
grad1, grad2 = deserialize(gradient(thetas, X, y))
print('grad1 {}\ngrad2 {}\n'.format(grad1.shape, grad2.shape))
res = nn_training(X, y)
print(res)
final_theta = res.x
_, _, _, _, h = feed_forward(final_theta, X)
_, y_answer = load_data()
y_pred = np.argmax(h, axis=1) + 1
print(classification_report(y_answer, y_pred))
# --------------------- 反向传播算法 END ---------------------

# plt.show()
