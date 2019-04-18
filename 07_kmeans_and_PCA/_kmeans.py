from scipy.io import loadmat
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import  seaborn as sns


def load_data(path='../datasets/07_kmeans_and_PCA/data1.mat'):
    data = loadmat(path)
    x_data = data.get('X')
    data_frame = pd.DataFrame(x_data, columns=['X1', 'X2'])
    print(data_frame.head())
    print('')
    return data_frame


def plot_origin_data(data_frame):
    sns.set(font_scale=1.25)
    sns.lmplot('X1', 'X2', data_frame, height=7, legend=False, fit_reg=False)


# 随机初始化
def random_init(data_frame, K):
    return data_frame.sample(K).values


# 为每个点分配聚类中心
def find_cluster(x, _centroids):
    distances = np.apply_along_axis(func1d=np.linalg.norm, axis=1, arr=_centroids - x)
    return np.argmin(distances)


# 为所有的点分配聚类中心
def assign_cluster(data_frame, _centroids):
    data = data_frame.values
    return np.apply_along_axis(func1d=lambda x: find_cluster(x, _centroids), axis=1, arr=data)


# 将每个点的聚类中心加入原始dataframe中
def combine_data_C(data_frame, _C):
    data_frame_copy = data_frame.copy()
    data_frame_copy['C'] = _C
    # print(data_frame.head())
    # print('')
    return data_frame_copy


# 为每个点分配新的聚类中心
def new_centroids(data_frame):
    return data_frame.groupby(by='C').mean().values


# 计算代价
def cost(data_frame, _centroids, C):
    expand_centroids = _centroids[C]
    distances = np.apply_along_axis(func1d=np.linalg.norm, axis=1, arr=data_frame[['X1', 'X2']].values - expand_centroids)
    return np.mean(distances)


# 随机初始化一次的K_mean
def k_mean_once(data_origin, K, epoch=100, threshold=0.0001):
    _centroids = random_init(data_origin, K)
    print(_centroids, '\n')
    costs = []
    for _ in range(epoch):
        C = assign_cluster(data_origin, _centroids)
        data_dealt = combine_data_C(data_origin, C)
        costs.append(cost(data_dealt, _centroids, C))
        sns.lmplot('X1', 'X2', data_dealt, hue='C', height=7, legend=False, fit_reg=False)
        plt.scatter(_centroids[:, 0], _centroids[:, 1], c=['b', 'y', 'g'], s=100, linewidths=10)
        plt.title('The {} iteration'.format(_))
        plt.show()
        _centroids = new_centroids(data_dealt)
        if len(costs) > 1:
            if np.abs(costs[-1] - costs[-2]) / costs[-1] < threshold:
                break
    return C, _centroids, costs[-1]


# 多次随机初始化的K_mean
def k_mean(data_origin, K, iterations=10):
    tries = np.array([k_mean_once(data_origin, K) for _ in range(iterations)])
    least_cost_index = np.argmin(tries[:, -1])
    return tries[least_cost_index]


data_origin_1 = load_data()
data_origin_2 = load_data('../datasets/07_kmeans_and_PCA/data2.mat')
plot_origin_data(data_origin_1)
plot_origin_data(data_origin_2)

# k_mean_once(data_origin_2, K=3)
print(k_mean(data_origin_2, 3))

plt.show()
