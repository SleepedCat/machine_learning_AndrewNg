from sklearn.cluster import KMeans
from scipy.io import loadmat
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def combine_data_C(data_frame, _C):
    data_frame_copy = data_frame.copy()
    data_frame_copy['C'] = _C
    # print(data_frame.head())
    # print('')
    return data_frame_copy


X = loadmat('../datasets/07_kmeans_and_PCA/data2.mat').get('X')
data = pd.DataFrame(X, columns=['X1', 'X2'])

sk_kmeans = KMeans(n_clusters=3)
sk_kmeans.fit(data)
sk_C = sk_kmeans.predict(data)
data_with_c = combine_data_C(data, sk_C)
sns.set(font_scale=1.25)
sns.lmplot('X1', 'X2', hue='C', data=data_with_c, fit_reg=False, legend=False, height=7)
plt.show()
