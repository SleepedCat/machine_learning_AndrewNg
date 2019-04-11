from sklearn.preprocessing import OneHotEncoder, LabelEncoder

'''
第一个特征，第一列 [0,1,0,1]，也就是说它有两个取值 0 或者 1，那么 one-hot 就会使用两位来表示这个特征，[1,0] 表示 0， [0,1] 表示 1
第二个特征，第二列 [0,1,2,0]，它有三种值，那么 one-hot 就会使用三位来表示这个特征，[1,0,0] 表示 0， [0,1,0] 表示 1，[0,0,1] 表示 2
第三个特征，第三列 [3,0,1,2]，它有四种值，那么 one-hot 就会使用四位来表示这个特征，[1,0,0,0] 表示 0， [0,1,0,0] 表示 1，[0,0,1,0] 表示 2，[0,0,0,1] 表示 3
'''
enc = OneHotEncoder(categories='auto')
enc.fit([[0, 0, 3],
         [1, 1, 0],
         [0, 2, 1],
         [1, 0, 2]])    # fit来学习编码
print(enc.transform([[0, 1, 3]]).toarray())    # 进行编码

'''
sparse=True 表示编码的格式，默认为 True，即为稀疏的格式，指定 False 则就不用 toarray() 了
'''
enc = OneHotEncoder(categories='auto', sparse=False)
enc.fit([[0, 0, 3],
         [1, 1, 0],
         [0, 2, 1],
         [1, 0, 2]])    # fit来学习编码
print(enc.transform([[0, 1, 3]]))
