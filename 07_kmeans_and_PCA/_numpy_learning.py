import numpy as np

'''
np.pad(array, pad_width, mode, **kwargs)
pad_width:
    输入方式：((before_1, after_1), (before_2, after_2), ...)
mode:
    -- constant: 表示连续填充相同的值，每个轴可以分别指定填充值，constant_values=（x, y）时前面用x填充，后面用y填充，缺省值填充0
    -- edge: 表示用边缘值填充
    -- linear_ramp: 表示用边缘递减的方式填充
    -- maximum: 表示最大值填充
    -- mean: 表示均值填充 
    -- median: 表示中位数填充
    -- minimum: 表示最小值填充
    -- reflect: 表示对称填充
    -- symmetric: 表示对称填充
    -- wrap: 表示用原数组后面的值填充前面，前面的值填充后面
'''
z = np.arange(9).reshape((3, 3))
z1 = np.pad(z, (1, ), 'constant', constant_values=(7, ))
z2 = np.pad(z, (2, 1), 'constant', constant_values=(3, 7))
z3 = np.pad(z, ((2, 1), (2, 1)), 'constant', constant_values=(3, 7))
# print(z, z1, z2, z3, sep='\n\n')

'''
np.diag(v, k=0)
'''
z = np.arange(16).reshape((4, 4))
z1 = np.diag(z)
z2 = np.diag(z1)
z3 = np.diag(z, k=1)
z4 = np.diag(z, k=2)
z5 = np.diag(z, k=-1)
z6 = np.diag(z, k=-2)
z7 = np.diag(z1, k=-1)
z8 = np.diag(z1, k=1)
# print(z, z1, z2, z3, z4, z5, z6, z7, z8, sep='\n\n')

'''
checkerboard pattern 
'''
z1 = np.zeros((8, 8), dtype=int)
z1[::2, ::2] = 1
z1[1::2, 1::2] = 1
z2 = np.zeros((8, 8), dtype=int)
z2[1::2, ::2] = 1
z2[::2, 1::2] = 1
# print(z1, z2, sep='\n\n')

'''
np.unravel_index(indices, shape, order='C)
order:
    -- C
    --F
'''
index1 = np.unravel_index(22, (7, 6))
index2 = np.unravel_index(22, (7, 6), 'F')
index3 = np.unravel_index([22, 41, 37], (7, 6))
# print(index1, index2, index3, sep='\n')

'''
np.tile(A, reps)
'''
z = np.array([[1, 2], [3, 4]])
z1 = np.tile(z, 2)
z2 = np.tile(z, (2, 1))
z3 = np.tile(z, (2, 2))
# print(z, z1, z2, z3, sep='\n\n')

'''
negate all elements between 3 and 7 in place
'''
z = np.arange(1, 11)
z1 = z[(3 <= z) & (z <= 7)]
z[z1] *= -1
# print(z1, type(z1), z, sep='\n\n')

'''
np.random.uniform(low=0.0, high=1.0, size=None)
np.copysign(x1, x2, *args, **kwargs): 将x2的符号赋给x1
np.ceil(x, *args, **kwargs): 向上舍入
'''
n1 = np.copysign(3.7, -7.31)
n2 = np.copysign([3, 7, -1], -1.3)
n3 = np.copysign([3, 7, -1], [-1, 3, 7])
n4 = np.ceil(np.array([-1.7, -1.5, -0.2, 0.2, 1.5, 1.7, 2.0]))
# print(n1, n2, n3, n4, sep='\n')

'''
np.intersect1d(ar1, ar2, assume_unique=False, return_indices=False)
'''
z1 = np.intersect1d([1, 3, 4, 3], [3, 1, 2, 1])
z2 = np.intersect1d([1, 3, 4, 3], [1, 7, 2, 3], return_indices=True)
# print(z1, z2, sep='\n')

'''
np.sqrt()  np.emath.sqrt()
'''
n1 = np.sqrt(-1)
n2 = np.emath.sqrt(-1)
# print(n1, n2, sep='\n')

'''
compute (A + B) * (-A / 2) without copy
'''
A = np.ones(3)
B = np.ones(3) * 2
np.add(A, B, out=B)
np.divide(A, 2, out=A)
np.negative(A, out=A)
np.multiply(A, B, out=A)
# print(A)

'''
extract the integer part
    -- %
    -- np.floor()
    -- np.ceil()
    -- astype()
    -- np.trunc(): 丢弃有符号数x的小数部分
'''
z = np.random.uniform(-10, 11, 10)
z1 = z - z % 1
z2 = np.floor(z)
z3 = np.ceil(z) - 1
z4 = z.astype(int)
z5 = np.trunc(z)
# print(z, z1, z2, z3, z4, z5, sep='\n')

'''
np.linspace()
'''
z = np.linspace(0, 1, 11)
z1 = np.linspace(0, 1, 11, endpoint=False)
z2 = np.linspace(0, 1, 11, endpoint=False)[1:]
# print(z, z1, z2, sep='\n')

'''
np.apply_along_axis(func1d, axis, arr, *args, **kwargs)
'''
b = np.array([[1, 2, 3, 4],
            [5, 6, 7, 8],
            [9, 10, 11, 12]])
output1 = np.apply_along_axis(lambda a: (a[0] + a[-1]) * 0.5, axis=0, arr=b)
output2 = np.apply_along_axis(lambda a: (a[0] + a[-1]) * 0.5, axis=1, arr=b)
print('output1 ({}) = {}\noutput2 ({}) = {}'.format(type(output1), output1, type(output2), output2))

'''
np.linalg.norm(x, ord=None, axis=None, keepdims=False)
x: 表示矩阵，也可以是一维
ord: 范数类型
    --- 向量的范数
        --- 默认:     二范数     np.sqrt(np.power(x1, 2) + np.power(x2, 2) + ... + np.power(xn, 2))
        --- ord=2:  二范数
        --- ord=1:  一范数     |x1| + |x2| + ... + |xn|
    --- 矩阵范数
        --- ord=1:  列和的最大值
        --- ord=2:  |λE - ATA| = 0，求特征值，然后求最大特征值的算术平方根
        --- ord=∞:  行和的最大值
axis：处理类型
    --- axis=1      表示按行向量处理，求多个行向量的范数
    --- axis=0      表示按列向量处理，求多个列向量的范数
    --- axis=None  表示矩阵范数。
'''
x = np.array([
    [0, 3, 4],
    [1, 6, 4]])
print("默认参数(矩阵整体元素平方和开根号，不保留矩阵二维特性)：", np.linalg.norm(x))
print("矩阵整体元素平方和开根号，保留矩阵二维特性：", np.linalg.norm(x, keepdims=True))
print("矩阵每个行向量求向量的2范数：", np.linalg.norm(x, axis=1, keepdims=True))
print("矩阵每个列向量求向量的2范数：", np.linalg.norm(x, axis=0, keepdims=True))
print("矩阵1范数：", np.linalg.norm(x, ord=1, keepdims=True))
print("矩阵2范数：", np.linalg.norm(x, ord=2, keepdims=True))
print("矩阵∞范数：", np.linalg.norm(x, ord=np.inf, keepdims=True))
print("矩阵每个行向量求向量的1范数：", np.linalg.norm(x, ord=1, axis=1, keepdims=True))



