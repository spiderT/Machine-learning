# pytorch

- [pytorch](#pytorch)
  - [1. NumPy](#1-numpy)
    - [1.1. NumPy 数组](#11-numpy-数组)
    - [1.2. 创建数组](#12-创建数组)
    - [1.3. 数组的属性](#13-数组的属性)
      - [1.3.1. ndim](#131-ndim)
      - [1.3.2. shape](#132-shape)
      - [1.3.3. size](#133-size)
      - [1.3.4. dtype](#134-dtype)
    - [1.4. 其他创建数组的方式](#14-其他创建数组的方式)
      - [1.4.1. np.ones() 与 np.zeros()](#141-npones-与-npzeros)
      - [1.4.2. np.arange()](#142-nparange)
      - [1.4.3. np.linspace()](#143-nplinspace)
    - [1.5. 数组的轴](#15-数组的轴)
    - [1.6. 深度学习中的常用操作](#16-深度学习中的常用操作)
      - [1.6.1. 数据加载阶段](#161-数据加载阶段)

## 1. NumPy

NumPy 是用于 Python 中科学计算的一个基础包。它提供了一个多维度的数组对象，以及针对数组对象的各种快速操作，例如排序、变换，选择等。可以使用 Conda 安装，命令：conda install numpy  或使用 pip 进行安装，命令：pip install numpy

### 1.1. NumPy 数组

数组对象是 NumPy 中最核心的组成部分，这个数组叫做 ndarray，是“N-dimensional array”的缩写。其中的 N 是一个数字，指代维度，在 NumPy 中，数组是由 numpy.ndarray 类来实现的，它是 NumPy 的核心数据结构。

NumPy 数组的特点

1. Python 中的列表可以动态地改变，而 NumPy 数组是不可以的，它在创建时就有固定大小了。改变 Numpy 数组长度的话，会新创建一个新的数组并且删除原数组。
2. NumPy 数组中的数据类型必须是一样的，而列表中的元素可以是多样的。
3. NumPy 针对 NumPy 数组一系列的运算进行了优化，使得其速度特别快，并且相对于 Python 中的列表，同等操作只需使用更少的内存。

### 1.2. 创建数组

最简单的方法就是把一个列表传入到 np.array() 或 np.asarray() 中，这个列表可以是任意维度的。np.array() 属于深拷贝，np.asarray() 则是浅拷贝。

```py
import numpy as np

arr_1_d = np.asarray([1])
print(arr_1_d)  # [1]

arr_2_d = np.asarray([[1, 2], [3, 4]])
print(arr_2_d)  # [[1 2] [3 4]]
```

### 1.3. 数组的属性

数组维度、形状、size 与数据类型。

#### 1.3.1. ndim

ndim 表示数组维度（或轴）的个数。刚才创建的数组 arr_1_d 的轴的个数就是 1，arr_2_d 的轴的个数就是 2。

```py
print(arr_1_d.ndim) # 1
print(arr_2_d.ndim) # 2
```

#### 1.3.2. shape

shape 表示数组的维度或形状， 是一个整数的元组，元组的长度等于 ndim。

arr_1_d 的形状就是（1，）（一个向量）， arr_2_d 的形状就是 (2, 2)（一个矩阵）。

```py
print(arr_1_d.shape) # (1,)
print(arr_2_d.shape) # (2, 2)
```

对数组的形状进行变换，就可以使用 arr.reshape() 函数，在不改变数组元素内容的情况下变换数组的形状。注意的是，**变换前与变换后数组的元素个数需要是一样的。**

```py
# 将arr_2_d reshape为(4，1)的数组
>>>arr_2_d.reshape((4,1))
array([[1],
       [2],
       [3],
       [4]])
```

还可以使用 np.reshape(a, newshape, order) 对数组 a 进行 reshape，新的形状在 newshape 中指定。

order 参数，它是指以什么样的顺序读写元素，其中有这样几个参数。

- ‘C’：默认参数，使用类似 C-like 语言（行优先）中的索引方式进行读写。
- ‘F’：使用类似 Fortran-like 语言（列优先）中的索引方式进行读写。
- ‘A’：原数组如果是按照‘C’的方式存储数组，则用‘C’的索引对数组进行 reshape，否则使用’F’的索引方式。

#### 1.3.3. size

size，也就是数组元素的总数，它就等于 shape 属性中元素的乘积。

```py
print(arr_2_d.size) # 4
```

#### 1.3.4. dtype

它是一个描述数组中元素类型的对象。使用 dtype 属性可以查看数组所属的数据类型。

NumPy 中大部分常见的数据类型都是支持的，例如 int8、int16、int32、float32、float64 等。dtype 是一个常见的属性，在创建数组，数据类型转换时都可以看到它。

```py
print(arr_2_d.dtype) # int64
```

如果没有指定数据类型，NumPy 会自动进行判断，然后给一个默认的数据类型。

```py
arr_3_d = np.asarray([[1, 2], [3, 4]], dtype="float")
print(arr_3_d.dtype) # float64
```

数组的数据类型可以改变，使用 astype() 改变数组的数据类型，不过改变数据类型会创建一个新的数组，而不是改变原数组的数据类型。

```py
arr_3_d_int = arr_3_d.astype('int32')
print(arr_3_d.dtype) # float64
print(arr_3_d_int.dtype) # int32
```

**不能通过直接修改数据类型来修改数组的数据类型**，这样代码虽然不会报错，但是数据会发生改变.

```py
print(arr_3_d) # [[ 1.  2.] [ 3.  4.]]
```

### 1.4. 其他创建数组的方式

#### 1.4.1. np.ones() 与 np.zeros()

np.ones() 用来创建一个全 1 的数组，必须参数是指定数组的形状，可选参数是数组的数据类型。  

```py
print(np.ones(shape=(2, 3)))  # [[ 1.  1.  1.][ 1.  1.  1.]]
print(np.ones(shape=(2, 3), dtype="int32"))  # [[1 1 1][1 1 1]]
```

创建全 0 的数组是 np.zeros()，用法与 np.ones() 类似.

这两个函数一般什么时候用呢？例如，如果需要初始化一些权重的时候就可以用上，比如说生成一个 2x3 维的数组，每个数值都是 0.5，可以这样做。

```py
print(np.ones((2, 3)) * 0.5)
# [[ 0.5  0.5  0.5]
#  [ 0.5  0.5  0.5]]
```

#### 1.4.2. np.arange()

np.arange([start, ]stop, [step, ]dtype=None) 创建一个在[start, stop) 区间的数组，元素之间的跨度是 step。

start 是可选参数，默认为 0。stop 是必须参数，区间的终点，**区间是一个左闭右开区间**，所以数组并不包含 stop。step 是可选参数，默认是 1。

```py
print(np.arange(5)) # [0 1 2 3 4]
print(np.arange(2, 5)) # [2 3 4]
print(np.arange(2, 9, 3)) # [2 5 8]
```

#### 1.4.3. np.linspace()

np.linspace（start, stop, num=50, endpoint=True, retstep=False, dtype=None）创建一个从开始数值到结束数值的等差数列。

- start：必须参数，序列的起始值。
- stop：必须参数，序列的终点。
- num：序列中元素的个数，默认是 50。
- endpoint：默认为 True，如果为 True，则数组最后一个元素是 stop。
- retstep：默认为 False，如果为 True，则返回数组与公差。# 从2到10，有3个元素的等差数列>>>np.linspace(start=2, stop=10, num=3)

```py
print(np.linspace(start=2, stop=10, num=3)) # [  2.   6.  10.]
```

np.arange 与 np.linspace 也是比较常见的函数，比如要作图的时候，可以用它们生成 x 轴的坐标。例如，生成一个 y=x2 的图片，x 轴可以用 np.linespace() 来生成。

```py
import numpy as np
import matplotlib.pyplot as plt

X = np.arange(-50, 51, 2)
Y = X ** 2

plt.plot(X, Y, color='blue')
plt.legend()
plt.show()
```

### 1.5. 数组的轴

经常出现在 np.sum()、np.max() 这样关键的聚合函数中。

数组的轴即数组的维度，它是从 0 开始的。对于我们这个二维数组来说，有两个轴，分别是代表行的 0 轴与代表列的 1 轴。

```text

>>>interest_score = np.random.randint(10, size=(4, 3))
>>>interest_score
array([[4, 7, 5],
       [4, 2, 5],
       [7, 2, 4],
       [1, 2, 4]])
```

![轴](./images/pytorch_1.png)

多维数据, 当 axis=i 时，就是按照第 i 个轴的方向进行计算的，或者可以理解为第 i 个轴的数据将会被折叠或聚合到一起。

形状为 (a, b, c) 的数组，沿着 0 轴聚合后，形状变为 (b, c)；沿着 1 轴聚合后，形状变为 (a, c)；沿着 2 轴聚合后，形状变为 (a, b)；更高维数组以此类推。  

```py
a = np.arange(18).reshape(3, 2, 3)
print(a)
# [[[ 0  1  2]
#   [ 3  4  5]]

#  [[ 6  7  8]
#   [ 9 10 11]]

#  [[12 13 14]
#   [15 16 17]]]
```

将同一个轴上的数据看做同一个单位，那聚合的时候，只需要在同级别的单位上进行聚合就可以了。  

绿框代表沿着 0 轴方向的单位，蓝框代表着沿着 1 轴方向的单位，红框代表着 2 轴方向的单位。

![轴](./images/pytorch_2.png)

当 axis=0 时，就意味着将三个绿框的数据聚合在一起，结果是一个（2，3）的数组，数组内容为：  
[ [(max(a000​,a100​,a200​),max(a001​,a101​,a201​),max(a002​,a102​,a202​))],  
[(max(a010​,a110​,a210​),max(a011​,a111​,a211​),max(a012​,a112​,a212​))] ]​

```py
b = a.max(axis=0)
print(b)

# [[12 13 14]
#  [15 16 17]]
```

当 axis=1 时，就意味着每个绿框内的蓝框聚合在一起，结果是一个（3，3）的数组，数组内容为：  
[ [(max(a000​,a010​),max(a001​,a011​),max(a002​,a012​))],  
[(max(a100​,a110​),max(a101​,a111​),max(a102​,a112​))],  
[(max(a200​,a210​),max(a201​,a211​),max(a202​,a212​))], ]​  

```py
c = a.max(axis=1)
print(c)

# [[ 3  4  5]
#  [ 9 10 11]
#  [15 16 17]]
```

当 axis=2 时，就意味着每个绿框中的红框聚合在一起，结果是一个（3，2）的数组，数组内容如下所示：  
[ [(max(a000​,a001​,a002​),max(a010​,a011​,a012​))],  
[(max(a100​,a101​,a102​),max(a110​,a111​,a112​))],  
[(max(a200​,a201​,a202​),max(a210​,a211​,a212​))], ]​

```py
d = a.max(axis=2)
print(d)

# [[ 2  5]
#  [ 8 11]
#  [14 17]]
```

### 1.6. 深度学习中的常用操作

解决图片分类问题，可以分解成数据加载、训练与模型评估三部分）。其中数据加载跟模型评估中，就经常会用到 NumPy 数组的相关操作。

#### 1.6.1. 数据加载阶段

对于图片的处理，一般会使用 Pillow 与 OpenCV 这两个模块。虽然 Pillow 和 OpenCV 功能看上去都差不多，但还是有区别的。在 PyTorch 中，很多图片的操作都是基于 Pillow 的，所以当使用 PyTorch 编程出现问题，或者要思考、解决一些图片相关问题时，要从 Pillow 的角度出发。

1. Pillow 方式

```py

from PIL import Image
im = Image.open('jk.jpg')
im.size
# 输出: 318, 116
```

利用 NumPy 的 asarray 方法，就可以将 Pillow 的数据转换为 NumPy 的数组格式。

```py
import numpy as np

im_pillow = np.asarray(im)

im_pillow.shape
# 输出：(116, 318, 3)
```

2. OpenCV 方式

直接读入图片后，就是以 NumPy 数组的形式来保存数据的

```py

import cv2
im_cv2 = cv2.imread('jk.jpg')
type(im_cv2)
# 输出：numpy.ndarray

im_cv2.shape
# 输出：(116, 318, 3)
```




