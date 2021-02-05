import numpy as np
import torch
from matplotlib import pyplot as plt


# a = np.arange(25).reshape(5, 5)
# print("a: \n", a)
# print("a[(0,1,2,3,4), (2,4,3,0,0)]: ", a[np.arange(5), np.array([2, 4, 3, 0, 0])])
#
# a = np.array([1, 2, 1, 3, 2, 1, 1])
#
# b = np.where(a > 3, a, 20 - a)
# inx = np.array([0, 1, 5])
# c = np.where(a == 1)
# print("c: ", c)


def cal_ece(prob: np.array):
    return prob.sum()


a = np.array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]])
b = map(cal_ece, a)
# print(list(b))

c = np.array([1, 2, 3, 4, 5, 6])
d = (c>=4).astype(np.float32)
print("d: ", d)

f = [3, 4, 5, 6]
e = []
e.extend(f)
e.extend(f)
print("e: ", e)

# plt.plot()
# plt.plot(np.arange(10), np.random.rand(10))
# plt.savefig("/data.local/all/hangd/v1/uncertainty1/save_plot1.png")

a = torch.arange(6).reshape((2, 3))
print("before: ", a, "shape: ", a.shape)
a = a.unsqueeze(1)
print("after: ", a, "shape: ", a.shape)

a = (a>3).float()
print("a ", a)


a = np.log10(0.001)
b = np.log2(0.001)
print("a: " , a)
print("b: ", b)