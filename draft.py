import numpy as np
import torch
from os.path import splitext
from matplotlib import pyplot as plt
from os import listdir
from glob import glob

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


# def cal_ece(prob: np.array):
#     return prob.sum()


# a = np.array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]])
# b = map(cal_ece, a)
# # print(list(b))

# c = np.array([1, 2, 3, 4, 5, 6])
# d = (c>=4).astype(np.float32)
# print("d: ", d)

# f = [3, 4, 5, 6]
# e = []
# e.extend(f)
# e.extend(f)
# print("e: ", e)

# # plt.plot()
# # plt.plot(np.arange(10), np.random.rand(10))
# # plt.savefig("/data.local/all/hangd/v1/uncertainty1/save_plot1.png")

# a = torch.arange(6).reshape((2, 3))
# # print("before: ", a, "shape: ", a.shape)
# a = a.unsqueeze(1)
# # print("after: ", a, "shape: ", a.shape)

# a = (a>3).float()
# # print("a ", a)


# a = np.log10(0.001)
# b = np.log2(0.001)
# print("a: " , a)
# print("b: ", b)

# imgs_dir = "/data.local/all/hangd/dynamic_data/imgs/"
# [splitext(file)[0] for file in listdir(imgs_dir)
#                     if not file.startswith('.')]
#
# for file in listdir(imgs_dir):
#     print(splitext(file))
#     print("file: ", file)
#     break

# print("1236")
# mask = glob("/data.local/all/hangd/dynamic_data/one32rd/masks/From40Frs__DOAN 9-10-2018__2C__IMG-0055-00001.dcm_11.*")
#
# print(mask)
from collections import Counter
# import numpy as np
#
# z= np.linspace(-5,5,10)
# print(z)
# # z = x**(1/3)
# # print("be4: ", z)
# z = np.where(z<0, -np.abs(z)**(1/3), z**(1/3))
# print("after: ",z )
#
# print(z)
# if __name__ == '__main__':
# n, m = map(int, input().split())
# array = list(map(int, input().split()))
# A = list(map(int, input().split()))
# B = list(map(int, input().split()))
# total_happiness = 0
#
# for i in array:
#     if i in A:
#         total_happiness += 1
#     if i in B:
#         total_happiness -= 1
# print(total_happiness)
#
# # print(sum([(i in A) - (i in B) for i in array]))
#
# print(5 in array)

# n = int(input())
# print("N: ", type(n), n)
# words = []
# for i in range(n):
#     words.append(input())
# print("words: ", words)
# res = dict(Counter(words))
# print("res: ", type(res), res)
# freq = list(res.values())
# print(*freq)

# import re
# 
# s = input()
# print(bool(re.match(r'^[1-9][\d]{5}$', s) and len(re.findall(r'(\d)(?=\d\1)', s)) < 2))
# a = {"ids":["mot", "hai", "ba"]}
# a["ids"].remove("mot")
# print(a)

import numpy as np
import torch
# a = torch.rand((3, 4))
# a = a.sum(1)
# print("shape a: ", a.shape)

std = [1,4,2,6,4,9]
imgs_id = ["one", "four", "two", "six", "four2", "nine"]
std, imgs_id = zip(*sorted(zip(std, imgs_id)))
print(std, imgs_id)
print(imgs_id[-3:])
