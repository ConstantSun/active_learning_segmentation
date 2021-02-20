# from itertools import product
#
# K, M = map(int, input().split())
# N = (list(map(int, input().split()))[1:] for _ in range(K))
# # print("N: ", *N)
#
#
# results = map(lambda x: sum(i**2 for i in x)%M, product(*N))
# results = map(lambda x: sum(i ** 2 for i in x) % M, product(*N))
# print(max(results))


import math
import os
import random
import re
import sys
import numpy as np

first_multiple_input = input().rstrip().split()

n = int(first_multiple_input[0])

m = int(first_multiple_input[1])

matrix = []

for _ in range(n):
    matrix_item = input()
    matrix.append(matrix_item)

print("matrix: ", matrix)
temp = []
for i in matrix:
    t = []
    for j in i :
        t.append(j)
    temp.append(t)
matrix = temp


matrix = np.array(matrix)
decode = matrix.transpose()
print("decode: ", decode)
# decoded_script =
