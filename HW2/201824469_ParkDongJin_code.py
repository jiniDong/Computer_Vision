from PIL import Image
import numpy as np
import math

def boxfilter(n):
    if n <= 0 or n % 2 == 0: # n이 조건에 맞지 않을 경우 Assertion Erorr를 발생시킵니다.
        raise AssertionError('Dimension must be odd')
    
    box_entry = 1.0 / n**2
    filter = np.full((n, n), box_entry)
    return filter

# print(boxfilter(3))
# print(boxfilter(4))
x = boxfilter(7)
print(x)