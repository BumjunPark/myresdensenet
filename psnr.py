
# coding: utf-8

# In[ ]:

import numpy
import math

def psnr(img1, img2, max_val):
    mse = numpy.mean( (img1 - img2) ** 2 )
    if mse == 0:
        return 100
    PIXEL_MAX = max_val
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))

