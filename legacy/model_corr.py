# MIT License

# Copyright (c) 2019 Churchland laboratory

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import numpy as np
import math
from tqdm import tqdm
from array_shrink import *

def model_corr(Vc,Vm,U):
    
    """
    short code to compute the correlation between lowD data Vc and modeled
    lowD data Vm. Vc and Vm are temporal components, U is the spatial
    components. corr_mat is a the correlation between Vc and Vm in each pixel.
    
    Originally written in MATLAB by Simon Musall, 2019
    
    Adapted to Python by Michael Sokoletsky, 2021
    """
    
    Vc = np.reshape(Vc,(np.size(Vc,0),-1))
    Vm = np.reshape(Vm,(np.size(Vm,0),-1))
    if len(np.shape(U)) == 3:
        U = array_shrink(U, np.isnan(U[:,:,0]).squeeze())

    cov_Vc = np.cov(Vc) # S x S
    cov_Vm = np.cov(Vm) # % S x S
    c_cov_V = (Vm - np.expand_dims(np.mean(Vm,1), axis=1)) @ Vc.T / (np.size(Vc, 1) - 1) # S x S
    cov_P = np.expand_dims(np.sum((U @ c_cov_V) * U, 1),axis=0) #  1 x P
    var_P1 = np.expand_dims(np.sum((U @ cov_Vc) * U, 1),axis=0) # 1 x Pii
    var_P2 = np.expand_dims(np.sum((U @ cov_Vm) * U, 1),axis=0) # 1 x P
    std_Px_Py = var_P1 ** 0.5 * var_P2 ** 0.5 # 1 x P
    corr_mat = (cov_P / std_Px_Py).T

    return corr_mat, var_P1, var_P2