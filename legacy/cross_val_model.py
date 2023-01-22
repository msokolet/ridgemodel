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
import copy
from tqdm.notebook import tqdm, trange
from ridge_MML import *

def cross_val_model(full_R, Vc, c_labels, reg_idx, reg_labels, folds):

    '''
    This function computed the cross-validated R^2.
    
    Originally written in MATLAB by Simon Musall, 2019
    
    Adapted to Python by Michael Sokoletsky, 2021
    '''

    labels_idx = [reg_labels.index(c_label) for c_label in c_labels if c_label in reg_labels]
    c_idx = np.isin(reg_idx,labels_idx) # get index for task regressors
    c_labels = [reg_labels[idx] for idx in np.sort(labels_idx)] # make sure c_labels is in the right order
    
    # create new regressor index that matches c labels
    sub_idx = copy.copy(reg_idx)
    sub_idx = sub_idx[c_idx]
    temp = np.unique(sub_idx)
    for x, x_idx in enumerate(temp):
        sub_idx[sub_idx == x] = x_idx

    cR = full_R[:,c_idx]

    Vm = np.zeros_like(Vc,dtype=np.float32) # pre-allocate c-reconstructed V
    rng = np.random.default_rng(1) # for reproducibility
    rand_idx = rng.permutation(np.size(Vc,1)) # generate randum number index
    fold_cnt = np.floor(np.size(Vc,1) / folds).astype(np.uint)
    c_beta = [0]*folds

    for i_folds in tqdm(range(folds), desc = 'Performing cross-validation'):
        data_idx = np.ones(np.size(Vc,1),dtype=np.bool_)
        
        if folds > 1:
            data_idx[rand_idx[(i_folds*fold_cnt) + np.arange(fold_cnt)]] = False # index for training data
            if i_folds == 0:
                c_ridge, c_beta[i_folds], _ = ridge_MML(Vc[:,data_idx].T, cR[data_idx,:], True) #  get beta weights and ridge penalty for task only model
            else:
                c_beta[i_folds] = ridge_MML(Vc[:,data_idx].T, cR[data_idx,:], True, c_ridge) # get beta weights for task only model. ridge value should be the same as in the first run.

            Vm[:,~data_idx] = (cR[~data_idx,:] @ c_beta[i_folds]).T # predict remaining data


        else:
            c_ridge, c_beta[i_folds] = ridge_MML(Vc.T, cR, True) # get beta weights for task-only model.
            Vm = (cR * c_beta[i_folds]).T # predict remaining data
            print('Ridgefold is <= 1, fit to complete dataset instead')

    return Vm, c_beta, cR, sub_idx, c_ridge, c_labels