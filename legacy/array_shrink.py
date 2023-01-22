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

def array_shrink(data_in, mask, mode='merge'):

    """
    Code to merge the first two dimensions of matrix 'DataIn' into one and remove
    values based on the 2D index 'mask'. The idea here is that DataIn is a stack
    of images with resolution X*Y and pixels in 'mask' should be removed to
    reduce datasize and computional load of subsequent analysis. The first 
    dimension of 'DataOut' will be the product of the X*Y of 'DataIn' minus 
    pixels in 'mask'. 
    Usage: data_out = array_shrink(data_in,mask,'merge')

    To re-assemble the stack after computations have been done, the code
    can be called with the additional argument 'mode' set to 'split'. This
    will reconstruct the original data structure removed pixels will be
    replaced by NaNs.
    Usage: data_out = array_shrink(data_in,mask,'split')
    
    Originally written in MATLAB by Simon Musal, 2016
    
    Adapted to Python by Michael Sokoletsky, 2021

    """

    d_size = np.shape(data_in) # size of input matrix
    if d_size[0] == 1:
        data_in = np.squeeze(data_in) # remove singleton dimensions
        d_size = np.shape(data_in)

    if len(d_size) == 2:
        if d_size[0] == 1:
            data_in = data_in.T
            d_size = np.shape(data_in) # size of input matrix

        d_size = d_size + (1,)

    if mode == 'merge': # merge x and y dimension into one
    
        data_in = np.reshape(data_in,(np.size(mask),np.prod(d_size[mask.ndim:]))) # merge x and y dimension based on mask size and remaining dimensions.
        mask = mask.flatten() # reshape mask to vector
        data_in = data_in[~mask,:]
        orig_size = [np.size(data_in,0),*d_size[2:]]
        data_in = np.reshape(data_in,tuple(orig_size))
        data_out =  data_in

    elif mode == 'split': # split first dimension into x- and y- dimension based on mask size

        # check if datatype is single. If not will use double as a default.
        if data_in.dtype == 'float32':
            d_type = 'float32'
        else:
            d_type = 'float64'

        m_size = np.shape(mask)
        mask = mask.flatten() # reshape mask to vector
        curr_size = [np.size(mask), *d_size[1:]]
        data_out = np.full(tuple(curr_size),np.nan,dtype=d_type) # pre-allocate new matrix
        data_out = np.reshape(data_out,(np.size(data_out,0),-1))
        data_out[~mask,:] = np.reshape(data_in,(np.sum(~mask),-1))
        orig_size = [*m_size, *d_size[1:]]
        data_out = np.squeeze(np.reshape(data_out,tuple(orig_size)))

    return data_out