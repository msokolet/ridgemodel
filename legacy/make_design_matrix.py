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
from tqdm.notebook import tqdm, trange

def make_design_matrix(events, event_types, opts):
    ''' 
    This function generates a design matrix from a column matrix with binaryevents. 
    event_types defines the type of design matrix that is generated.
    (1 = full trial, 2 = post-event, 3 = peri-event)

    Originally written in MATLAB by Simon Musall, 2019
    
    Adapted to Python by Michael Sokoletsky, 2021
    
    '''
    frames = opts['frames_per_trial']
    full_mat = [None] * len(event_types)
    event_idx = [None] * len(event_types)
    events = events.reshape((-1, frames, len(event_types))) # reshape to trials
    trial_cnt = np.size(events,0) # nr of trials

    for i_reg, event_type in tqdm(enumerate(event_types), total=len(event_types),
                              desc = 'Building design matrix'):

        # determine index for current event type
        if event_type == 1:
            kernel_idx = np.arange(frames) # index for whole trial        
        elif event_type == 2:
            kernel_idx = np.arange(opts['s_post_time']+1) # index for design matrix to cover post event activity
        elif event_type == 3:
            kernel_idx = np.arange(-opts['m_pre_time'],opts['m_post_time']+1)
        else:
            print('Unknown event type. Must be a value between 1 and 3.')

        # run over trials
        d_mat = [None] * trial_cnt

        for i_trial in range(trial_cnt):

            # get the zero lag regressor.
            trace = events[i_trial,:,i_reg].astype(bool)

            # create full design matrix
            c_idx = np.where(trace)+kernel_idx[:,np.newaxis]
            c_idx = np.clip(c_idx,-1,frames-1)
            c_idx = c_idx + np.arange(0,frames*len(kernel_idx),frames)[:,np.newaxis]

            c_idx[c_idx < 0] = frames-1
            c_idx[c_idx > (frames*len(kernel_idx) - 1)] = frames*len(kernel_idx) - 1

            d_mat[i_trial] = np.zeros((frames,len(kernel_idx)))

            d_mat[i_trial][c_idx % frames,c_idx // frames] = True
            
            d_mat[i_trial][-1,:] = False #  don't use last timepoint of design matrix to avoid confusion with indexing.
            d_mat[i_trial][-1,1:] = d_mat[i_trial][-2,:-1] #  replace with shifted version of previous timepoint

        full_mat[i_reg] = np.vstack(d_mat) # combine all trials
        c_idx = np.sum(full_mat[i_reg],0) > 0 # don't use empty regressors
        full_mat[i_reg] = full_mat[i_reg][:,c_idx]
        event_idx[i_reg] = np.zeros(sum(c_idx))+i_reg        
    
    full_mat = np.hstack(full_mat) # combine all regressors into larger matrix
    event_idx = np.concatenate(event_idx) #  combine index so we know what is what

    return full_mat, event_idx