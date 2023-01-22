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

import h5py

from .utils import *

def load_stack(localdisk):

    fname = pjoin(localdisk,'SVTcorr_wfield.npy')
    if os.path.isfile(fname):
        SVT = np.load(fname) # load adjusted temporal components
    else:
        raise OSError(f'Could not find: {fname}')

    fname = pjoin(localdisk,'U_atlas_wfield.npy')
    if os.path.isfile(fname):
        U = np.load(fname) # load aligned spatial components
    else:
        fname = pjoin(localdisk,'U_wfield.npy')
        if os.path.isfile(fname):
            U = np.load(fname) # If no aligned spatial components, load regular spatial components
        else:
            raise OSError(f'Could not find: {fname}')

    return SVDStack(U,SVT)


def load_opts(localdisk):

    opts_fname = pjoin(localdisk,'opts.json')
    if os.path.isfile(opts_fname):
        with open(opts_fname, 'r') as opts_f:
            opts = json.load(opts_f) # Load some options. 
    else:
        raise OSError(f'Could not find: {opts_fname}')
            
    return opts

def load_events(localdisk):

    events_fname = pjoin(localdisk,'events.npz')
    if os.path.isfile(events_fname):
        events = np.load(events_fname, allow_pickle=True)['events']
    else:
        opts_fname = pjoin(localdisk,'events.npy') # If no npz file, load npy
        if os.path.isfile(events_fname):
            events = np.load(events_fname, allow_pickle=True)
        else:
            raise OSError(f'Could not find: {opts_fname}')

    return events

