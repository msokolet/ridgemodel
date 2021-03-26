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

from .utils import *
from .design import make_design_matrix
from .utils import calc_regressor_orthogonality, cross_val_model, model_corr
from .io import load_stack
from .plots import plot_regressor_orthogonality, plot_model_corr

import argparse
import sys

class CLIParser(object):
    def __init__(self):
        parser = argparse.ArgumentParser(
            description='ridgemodel - performs ridge regression on widefield imaging data',
            usage='''ridgemodel <command> [<args>]

The commands are:
    process             Performs ridge regression on widefield imaging data using events as regressors
    design              Builds a design matrix from events (output: design.npz)
    cross_val           Performs cross-validated ridge-regression on widefield imaging data (output: (reg)-m.npz, for each regressor)
''')
        parser.add_argument('command', help='type ridgemodel <command> -h for help')

        args = parser.parse_args(sys.argv[1:2])
        if not hasattr(self, args.command): # This is not Pythonic - EAFP not LBYL
            print('Command {0} not recognized.'.format(args.command))
            parser.print_help()
            exit(1) 
        getattr(self, args.command)()
        
        
    def process(self):    
        parser = argparse.ArgumentParser(
        description='Performs ridge regression on widefield imaging data using events as regressors')
        parser.add_argument('foldername', action='store',
                    default=None, type=str,
                    help='Folder where to search for events, trial onsets, options, and imaging files (U and STV) files')  
        parser.add_argument('-r', '--regressors', nargs='+', action='store',
                    default=['full'], type=str,
                    help='Regressors or regressor categories to use. \'full\' will use all regressors, \'task\' will use only task regressors (event IDs 1 and 2), and \'move\' will use only movement regressors (event ID 3)')
        parser.add_argument('--remove_redundant', action='store_true',
                    default=True, help='Automatically remove any redundant regressors.')   

        args = parser.parse_args(sys.argv[2:])                     
        localdisk = args.foldername
        remove_redundant = args.remove_redundant
        regressors = args.regressors  
        
        if localdisk is None:
            print('Specify a fast local disk.')
            exit(1)
        if not os.path.isdir(localdisk):
            os.makedirs(localdisk)
            print(f'Created {localdisk}')       
               
        _design(localdisk, remove_redundant) # build design matrix
        
        _cross_val(localdisk, regressors) # perform cross-validation
        
    def design(self):     
        parser = argparse.ArgumentParser(
        description='Builds a design matrix out of events')
        parser.add_argument('foldername', action='store',
                            default=None, type=str,
                            help='Folder where to search for events, trial onsets, and options files')         
        parser.add_argument('--remove_redundant', action='store_true',
                            default=True, help='Automatically remove any redundant regressors.')   
                            
        args = parser.parse_args(sys.argv[2:])                     
        localdisk = args.foldername
        remove_redundant = args.remove_redundant

        if localdisk is None:
            print('Specify a fast local disk.')
            exit(1)
        if not os.path.isdir(localdisk):
            os.makedirs(localdisk)
            print(f'Created {localdisk}')       
            
        _design(localdisk, remove_redundant)                    
                            
    def cross_val(self):     
        parser = argparse.ArgumentParser(
        description='Performs cross-validated ridge regression')
        parser.add_argument('foldername', action='store',
                    default=None, type=str,
                    help='Folder where to search for design matrix, options, and imaging files (U and STV)')     
        parser.add_argument('-r', '--regressors', nargs='+', action='store',
                    default=['full'], type=str,
                    help='Regressors or regressor categories to use. \'full\' will use all regressors, \'task\' will use only task regressors (event IDs 1 and 2), and \'move\' will use only movement regressors (event ID 3)')

        args = parser.parse_args(sys.argv[2:])
        localdisk = args.foldername
        regressors = args.regressors
                    
        if localdisk is None:
            print('Specify a fast local disk.')
            exit(1)
        if not os.path.isdir(localdisk):
            os.makedirs(localdisk)
            print(f'Created {localdisk}')          
            
        _cross_val(localdisk, regressors)                    

                                  
def _cross_val(localdisk, regressors):
    
    fname = pjoin(localdisk,'design.npz')
    if os.path.isfile(fname):                               
        with np.load(fname) as design_f: # load design matrix, event IDs, event labels, and event types
            full_R = design_f['full_R']
            event_idx = design_f['event_idx']
            event_labels = design_f['event_labels']
            event_types = design_f['event_types']
    else:
        raise OSError('Could not find design.npz')   
    
    fname=pjoin(localdisk,'opts.json')                            
    if os.path.isfile(fname):                        
        with open(fname, 'r') as opts_f:
            opts = json.load(opts_f) # load some options
    else:
        raise OSError('Could not find opts.json')
                            
    r_stack = load_stack(localdisk) # load image stock                      
    
    for regressor in regressors:                        
    
        if regressor == 'full':
            labels = event_labels
        elif regressor == 'task':
            labels = event_labels(np.bitwise_or(event_types == 1, event_types == 2))
        elif regressor == 'move':
            labels = event_labels(event_types == 3)
        else:
            if regressor in event_labels:
                labels = regressor
            else:
                raise ValueError(f'Could not find regressor {regressor}')
                            
        [m_stack, beta, _, idx, ridge, labels] = cross_val_model(full_R, r_stack, labels, event_idx, event_labels, opts['n_folds'])
        
        # calculate correlation            
        cvR2 = model_corr(r_stack, m_stack)[0] ** 2
                            
        np.savez(pjoin(localdisk, f'{regressor}_m'), U=m_stack.U, SVT=m_stack.SVT, beta=beta, full_R=full_R, idx=idx, ridge=ridge, labels=labels, cvR2=cvR2) # save the results
                            
        # output pdf of correlation
        plot_model_corr(cvR2, regressor, localdisk)
                            
def _design(localdisk, rmv = True):
    
    fname=pjoin(localdisk,'events.npy')
    if os.path.isfile(fname):                        
        events_f = np.load(fname, allow_pickle=True)
        event_frames = events_f['iframes'] 
        event_types = events_f['type']
        event_labels = events_f['label']        
        
    else:
        raise OSError('Could not find events.npy')      
    
    fname = pjoin(localdisk,'SVTcorr.npy')
    if os.path.isfile(fname):                        
        SVT = np.load(fname,mmap_mode='r')
        frames = np.size(SVT,1)
    else:
        raise OSError('Could not find SVTcorr.npy')     
        
    fname=pjoin(localdisk,'trial_onsets.npy')
    if os.path.isfile(fname):                        
        trial_onsets = np.load(fname)['iframe'] # load trial onsets
        trial_onsets = np.append(trial_onsets, frames) # add last frame
    else:
        raise OSError('Could not find trial_onsets.npy')    
        
    fname=pjoin(localdisk,'opts.json')                            
    if os.path.isfile(fname):                        
        with open(fname, 'r') as opts_f:
            opts = json.load(opts_f) # load some options
    else:
        raise OSError('Could not find opts.json')      
                            

    # make design matrix
    full_R, event_idx = make_design_matrix(event_frames, event_types, trial_onsets, opts) # make design matrix for events
                            
    # calculate regressor orthogonality
    full_QRR, full_R, event_idx = calc_regressor_orthogonality(full_R, event_idx, rmv)         
                            
    # plot regressor orthogonality    
    plot_regressor_orthogonality(full_QRR, localdisk)
    
    # save design matrix and event labels
    np.savez(pjoin(localdisk, 'design'), full_R=full_R, event_idx=event_idx, event_labels=event_labels, event_types = event_types, full_QRR=full_QRR) # save design matrix and event labels
                                           
             
def main():
    CLIParser()

if __name__ == '__main__':
    main()
