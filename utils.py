
import pickle
import os
import numpy as np
from timeit import default_timer as timer
import multiprocessing as mp
from multiprocessing import Pool
from timeit import default_timer as timer
import time
from time import sleep

##############################################
# DATA OPS
##############################################

def get_pickle_filename(filename):
    filename, file_extension = os.path.splitext(filename)
    return ''.join((filename, '.pkl')) if (file_extension == '' or file_extension != 'pickle' or file_extension != 'pkl') else ''.join((filename, file_extension))


def load_from_pickle(filename):
    return pickle.load(open(get_pickle_filename(filename), 'rb')) if os.path.exists(get_pickle_filename(filename)) else None


def save_as_pickle(filename, data):
    pickle.dump(data, open(get_pickle_filename(filename), 'wb'), protocol = 4)


def concatenate_list_of_dicts(list_of_dicts):

    concat_dict = {k: np.concatenate([d[k] for d in list_of_dicts]) for k in list_of_dicts[0].keys()}

    return concat_dict
    

def check_load(filename, fn_execute, check_obj, save_to_file=True, args_in=None, verbose=False):

    """
    Function that checks whether filename['check_obj'] is the same as the 
    check_obj passed into the function and if it is, then the file from disk
    is returned. Otherwise fn_execute is executed and returned.
    """
    
    if os.path.exists(filename):
        try:
            starttime = timer()
            wrapper = load_from_pickle(filename)

            if wrapper is not None:
                if objects_equal(wrapper['check_obj'], check_obj):
                    if verbose: print(' '.join(('Loaded ', filename, ' from disk. Time elapsed: ', str(timer() - starttime), 'secs.')))
                    return wrapper['data']
        except:
            if verbose: print(' '.join(('Load failed: ', filename)))

    starttime = timer()
    data = fn_execute() if args_in is None else fn_execute(*args_in)
    wrapper = {'check_obj': check_obj, 'data': data}
    if verbose: print(' '.join(('Time elapsed: ', str(timer() - starttime), 'secs.')))

    if filename is not None and save_to_file:
        starttime = timer()
        try:
            save_dir, _ = os.path.split(filename)
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            save_as_pickle(filename, wrapper)
            if verbose: print(' '.join(('Saved ', filename, '. Time elapsed: ', str(timer() - starttime), 'secs.')))
        except:
            if verbose: print(' '.join(('Save of file', filename, 'failed.')))

    return data


def objects_equal(obj1, obj2):
    if isinstance(obj1, dict) and isinstance(obj2, dict):
        return dict_difference(obj1, obj2)
    elif is_numpy(obj1) and is_numpy(obj2):
        return np.array_equal(obj1, obj2)
    else:
        try:
            eq = obj1 == obj2
        except:
            eq = False
        return eq


def is_numpy(obj):
    return type(obj).__module__ == np.__name__


def dict_difference(dict1, dict2):
    if sorted(dict1.keys()) == sorted(dict2.keys()):
        return all(dict_difference(dict1[k], v) if isinstance(v, dict) else np.array_equal(dict1[k], v) if is_numpy(v) else dict1[k] == v for k, v in dict2.items())
    else:
        return False


def execute_mp_map(fn_target, arg_list, fn_init = None, init_args = None, boo_async = False, display_iter = 1, num_parallel_cores = 8):
    """
    Function which executes fn_target over multiple processes using map (if boo_async = False) or map_async (if boo_async = True)
    """
   
    num_batches = len(arg_list)

    # Initiate processes
    if fn_init is not None:
        p = Pool(processes = min(num_batches, num_parallel_cores), initializer = fn_init, initargs = init_args)
    else:
        p = Pool(processes = min(num_batches, num_parallel_cores))

    if boo_async:

        this_process_set = p.map_async(fn_target, arg_list, chunksize = 1) # include chunksize = 1 so that you get accurate progress reports below, otherwise _number_left refers to the number of chunks left (after arg_list has been split into chunks)

        # Wait for the processes to complete, updating on progress as you go
        last_time = timer()
        num_displayed = 0
        while (True):
            if (this_process_set.ready()): break
            if (np.mod(num_batches - this_process_set._number_left, display_iter) == 0) and ((num_batches - this_process_set._number_left) != num_displayed):
                num_displayed = num_batches - this_process_set._number_left
                this_time = timer()
                print(''.join(('execute_mp_map> ', fn_target.__name__, ' completed batch ', str(num_batches - this_process_set._number_left),' / ', str(num_batches), '... \n Time elapsed: ', str(this_time - last_time), ' secs')))
                last_time = this_time
            sleep(1)
        
        # Estimation has completed, get the results
        results = this_process_set.get()

    else:

        results = p.map(fn_target, arg_list, chunksize = 1) 

    # Clean up
    p.close()
    p.join()

    return results



if __name__ == '__main__':

    pass
