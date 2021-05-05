import multiprocessing as mp
from numba import njit

def parallel_for(func, n_iters, *args):
    """
    Implementation of a parallel for loop that can execute a generic function. The signature of the function must be:
    def func_name(param_1,..,param_n, i)

    Parameters:
    
        func : The function that will be executed in the loop
        n_iters (int): Number of iterations
        *args : The parameters of the function
        
    Return:
         List : List with the computation results
    
    """

    num_cpus = mp.cpu_count()
    pool = mp.Pool(num_cpus)
    results = pool.starmap(func, [(*args, i) for i in range(n_iters)])
    
    return results

def numba_for(func, n_iters, *args):
    """
    Implementation of a fast for loop that uses numba as backend. The loop can execute a generic function. The signature of the function must be:
    def func_name(param_1,..,param_n, i)

    Parameters:
    
        func : The function that will be executed in the loop
        n_iters (int): Number of iterations
        *args : The parameters of the function
        
    Return:
         List : List with the computation results
    
    """
    func = njit(func)
    @njit
    def inner(n_iters, *args):
        result = []
        for i in range(n_iters):
            result.append(func(*args, i))
        return result
     
    return inner(n_iters, *args)