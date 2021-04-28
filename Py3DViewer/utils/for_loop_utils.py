import multiprocessing as mp
from numba import njit

def parallel_for(func, n_iters, *args):

    num_cpus = mp.cpu_count()
    pool = mp.Pool(num_cpus)
    results = pool.starmap(func, [(*args, i) for i in range(n_iters)])
    
    return results

def numba_for(func, n_iters, *args):
    func = njit(func)
    @njit
    def inner(n_iters, *args):
        result = []
        for i in range(n_iters):
            result.append(func(*args, i))
        return result
     
    return inner(n_iters, *args)