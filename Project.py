import pyedflib
import numpy as np
import tensorly as tl
from tensorly.decomposition import tucker
import time
import multiprocessing as mp
import os
import csv

'''
i = 3
j = str(i)
l = len(j)
n = "0" * (5 - l)
fn = n + j

os.system("./my_script.sh shhs/polysomnography/edfs/shhs1/shhs1-2" + fn + ".edf")

time.sleep(10)
'''


def num_to_text(x):
    j = str(x)
    l = len(j)
    n = "0" * (5 - l)
    return n + j


def pool_function(i):
    basename = "./shhs/polysomnography/edfs/shhs1/shhs1-2"
    fn = num_to_text(i)
    filename = basename + fn + ".edf"
    os.system("./my_script.sh shhs/polysomnography/edfs/shhs1/shhs1-2" + fn + ".edf")

    f = None
    bruh = 0
    while f is None:
        try:
            f = pyedflib.EdfReader(filename)
        except:
            time.sleep(0.1)
            bruh = bruh + 0.1
            if bruh > 10:
                #print(filename)
                os.system("./my_script.sh shhs/polysomnography/edfs/shhs1/shhs1-2" + fn + ".edf")
            pass

    n = f.signals_in_file
    N = max(f.getNSamples()) * 2
    sigbufs = np.zeros((n, N))
    for j in np.arange(n):
        sigbufs[j, :] = np.repeat(f.readSignal(j), N / f.getNSamples()[j])

    [g, h] = sigbufs.shape
    for j in range(3):
        if h % 10 == 0:
            a = 10
        else:
            a = h % 10
        g = g * a
        h = h / a

    g = int(g)
    h = int(h)

    sigbufs = sigbufs.reshape((g, h))
    X = tl.tensor(sigbufs, dtype='float32')
    tucker_rank = [100, 100]
    core, tucker_factors = tucker(X, ranks=tucker_rank, init='random', tol=10e-4)
    return np.array(core).flatten()


def collect_result(result):
    global results
    #results[result[0] - 1, :] = result[1].flatten()
    results.append(result[0])


if __name__ == '__main__':
    print(mp.cpu_count())

    result = np.zeros(shape=[5000, 10000])
    basename = "./shhs/polysomnography/edfs/shhs1/shhs1-2"

    for i in range(0, 5000, 10):
        pool = mp.Pool(processes=3)
        result[i:(i+10), :] = pool.map(pool_function, range(i+1, i+11))
        pool.close()
        pool.join()
        for j in range(i+1, i+11):
            fn = num_to_text(j)
            filename = basename + fn + ".edf"
            os.remove(filename)


    # Step 4: Close Pool and let all the processes complete
      # postpones the execution of next line of code until all processes in the queue are done.

    #np.set_printoptions(precision=2)
    #print(np.array(result[:, 1:10]))

    myFile = open('compressed.csv', 'w')
    with myFile:
        writer = csv.writer(myFile)
        writer.writerows(result)



'''
import numpy as np
from time import time



# Step 1: Redefine, to accept `i`, the iteration number
def howmany_within_range2(i, row, minimum, maximum):
    """Returns how many numbers lie within `maximum` and `minimum` in a given `row`"""
    count = 0
    for n in row:
        if minimum <= n <= maximum:
            count = count + 1
    return (i, count)


# Step 2: Define callback function to collect the output in `results`
def collect_result(result):
    global results
    results.append(result)


# Prepare data
np.random.RandomState(100)
arr = np.random.randint(0, 10, size=[200000, 5])
data = arr.tolist()


import multiprocessing as mp
pool = mp.Pool(mp.cpu_count())

results = []



# Step 3: Use loop to parallelize
for i, row in enumerate(data):
    pool.apply_async(howmany_within_range2, args=(i, row, 4, 8), callback=collect_result)

# Step 4: Close Pool and let all the processes complete
pool.close()
pool.join()  # postpones the execution of next line of code until all processes in the queue are done.

# Step 5: Sort results [OPTIONAL]
results.sort(key=lambda x: x[0])
results_final = [r for i, r in results]

print(results)
'''

'''
def get_core(x):
    f = pyedflib.EdfReader(x)
    n = f.signals_in_file
    N = max(f.getNSamples()) * 2
    sigbufs = np.zeros((n, N))
    for i in np.arange(n):
        sigbufs[i, :] = np.repeat(f.readSignal(i), N / f.getNSamples()[i])

    sigbufs = sigbufs.reshape((14000, 8130))
    X = tl.tensor(sigbufs, dtype='float32')
    tucker_rank = [100, 100]
    core, tucker_factors = tucker(X, ranks=tucker_rank, init='random', tol=10e-4)
    return np.array(core)


def multiprocessing_func(x):
    X[x, :, :] = get_core("/Users/danielcazzaniga/Downloads/shhs1-200001.edf")
    #time.sleep(4)
    print(x)


if __name__ == '__main__':
    X = np.ndarray(shape=(2, 100, 100), dtype='float32')

    print("____Baseline____")
    starttime = time.time()
    processes = []
    for i in range(2):
        multiprocessing_func(i)

    print('That took {} seconds'.format(time.time() - starttime))

    print("_____MultiProcessing_____")
    starttime = time.time()
    p1 = multiprocessing.Process(target=multiprocessing_func, args=(0,))
    p2 = multiprocessing.Process(target=multiprocessing_func, args=(1,))
    #p3 = multiprocessing.Process(target=multiprocessing_func, args=(2,))
    p1.start()
    p2.start()
    #p3.start()
    p1.join()
    p2.join()
    #p3.join()

    print('That took {} seconds'.format(time.time() - starttime))

'''