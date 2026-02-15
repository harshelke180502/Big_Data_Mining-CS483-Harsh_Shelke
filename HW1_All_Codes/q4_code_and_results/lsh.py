# Authors: Jessica Su, Wanzi Zhou, Pratyaksh Sharma, Dylan Liu, Ansh Shukla
#Modified: Alex Porter
import numpy as np
import random
import time
import pdb
import unittest
from PIL import Image
import matplotlib.pyplot as plt

# Finds the L1 distance between two vectors
# u and v are 1-dimensional np.array objects
def l1(u, v):
    return np.sum(np.abs(u - v))

# Loads the data into a np array, where each row corresponds to
# an image patch -- this step is sort of slow.
# Each row in the data is an image, and there are 400 columns.
def load_data(filename):
    return np.genfromtxt(filename, delimiter=',')

# Creates a hash function from a list of dimensions and thresholds.
def create_function(dimensions, thresholds):
    def f(v):
        boolarray = [v[dimensions[i]] >= thresholds[i] for i in range(len(dimensions))]
        return "".join(map(str, map(int, boolarray)))
    return f

# Creates the LSH functions (functions that compute L K-bit hash keys).
# Each function selects k dimensions (i.e. column indices of the image matrix)
# at random, and then chooses a random threshold for each dimension, between 0 and
# 255.  For any image, if its value on a given dimension is greater than or equal to
# the randomly chosen threshold, we set that bit to 1.  Each hash function returns
# a length-k bit string of the form "0101010001101001...", and the L hash functions 
# will produce L such bit strings for each image.
def create_functions(k, L, num_dimensions=400, min_threshold=0, max_threshold=255):
    functions = []
    for i in range(L):
        dimensions = np.random.randint(low = 0, 
                                   high = num_dimensions,
                                   size = k)
        thresholds = np.random.randint(low = min_threshold, 
                                   high = max_threshold + 1, 
                                   size = k)

        functions.append(create_function(dimensions, thresholds))
    return functions

# Hashes an individual vector (i.e. image).  This produces an array with L
# entries, where each entry is a string of k bits.
def hash_vector(functions, v):
    return np.array([f(v) for f in functions])

# Hashes the data in A, where each row is a datapoint, using the L
# functions in "functions."
def hash_data(functions, A):
    return np.array(list(map(lambda v: hash_vector(functions, v), A)))

# Retrieve all of the points that hash to one of the same buckets 
# as the query point.  Do not do any random sampling (unlike what the first
# part of this problem prescribes).
# Don't retrieve a point if it is the same point as the query point.
def get_candidates(hashed_A, hashed_point, query_index):
    return filter(lambda i: i != query_index and \
        any(hashed_point == hashed_A[i]), range(len(hashed_A)))

# Sets up the LSH.  You should try to call this function as few times as 
# possible, since it is expensive.
# A: The dataset in which each row is an image patch.
# Return the LSH functions and hashed data structure.
def lsh_setup(A, k = 24, L = 10):
    functions = create_functions(k = k, L = L)
    hashed_A = hash_data(functions, A)
    return (functions, hashed_A)

# Run the entire LSH algorithm
def lsh_search(A, hashed_A, functions, query_index, num_neighbors = 10):
    hashed_point = hash_vector(functions, A[query_index, :])
    candidate_row_nums = get_candidates(hashed_A, hashed_point, query_index)
    
    distances = map(lambda r: (r, l1(A[r], A[query_index])), candidate_row_nums)
    best_neighbors = sorted(distances, key=lambda t: t[1])[:num_neighbors]

    return [t[0] for t in best_neighbors]

# Plots images at the specified rows and saves them each to files.
def plot(A, row_nums, base_filename):
    for row_num in row_nums:
        patch = np.reshape(A[row_num, :], [20, 20])
        im = Image.fromarray(patch)
        if im.mode != 'RGB':
            im = im.convert('RGB')
        im.save(base_filename + "-" + str(row_num) + ".png")

def linear_search(A, query_index, num_neighbors):
    distances = []
    for i in range(len(A)):
        if i != query_index:
            distances.append((i, l1(A[i], A[query_index])))
    
    # Sort by distance and take the top k
    distances.sort(key=lambda x: x[1])
    return [t[0] for t in distances[:num_neighbors]]

# TODO: Write a function that computes the error measure
def error_measure(A, query_indices, lsh_results, linear_results):
    errors = []
    for i in range(len(query_indices)):
        q_idx = query_indices[i]
        lsh_neighs = lsh_results[i]
        linear_neighs = linear_results[i]
        
        # Sum of distances for top 3 neighbors
        # need at least 3 neighbors found by each method
        lsh_dist_sum = sum(l1(A[q_idx], A[idx]) for idx in lsh_neighs[:3])
        linear_dist_sum = sum(l1(A[q_idx], A[idx]) for idx in linear_neighs[:3])
        
        errors.append(lsh_dist_sum / linear_dist_sum)
    return np.mean(errors)

# TODO: Solve Problem 4
def problem4():
    A = load_data('data/patches.csv')
    query_indices = [100 * (i + 1) for i in range(10)]
    
    
    print("Running Linear Search for all queries...")
    start_time = time.time()
    linear_results = []
    for q_idx in query_indices:
        linear_results.append(linear_search(A, q_idx, 10))
    avg_linear_time = (time.time() - start_time) / 10
    print(f"Average Linear Search Time: {avg_linear_time:.4f}s")

    # 1. LSH with default parameters k=24, L=10
    print("Running LSH (L=10, k=24)...")
    functions, hashed_A = lsh_setup(A, k=24, L=10)
    start_time = time.time()
    lsh_results_default = []
    for q_idx in query_indices:
        lsh_results_default.append(lsh_search(A, hashed_A, functions, q_idx, 10))
    avg_lsh_time_default = (time.time() - start_time) / 10
    print(f"Average LSH Search Time: {avg_lsh_time_default:.4f}s")
    
    error_default = error_measure(A, query_indices, lsh_results_default, linear_results)
    print(f"Error for L=10, k=24: {error_default:.4f}")

    # 2. Varying L (k=24)
    Ls = [10, 12, 14, 16, 18, 20]
    errors_L = []
    print("Varying L (k=24)...")
    for L in Ls:
        functions, hashed_A = lsh_setup(A, k=24, L=L)
        lsh_results = [lsh_search(A, hashed_A, functions, q_idx, 10) for q_idx in query_indices]
        err = error_measure(A, query_indices, lsh_results, linear_results)
        errors_L.append(err)
        print(f"L={L}, Error={err:.4f}")
    
    # Plot L vs Error
    plt.figure()
    plt.plot(Ls, errors_L, marker='o')
    plt.xlabel('L')
    plt.ylabel('Error')
    plt.title('Error vs L (k=24)')
    plt.grid(True)
    plt.savefig('error_vs_L.png')
    print("Saved error_vs_L.png")

    # 3. Varying k (L=10)
    ks = [16, 18, 20, 22, 24]
    errors_k = []
    print("Varying k (L=10)...")
    for k in ks:
        functions, hashed_A = lsh_setup(A, k=k, L=10)
        lsh_results = [lsh_search(A, hashed_A, functions, q_idx, 10) for q_idx in query_indices]
        err = error_measure(A, query_indices, lsh_results, linear_results)
        errors_k.append(err)
        print(f"k={k}, Error={err:.4f}")
    
    # Plot k vs Error
    plt.figure()
    plt.plot(ks, errors_k, marker='o')
    plt.xlabel('k')
    plt.ylabel('Error')
    plt.title('Error vs k (L=10)')
    plt.grid(True)
    plt.savefig('error_vs_k.png')
    print("Saved error_vs_k.png")

    # 4. Visualizing neighbors for row 100
    print("Visualizing neighbors for row 100...")
    
    def plot_grid(query_idx, neighbor_indices, title, filename):
        # Include original query as the first image
        indices = [query_idx] + list(neighbor_indices)
        plt.figure(figsize=(18, 3))
        for i, idx in enumerate(indices):
            plt.subplot(1, len(indices), i + 1)
            plt.imshow(A[idx].reshape(20, 20), cmap='gray')
            plt.axis('off')
            if i == 0:
                plt.title(f"Query: {idx}", fontweight='bold')
            else:
                plt.title(f"Idx: {idx}")
        plt.suptitle(title)
        plt.tight_layout()
        plt.savefig(filename)
        print(f"Saved {filename}")

    # Plotted LSH neighbors including original query
    plot_grid(100, lsh_results_default[0], "LSH Top 10 Neighbors for Row 100 (Primary: Query)", "lsh_neighbors_grid.png")
    
    # Plotted Linear neighbors including original query
    plot_grid(100, linear_results[0], "Linear Search Top 10 Neighbors for Row 100 (Primary: Query)", "linear_neighbors_grid.png")

#### TESTS #####

class TestLSH(unittest.TestCase):
    def test_l1(self):
        u = np.array([1, 2, 3, 4])
        v = np.array([2, 3, 2, 3])
        self.assertEqual(l1(u, v), 4)

    def test_hash_data(self):
        f1 = lambda v: sum(v)
        f2 = lambda v: sum([x * x for x in v])
        A = np.array([[1, 2, 3], [4, 5, 6]])
        self.assertEqual(f1(A[0,:]), 6)
        self.assertEqual(f2(A[0,:]), 14)

        functions = [f1, f2]
        self.assertTrue(np.array_equal(hash_vector(functions, A[0, :]), np.array([6, 14])))
        self.assertTrue(np.array_equal(hash_data(functions, A), np.array([[6, 14], [15, 77]])))

    ### TODO: Write your tests here (they won't be graded, 
    ### but you may find them helpful)


if __name__ == '__main__':
    unittest.main(exit=False) ### TODO: Uncomment this to run tests
    problem4()
