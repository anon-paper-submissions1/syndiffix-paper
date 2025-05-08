import sys
import pandas as pd
import numpy as np
import statistics
import itertools
import json
from syndiffix import Synthesizer
from syndiffix.common import AnonymizationParams
import pprint
import os

pp = pprint.PrettyPrinter(indent=4)

'''
This program tests the ability for an attacker to determine the exact count of
a given column value.

The idea is that, by taking many samples of the column value count over many
different synthetic datasets, the attacker can average out the noise. 

In this test, we test averaging multiple 2dim and 3dim synthetic datasets.

This test sets three parameters, num_col the number of columns in the dataset, num_val the number of distinct values in each column other than that of the targeted column, and agg_size the number of rows of the target value in the target column being predicted. We vary these parameters to see how they affect the attacker's ability to determine the exact count of the target value.
'''

save_tree_walk = False
TEST = False
if TEST is False:
    from syndiffix_tools.tree_walker import *
num_other_rows = 50
GET_GOOD_PARAMS = True
GET_LOW_PARAMS = False
if GET_LOW_PARAMS is True:
    num_cols = [5, 6]
    dims = [1,3]
    num_vals = [2, 3]
    agg_sizes = [5, 200]
    rows_per_val = [24, 200]
elif GET_GOOD_PARAMS is True:
    num_cols = [5, 200]
    dims = [1,3]
    num_vals = [2, 10]
    agg_sizes = [100, 200]
    rows_per_val = [24, 200]
else:
    num_cols = [5, 200]
    dims = [1,3]
    num_vals = [2, 20]
    agg_sizes = [5, 200]
    rows_per_val = [24, 200]

# read command line arguments
make_slurm = False
slurm_num = 50
do_attack_num = slurm_num + 1
if len(sys.argv) > 1:
    if sys.argv[1] == 'slurm':
        make_slurm = True
    else:
        do_attack_num = int(sys.argv[1])

def get_forest_stats(forest):
    '''
    `forest` is the output of `TreeWalker.get_forest_nodes()`
    '''
    stats = {
        'overall': {
            'num_trees': 0,
            'num_nodes': 0,
            'num_leaf': 0,
            'num_branch': 0,
            'leaf_singularity': 0,
            'branch_singularity': 0,
            'leaf_over_threshold': 0,
            'branch_over_threshold': 0,
        },
        'per_tree': {
        },
    }
    overall = stats['overall']
    for node in forest.values():
        comb = str(tuple(node['columns']))
        if comb not in stats['per_tree']:
            stats['per_tree'][comb] = {
                'num_cols': len(node['columns']),
                'num_nodes': 0,
                'num_leaf': 0,
                'num_branch': 0,
                'leaf_singularity': 0,
                'branch_singularity': 0,
                'leaf_over_threshold': 0,
                'branch_over_threshold': 0,
            }
        tree = stats['per_tree'][comb]
        overall['num_nodes'] += 1
        tree['num_nodes'] += 1
        if node['node_type'] == 'leaf':
            overall['num_leaf'] += 1
            tree['num_leaf'] += 1
            if node['singularity']:
                overall['leaf_singularity'] += 1
                tree['leaf_singularity'] += 1
            if node['over_threshold']:
                overall['leaf_over_threshold'] += 1
                tree['leaf_over_threshold'] += 1
        elif node['node_type'] == 'branch':
            overall['num_branch'] += 1
            tree['num_branch'] += 1
            if node['singularity']:
                overall['branch_singularity'] += 1
                tree['branch_singularity'] += 1
            if node['over_threshold']:
                overall['branch_over_threshold'] += 1
                tree['branch_over_threshold'] += 1
    return stats

def get_dim_stats(forest_stats, dim):
    num_leaf_avg = 0
    leaf_over_frac_avg = 0
    branch_over_frac_avg = 0
    total = 0
    #pp.pprint(forest_stats)
    for tree in forest_stats['per_tree'].values():
        if tree['num_cols'] == dim:
            total += 1
            num_leaf_avg += tree['num_leaf']
            leaf_over_frac_avg += tree['leaf_over_threshold'] / tree['num_leaf']
            if tree['num_branch'] == 0:
                branch_over_frac_avg -= 100
            else:
                branch_over_frac_avg += tree['branch_over_threshold'] / tree['num_branch']
    num_leaf_avg /= total
    leaf_over_frac_avg /= total
    branch_over_frac_avg /= total
    return num_leaf_avg, leaf_over_frac_avg, branch_over_frac_avg

def print_progress_wheel(wheel):
    print(next(wheel) + '\b', end='', flush=True)

def progress_wheel():
    wheel = itertools.cycle(['-', '/', '|', '\\'])
    while True:
        yield next(wheel)

def get_precision(noisy_counts, exact_count):
    num_correct = 0
    # true_row_count assumes that we can determine the exact number of
    # rows by taking average from all tables
    guess = round(sum(noisy_counts)/len(noisy_counts))
    if guess == exact_count:
        correct = 1
    else:
        correct = 0
    error = abs(guess - exact_count)
    return {'correct': correct, 'guessed': guess, 'exact': exact_count, 'error': error}

def make_df(num_val, num_col, agg_size, base_rows_per_val):
    data = {}
    agg_sizes_base = base_rows_per_val * num_val
    agg_sizes_total = agg_sizes_base + agg_size

    # Create a custom probability distribution
    prob_dist = np.linspace(1, 1.3, num_val)
    prob_dist /= prob_dist.sum()

    ran_col_num = np.random.randint(0, 1000000)
    for i in range(num_col):
        if i == 0:
            # The 0th column has the value 1
            data[f'col{ran_col_num}_{i}'] = [1] * agg_sizes_base
            # Now add the target rows, with value 0
            for j in range(agg_size):
                data[f'col{ran_col_num}_{i}'].append(0)
        else:
            # The remaining columns each have num_val distinct values, assigned according to the custom probability distribution
            values = np.random.choice(range(num_val), agg_sizes_total, p=prob_dist)
            data[f'col{ran_col_num}_{i}'] = values
    df = pd.DataFrame(data)
    df = df.sample(frac=1).reset_index(drop=True)
    return df

def get_col_combs(df, col0, dim):
    cols_without_col0 = [col for col in df.columns if col != col0]
    col_combs = []
    if dim == 1:
        col_combs.append([col0])
    if dim == 2:
        for col in cols_without_col0:
            col_combs.append([col0,col])
    elif dim == 3:
        for comb in itertools.combinations(cols_without_col0, 2):
            col_combs.append([col0] + list(comb))
    return col_combs

def do_attack(num_val, num_col, dim, agg_size, base_rows_per_val):
    '''
    num_val: number of distinct values in each other column
    num_col: number of columns
    dim: number of dimensions of the attack
    agg_size: number of rows of the value being predicted
    '''
    result = {
        'num_val': num_val,
        'num_col': num_col,
        'dim': dim,
        'agg_size': agg_size,
        'exact_count': None,
        'guessed_count': None,
        'error': None,
        'correct': None,
        'num_leaf': None,
        'frac_leaf_over': None,
        'total_table_rows': None,
        'tree_walk': None,
    }
    df = make_df(num_val, num_col, agg_size, base_rows_per_val)
    result['total_table_rows'] = df.shape[0]
    # get the name of the first column
    col0 = df.columns[0]
    exact_count = df[df[col0] == 0].shape[0]
    result['exact_count'] = exact_count
    col_combs = get_col_combs(df, col0, dim)
    # get the count of the target value 0 for col0 in df
    if TEST: print(col_combs)
    noisy_counts = []
    num_leafs = []
    frac_leaf_overs = []
    # select a random seed for the synthesizer
    seed = np.random.randint(0, 1000000)
    sdx_seed = str(seed).encode()
    for col_comb in col_combs:
        syn = Synthesizer(df[col_comb],
                anonymization_params=AnonymizationParams(salt=sdx_seed))
        df_syn = syn.sample()
        ncount = df_syn[df_syn[col0] == 0].shape[0]
        noisy_counts.append(ncount)
        if TEST is False:
            tw = TreeWalker(syn)
            forest = tw.get_forest_nodes()
            if save_tree_walk:
                result['tree_walk'] = forest
            stats = get_forest_stats(forest)
            num_leaf, frac_leaf_over, _ = get_dim_stats(stats, dim)
            num_leafs.append(num_leaf)
            frac_leaf_overs.append(frac_leaf_over)
    result['num_leaf'] = statistics.mean(num_leafs)
    result['frac_leaf_over'] = statistics.mean(frac_leaf_overs)
    prec = get_precision(noisy_counts, exact_count)
    result['correct'] = prec['correct']
    result['error'] = prec['error']
    result['guessed_count'] = prec['guessed']
    return result

if make_slurm:
    with open('exact_count_slurm.sh', 'w') as f:
        f.write('#!/bin/bash\n')
        # set the time limit for 48 hours
        f.write('#SBATCH --time=0-48:00\n')
        f.write('#SBATCH --mem=10G\n')
        f.write('#SBATCH --output /dev/null\n')
        f.write('#SBATCH --array=0-' + str(slurm_num - 1) + '\n')
        f.write('source ../sdx_tests/sdx_venv/bin/activate' + '\n')
        f.write('python exact_count.py $SLURM_ARRAY_TASK_ID\n')
    quit()

wheel = progress_wheel()
file_name = f'results.{do_attack_num}.json'
file_path = os.path.join('exact_count_results', file_name)
# Let the seed be random
# np.random.seed(do_attack_num)
if not os.path.exists('exact_count_results'):
    os.makedirs('exact_count_results')
for _ in range(10000):
    # Just make attack after attack. Can use scancel to end (or timeout
    # of slurm job)
    num_col = np.random.randint(num_cols[0], num_cols[1]+1)
    dim = np.random.randint(dims[0], dims[1]+1)
    num_val = np.random.randint(num_vals[0], num_vals[1]+1)
    agg_size = np.random.randint(agg_sizes[0], agg_sizes[1]+1)
    base_rows_per_val = np.random.randint(rows_per_val[0], rows_per_val[1])
    # read in the json file at file_path if it exists
    if os.path.exists(file_path):
        with open(file_path, 'r') as f:
            data = json.load(f)
    else:
        data = []
    result = do_attack(num_val, num_col, dim, agg_size, base_rows_per_val)
    data.append(result)
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=4)