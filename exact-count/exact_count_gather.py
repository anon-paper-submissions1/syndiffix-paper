import json
import os


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
    for stats in forest_stats.values():
        for tree in stats['per_tree'].values():
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

json_files = [pos_json for pos_json in os.listdir("./exact_count_results") if pos_json.endswith('.json')]

data_list = []

for file in json_files:
    with open(f"./exact_count_results/{file}", "r") as json_file:
        data = json.load(json_file)
        data_dict = {key: data[key] for key in ("num_val", "num_col", "dim", "num_row", "correct_averages", "error_averages", "error_std_devs", "samples", "total_table_rows")}
        data_dict['forest_stats'] = {}
        for key in data['tree_walks']:
            data_dict['forest_stats'][key] = get_forest_stats(data['tree_walks'][key])
        # Now I have the stats, let's dig out the one pertaining to the dim
        num_leaf_avg, leaf_over_frac_avg, branch_over_frac_avg = get_dim_stats(data_dict['forest_stats'], data_dict['dim'])
        data_dict['num_leaf_avg'] = num_leaf_avg
        data_dict['leaf_over_frac_avg'] = leaf_over_frac_avg
        data_dict['branch_over_frac_avg'] = branch_over_frac_avg
        data_list.append(data_dict)
# create directory 'results' if it doesn't exist
if not os.path.exists("results"):
    os.makedirs("results")
out_path = os.path.join("results", "exact_count_data.json")

with open(out_path, "w") as outfile:
    json.dump(data_list, outfile, indent=4)