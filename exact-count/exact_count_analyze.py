import json
import pprint

'''
This is a one-off script to try to debug some unexpected results.
'''

pp = pprint.PrettyPrinter(indent=4)

# read in the json file at results/exact_count_data.json
with open('./results/exact_count_data.json', 'r') as f:
    data = json.load(f)

results = []
for res in data:
    total = 0
    num_leaf_avg = 0
    leaf_over_frac_avg = 0
    branch_over_frac_avg = 0
    for stats in res['forest_stats'].values():
        for tree in stats['per_tree'].values():
            if tree['num_cols'] == res['dim']:
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
    results.append(f"d{res['dim']}.v{res['num_val']}.r{res['num_row']}.c{res['num_col']}, prec {round(res['correct_averages'],2)}, err {round(res['error_averages'],2)}, errsd {round(res['error_std_devs'],2)}, num_leaf {round(num_leaf_avg)}, leaf_over {round(leaf_over_frac_avg,2)}, branch_over {round(branch_over_frac_avg,2)}  ")
# sort results alphabetically
results.sort()
for thing in results:
    print(thing)