import argparse
import os
import pandas as pd
import json
import sys
import random
import statistics
from syndiffix import Synthesizer
from anonymity_loss_coefficient import AnonymityLossCoefficient
import pprint
import sys

'''
All outlier tests are based on detecting whether there are:
        0 or 1 outlier
        1 or 2 outliers
        2 or 3 outliers
Value outlier tests:
    Detect presence:
        We know the presence of the outlier will raise the average value
        of the column with the outlier value. But, we need to know what the expected value without the outlier should be...
    Infer value of another column:
'''

predict_thresholds = [1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0]
col_pre = 'vals'
num_1col_runs = 500000
prediction_multipliers = [1, 2, 3]
if 'SDX_TEST_DIR' in os.environ:
    base_path = os.getenv('SDX_TEST_DIR')
else:
    base_path = os.getcwd()
os.makedirs(base_path, exist_ok=True)
runs_path = os.path.join(base_path, 'outlier_theory')
os.makedirs(runs_path, exist_ok=True)
tests_path = os.path.join(runs_path, 'tests')
os.makedirs(tests_path, exist_ok=True)
results_path = os.path.join(runs_path, 'results')
os.makedirs(results_path, exist_ok=True)
pp = pprint.PrettyPrinter(indent=4)

def build_basic_table(num_vals, ex_factor, num_ex, dist, num_aid):
    num_other_cols = 5
    num_vals_per_other_col = 5
    # Generate a random 5-digit number for the column names
    random_suffix = ''.join(random.choices('0123456789', k=5))
    aid_col = f'aid_{random_suffix}'
    vals_col = f'{col_pre}_{random_suffix}'
    other_cols = [f'{col_pre}_{i}_{random_suffix}' for i in range(num_other_cols)]

    # Generate num_aid distinct aid values
    aid_values = random.sample(range(10000000), num_aid)
    aid_values = [str(e) for e in aid_values]
    
    # Select the unknown_aid
    unknown_aid = random.choice(aid_values)
    
    # Select the known_aid values
    known_aid_values = random.sample([aid for aid in aid_values if aid != unknown_aid], num_ex - 1)
    
    # Select the other_aid values
    other_aid_values = [aid for aid in aid_values if aid != unknown_aid and aid not in known_aid_values]
    
    # Generate num_vals distinct values for the vals column
    vals_values = random.sample(range(1000, 100000), num_vals)
    vals_values = [str(e) for e in vals_values]
    #print(f'vals_values: {vals_values}')
    if len(vals_values) != num_vals:
        print(f"Fail: Expected {num_vals} distinct values, but found {len(vals_values)}.")
        print(vals_values)
        sys.exit(1)

    # other_cols_vals is indexed by col_index, then value_index
    other_cols_vals = [random.sample(range(1000, 100000), num_vals_per_other_col) for _ in range(num_other_cols)]
    
    # Assign a random val to each aid
    aid_to_val = {aid: random.choice(vals_values) for aid in aid_values}
    
    # Ensure the known_aids have the same val: known_val
    known_val = random.choice(vals_values)
    for aid in known_aid_values:
        aid_to_val[aid] = known_val

    # Create the DataFrame
    rows = []
    
    # Add rows for other_aid values
    max_rows = 0
    for aid in other_aid_values:
        num_rows = 0   # just to please lint
        if dist == 'uniform':
            num_rows = random.randint(40, 60)
        elif dist == 'normal':
            num_rows = max(40, int(random.normalvariate(80, 20)))
        elif dist == 'flat':
            num_rows = 50
        if num_rows > max_rows:
            max_rows = num_rows
        for _ in range(num_rows):
            rows.append({aid_col: aid, vals_col: aid_to_val[aid]})
    
    # Add rows for known_aid values
    for aid in known_aid_values:
        for _ in range(max_rows * ex_factor):
            rows.append({aid_col: aid, vals_col: known_val})
    
    # Add rows for unknown_aid
    unknown_val = aid_to_val[unknown_aid]
    for _ in range(max_rows * ex_factor):
        rows.append({aid_col: unknown_aid, vals_col: unknown_val})

    # add all of the other columns
    for row in rows:
        for i, col in enumerate(other_cols):
            row[col] = random.choice(other_cols_vals[i])
    
    df = pd.DataFrame(rows)
    
    # Shuffle the DataFrame
    df = df.sample(frac=1).reset_index(drop=True)
    df = df.astype(str)
    
    return df, known_val, unknown_aid, unknown_val

def make_slurm():
    pass

def make_plot():
    output_path = os.path.join(runs_path, "results.parquet")
    # Read the Parquet file into a DataFrame
    df = pd.read_parquet(output_path)
    print(df.columns)
    print("Get info for ex_factor == 100")
    make_one_plot(df[(df['ex_factor'] == 100)])
    print("Get info for ex_factor < 100")
    make_one_plot(df[(df['ex_factor'] != 100)])
    print("Get info for dist is flat")
    make_one_plot(df[(df['dist'] == 'flat')])
    print("Get info for dist not flat")
    make_one_plot(df[(df['dist'] != 'flat')])


def make_one_plot(df):
    alcm = AnonymityLossCoefficient()
    print(f"Total rows = {len(df)}")
    df_counts = df.groupby(['threshold', 'prediction']).size().unstack(fill_value=0)
    print(df_counts)
    # get the average of the num_vals column
    num_vals_avg = df['num_vals'].mean()
    # compute the precision of a statistical guess
    prec_baseline = 1 / num_vals_avg
    print(f"Precision baseline: {prec_baseline}")
    # loop through each row in df_counts
    for index, row in df_counts.iterrows():
        prec = row['tp'] / (row['tp'] + row['fp'])
        recall = (row['tp'] + row['fp']) / (row['tp'] + row['fp'] + row['abstain'])
        print(f"Threshold: {index}, Precision: {prec}, Recall: {recall}")
        alc = alcm.alc(p_base=prec_baseline, r_base=1, p_attack=prec, r_attack=recall)
        print(f"ALC: {alc}")

    df_fp = df[(df['threshold'] == 2.0) & (df['prediction'] == 'fp')]
    df_tp = df[(df['threshold'] == 2.0) & (df['prediction'] == 'tp')]
    print("num_vals:")
    print(f"    fp: avg: {df_fp['num_vals'].mean()}, std: {df_fp['num_vals'].std()}")
    print(f"    tp: avg: {df_tp['num_vals'].mean()}, std: {df_tp['num_vals'].std()}")
    print("ex_factor:")
    print(f"    fp: avg: {df_fp['ex_factor'].mean()}, std: {df_fp['ex_factor'].std()}")
    print(f"    tp: avg: {df_tp['ex_factor'].mean()}, std: {df_tp['ex_factor'].std()}")
    print("num_aid:")
    print(f"    fp: avg: {df_fp['num_aid'].mean()}, std: {df_fp['num_aid'].std()}")
    print(f"    tp: avg: {df_tp['num_aid'].mean()}, std: {df_tp['num_aid'].std()}")


def most_frequent_value(df):
    # Get the column name
    column_name = df.columns[0]
    
    # Use value_counts to count occurrences of each value
    value_counts = df[column_name].value_counts()
    
    # Select the most frequent value
    most_frequent = value_counts.idxmax()
    
    # Calculate gap_1_2
    if len(value_counts) > 1:
        second_most_frequent_count = value_counts.iloc[1]
    else:
        second_most_frequent_count = 0
    gap_1_2 = value_counts.iloc[0] - second_most_frequent_count
    
    # Calculate gap_avg
    if len(value_counts) > 1:
        avg_count = value_counts.iloc[1:].mean()
    else:
        avg_count = 0
    gap_avg = value_counts.iloc[0] - avg_count
    
    return most_frequent, gap_1_2, gap_avg

def run_one_attack(num_vals, ex_factor, num_ex, dist, num_aid):
    # Returns 1 if predictions is correct, 0 otherwise
    #print(f"run_one_attack() with params num_vals = {num_vals}, ex_factor = {ex_factor}, num_ex = {num_ex}, dist = {dist}, num_aid = {num_aid}")
    df, known_val, unknown_aid, unknown_val = build_basic_table(num_vals, ex_factor, num_ex, dist, num_aid)
    #print(f"the known_val is {known_val}, the unknown_val is {unknown_val}")
    # split df into two dataframes, one with the aid column and one with the vals column
    aid_col = df.columns[0]
    target_col = df.columns[1]
    # print the number of rows for each distinct value in df[target_col]
    #print("Rows per value in target_col of original df")
    #print(df[target_col].value_counts())
    df_aid = df[[aid_col]]
    df_syn_target = Synthesizer(df[[target_col]], pids=df_aid).sample()
    #print("Rows per value in target_col of df_syn_target")
    #print(df_syn_target[target_col].value_counts())
    #print(f"num distinct vals in orig target: {df[target_col].nunique()}")
    #print(f"num distinct vals in syn target: {df_syn_target[target_col].nunique()}")
    #print(f"num distinct vals in syn target: {df_syn_target[target_col].nunique()}")
    # Get the count of rows in syn_target where target_col == known_val
    known_val_count = len(df_syn_target[df_syn_target[target_col] == known_val])
    #print(f"known_val_count: {known_val_count}")
    df_others = []
    target_val_counts = []
    for col in df.columns:
        if col == target_col or col == aid_col:
            continue
        df_syn = Synthesizer(df[[target_col, col]], pids=df_aid).sample()
        #print(f"num distinct vals in {col}: {df_syn[col].nunique()}")
        tar_val_count = len(df_syn[df_syn[target_col] == known_val])
        #print(f"tar_val_count: {tar_val_count}")
        df_others.append(df_syn)
        target_val_counts.append(tar_val_count)
    # the median of the target_val_counts is the baseline where all of the known
    # and unknown AIDs are flattened
    flattened_count = statistics.median(target_val_counts)
    #print(f"flattened_count: {flattened_count} is median of {target_val_counts}")
    # To identify the unknown outlier value, we expect the count of known_val_count
    # to be much higher than the flattened_count
    predictions = {}
    for predict_threshold in predict_thresholds:
        predictions[predict_threshold] = {'known_val_count': known_val_count, 'flattened_count': flattened_count}
        if known_val_count > flattened_count * predict_threshold:
            if known_val == unknown_val:
                predictions[predict_threshold]['prediction'] = 'tp'
            else:
                predictions[predict_threshold]['prediction'] = 'fp'
        else:
            predictions[predict_threshold]['prediction'] = 'abstain'
    if df[target_col].nunique() != df_syn_target[target_col].nunique():
        print(f"Bad num unique: {df[target_col].nunique()} {df_syn_target[target_col].nunique()}")
        pp.pprint(predictions)
    return predictions

def get_sorted_value_counts(value_counts):
    # Sort the dictionary items by value in descending order
    sorted_items = sorted(value_counts.items(), key=lambda item: item[1], reverse=True)

    # Separate the keys and values into two lists
    vals = [item[0] for item in sorted_items]
    counts = [item[1] for item in sorted_items]
    return vals, counts

def update_attack(alcm, res_key_prefix, filtered_df, precision_results, all_results):
    for mult in prediction_multipliers:
        res_key = f"{res_key_prefix}x{mult}"
        res_col = f"result_1_2_{mult}"
        df_predict = filtered_df[filtered_df[res_col] != 'abstain']
        if len(df_predict) == 0:
            precision_results[res_key] = {
                                    'prec_attack': None,
                                    'prec_base': None,
                                    'coverage': 0,
                                    'alc': 0,
                                    'num_rows': len(filtered_df),
                                    }
            all_results['summary'].append([res_key, 0])
            return
        df_true = filtered_df[filtered_df[res_col] == 'true']
        df_false = filtered_df[filtered_df[res_col] == 'false']
        coverage = (len(df_true) + len(df_false)) / len(filtered_df)
        prec_attack = len(df_true) / (len(df_true) + len(df_false))
        prec_base = 1 / (df_predict['num_vals'].sum() / len(df_predict))
        alc = alcm.alc(p_base=prec_base, r_base=coverage, p_attack=prec_attack, r_attack=coverage)

        # Store the precision result
        precision_results[res_key] = {
                                    'prec_attack': prec_attack,
                                    'prec_base': prec_base,
                                    'coverage': coverage,
                                    'alc': alc,
                                    'num_rows': len(filtered_df),
                                    }
        all_results['summary'].append([res_key, alc])

class Attack1colResults:
    def __init__(self, path):
        self.parquet_path = os.path.join(path, f'1col_results.parquet')
        self.analyze_path = os.path.join(path, f'1col_results.json')
        if os.path.exists(self.parquet_path):
            self.df = pd.read_parquet(self.parquet_path)
            print(f"Loaded {len(self.df)} rows from {self.parquet_path}")
        else:
            self.df = pd.DataFrame(columns=[
                'num_vals', 'ex_factor', 'num_ex', 'dist', 'num_aid', 'predict_val', 'known_val', 'unknown_val', 'gap_1_2', 'gap_avg', 'value_counts'
            ])
        self.add_count = 0

    def add(self, num_vals, ex_factor, num_ex, dist, num_aid, predict_val, known_val, unknown_val, gap_1_2, gap_avg, value_counts):
        value_counts = json.dumps(value_counts)
        new_row = pd.DataFrame([{
            'num_vals': num_vals,
            'ex_factor': ex_factor,
            'num_ex': num_ex,
            'dist': dist,
            'num_aid': num_aid,
            'predict_val': predict_val,
            'known_val': known_val,
            'unknown_val': unknown_val,
            'gap_1_2': gap_1_2,
            'gap_avg': gap_avg,
            'value_counts': value_counts
        }])
        self.df = pd.concat([self.df, new_row], ignore_index=True)
        self.add_count += 1

        if self.add_count >= 100:
            self.write_to_parquet()
            self.add_count = 0

    def write_to_parquet(self):
        self.df.to_parquet(self.parquet_path, index=False)

    def make_predictions(self):
        def val_prediction(row, mult):
            # Example logic to assign a value
            value_counts = json.loads(row['value_counts'])
            # find the key with the highest corresponding value
            vals, counts = get_sorted_value_counts(value_counts)
            if counts[0] > counts[1] * mult:
                return vals[0]
            else:
                return 'abstain'
        def get_result(row, col_name):
            if row[col_name] == 'abstain':
                return 'abstain'
            elif row[col_name] == row['unknown_val']:
                return 'true'
            else:
                return 'false'
        for mult in prediction_multipliers:
            pred_col_name = f"pred_1_2_{mult}"
            self.df[pred_col_name] = self.df.apply(val_prediction, axis=1, args=(mult,))
            res_col_name = f"result_1_2_{mult}"
            self.df[res_col_name] = self.df.apply(get_result, axis=1, args=(pred_col_name,))

    def analyze(self):
        alcm = AnonymityLossCoefficient()
        # List of columns to analyze
        columns_to_analyze = ['num_vals', 'ex_factor', 'num_ex', 'num_aid', 'dist']

        all_results = {'precision_results': {}, 'summary':[]}
        
        # Dictionary to store the precision results
        precision_results = all_results['precision_results']
        
        for column in columns_to_analyze:
            # Get the distinct parameters in the column
            distinct_params = self.df[column].unique()
            for param in distinct_params:
                # Filter the DataFrame for the current column/param
                filtered_df = self.df[self.df[column] == param]
                res_key_prefix = f"{column}_{param}_"
                update_attack(alcm, res_key_prefix, filtered_df, precision_results, all_results)
        
        filtered_df = self.df[(self.df['ex_factor'] == 20) & (self.df['num_ex'] == 3) & (self.df['num_aid'] == 150)]
        res_key_prefix = f"ex_factor_20_num_ex_3_num_aid_150_"
        update_attack(alcm, res_key_prefix, filtered_df, precision_results, all_results)
        filtered_df = self.df[(self.df['ex_factor'] == 20) & (self.df['num_ex'] == 3) & (self.df['num_aid'] == 150) & (self.df['num_vals'] == 2)]
        res_key_prefix = f"ex_factor_20_num_ex_3_num_aid_150_num_vals_2_"
        update_attack(alcm, res_key_prefix, filtered_df, precision_results, all_results)
        # sort all_results['summary'] by the second element in each sublist
        all_results['summary'] = sorted(all_results['summary'], key=lambda x: x[1], reverse=True)
        # Write the precision results to a JSON file
        with open(self.analyze_path, 'w') as f:
            json.dump(all_results, f, indent=4)
        pp.pprint(all_results)

def continuous_attacks(job_num=None):
    if job_num is None:
        job_num = ''
    else:
        job_num = f'_{job_num}'
    parquet_path = os.path.join(results_path, f'results{job_num}.parquet')
    json_path = os.path.join(results_path, f'results{job_num}.json')
    if os.path.exists(json_path):
        with open(json_path, 'r') as f:
            results = json.load(f)
    else:
        results = []
    for _ in range(100000000):
        dist = random.choice(['uniform', 'normal', 'flat'])
        num_vals = random.randint(2,10)
        ex_factor = random.randint(100,100)
        num_aid = random.randint(50,100)
        num_ex = 3
        predictions = run_one_attack(num_vals, ex_factor, num_ex, dist, num_aid)
        for threshold, prediction in predictions.items():
            results.append({
                'num_vals': num_vals,
                'ex_factor': ex_factor,
                'num_ex': num_ex,
                'dist': dist,
                'num_aid': num_aid,
                'prediction': prediction['prediction'],
                'threshold': threshold,
                'known_val_count': prediction['known_val_count'],
                'flattened_count': prediction['flattened_count']
            })
        # Write the results to a JSON file
        with open(json_path, 'w') as f:
            json.dump(results, f, indent=4)
        # Write the results to a parquet file
        df = pd.DataFrame(results)
        df.to_parquet(parquet_path, index=False)

def gather_results():
    # List to hold DataFrames
    dataframes = []
    # Iterate over all files in the directory
    print(f"Searching for Parquet files in {results_path}")
    for filename in os.listdir(results_path):
        if filename.endswith(".parquet"):
            file_path = os.path.join(results_path, filename)
            # Read the Parquet file into a DataFrame
            df = pd.read_parquet(file_path)
            # Append the DataFrame to the list
            dataframes.append(df)
    # Concatenate all DataFrames into one
    if dataframes:
        combined_df = pd.concat(dataframes, ignore_index=True)
        # Write the combined DataFrame to a Parquet file
        output_path = os.path.join(runs_path, "results.parquet")
        combined_df.to_parquet(output_path, index=False)
        print(f"Combined DataFrame written to {output_path}")
    else:
        print("No Parquet files found in the directory.")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", help="'slurm' to make slurmscript, 'plot' to plot the results, 'attacks' to run all attacks, or an integer to run a specific attack")
    args = parser.parse_args()

    if args.command == 'slurm':
        make_slurm()
    elif args.command == 'plot':
        make_plot()
    elif args.command == 'gather':
        gather_results()
    elif args.command == 'continuous':
        continuous_attacks()
    else:
        try:
            job_num = int(args.command)
            continuous_attacks(job_num=job_num)
        except ValueError:
            print(f"Unknown command: {args.command}")

if __name__ == "__main__":
    main()