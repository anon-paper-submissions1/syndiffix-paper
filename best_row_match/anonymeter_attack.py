import anonymeter_mods
import argparse
import os
import pandas as pd
import json
import sys
import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from collections import Counter
from alscore import ALScore
import pprint

pp = pprint.PrettyPrinter(indent=4)

if 'SDX_TEST_DIR' in os.environ:
    base_path = os.getenv('SDX_TEST_DIR')
else:
    base_path = os.getcwd()
if 'SDX_TEST_CODE' in os.environ:
    code_path = os.getenv('SDX_TEST_CODE')
    from syndiffix_tools.tables_manager import TablesManager
else:
    code_path = None
syn_path = os.path.join(base_path, 'synDatasets')
attack_path = os.path.join(base_path, 'anonymeter_attacks')
os.makedirs(attack_path, exist_ok=True)
plots_path = os.path.join(attack_path, 'plots')
os.makedirs(plots_path, exist_ok=True)
# This is the total number of attacks that will be run
num_attacks = 1000000
# This is the number of attacks per slurm job, and determines how many slurm jobs are created
num_attacks_per_job = 1000
max_subsets = 100
debug = False


# These are the variants of the attack that exploits sub-tables
variants = {
            'syn_meter_vanilla':[],
            'syn_meter_modal':[],
            'syn_meter_modal_50':[],
            'syn_meter_modal_90':[],
            'base_meter_vanilla':[],
            'base_meter_modal':[],
            'base_meter_modal_50':[],
            'base_meter_modal_90':[],
}
variant_labels = [ 'vanilla', 'modal', 'modal_50', 'modal_90', ]
# These are the thresholds we use to decide whether to use a prediction
col_comb_thresholds = {
                            'thresh_0':0,
                            'thresh_50':50,
                            'thresh_90':90,
}

num_known_config = {'num_known_all': 'all', 'num_known_3': '3', 'num_known_6': '6'}

def init_variants():
    for v_label in variants.keys():
        variants[v_label] = []

def convert_datetime_to_timestamp(df):
    for col in df.columns:
        if df[col].dtype == 'datetime64[ns]':
            df[col] = df[col].astype(int) / 10**9
    return df

def fit_encoders(dfs):
    # Get the string columns
    string_columns = dfs[0].select_dtypes(include=['object']).columns

    encoders = {col: LabelEncoder() for col in string_columns}

    for col in string_columns:
        # Concatenate the values from all DataFrames for this column
        values = pd.concat(df[col] for df in dfs).unique()
        # Fit the encoder on the unique values
        encoders[col].fit(values)

    return encoders

def transform_df(df, encoders):
    for col, encoder in encoders.items():
        df[col] = encoder.transform(df[col])
    return df

def transform_df_with_update(df, encoders):
    for col, encoder in encoders.items():
        if col in df.columns:
            unique_values = pd.Series(df[col].unique())
            unseen_values = unique_values[~unique_values.isin(encoder.classes_)]
            encoder.classes_ = np.concatenate([encoder.classes_, unseen_values])
            df[col] = encoder.transform(df[col])
    return df

def find_most_frequent_value(lst, fraction):
    if len(lst) == 0:
        return None

    # Count the frequency of each value in the list
    counter = Counter(lst)
    
    # Find the most common value and its count
    most_common_value, most_common_count = counter.most_common(1)[0]
    
    # Check if the most common value accounts for at least the given fraction of total entries
    if most_common_count / len(lst) > fraction:
        return most_common_value
    else:
        return None

def build_and_train_model(df, target_col, target_type):
    X = df.drop(target_col, axis=1)
    y = df[target_col]

    # If the target is categorical, encode it to integers
    if target_type == 'categorical':
        le = LabelEncoder()
        y = le.fit_transform(y)

    # Build and train the model
    if target_type == 'categorical':
        #print(f"building RandomForestClassifier with shape {X.shape}")
        try:
            model = RandomForestClassifier(random_state=42)
            #print("finished building RandomForestClassifier")
        except Exception as e:
            #print(f"A RandomForestClassifier error occurred: {e}")
            sys.exit(1)
    elif target_type == 'continuous':
        #print(f"building RandomForestRegressor with shape {X.shape}")
        try:
            model = RandomForestRegressor(random_state=42)
            #print("finished building RandomForestRegressor")
        except Exception as e:
            #print(f"A RandomForestRegressor error occurred: {e}")
            sys.exit(1)
    else:
        raise ValueError("target_type must be 'categorical' or 'continuous'")

    model.fit(X, y)
    return model

def attack_stats():
    with open(os.path.join(attack_path, 'attack_jobs.json'), 'r') as f:
        jobs = json.load(f)
    secrets = {3:{}, 6:{}, -1:{}}
    for job in jobs:
        nk = job['num_known']
        if job['secret'] not in secrets[nk]:
            secrets[nk][job['secret']] = job['num_runs']
        else:
            secrets[nk][job['secret']] += job['num_runs']
    
    for nk in [-1,3,6]:
        print(f"Number of secrets for {nk} known columns: {len(secrets[nk])}")
        print(f"Average count per secret: {sum(secrets[nk].values()) / len(secrets)}")

def make_config():
    ''' I want to generate num_attacks attacks. Each attack will be on a given secret
    column in a given table with given known columns. I will run multiple of these
    attacks per secret/table if necessary.
    '''

    num_known_columns = [-1,3,6]
    # Initialize attack_jobs
    attack_jobs = []

    # Loop over each directory name in syn_path
    for num_known in num_known_columns:
        attacks_so_far = 0
        while attacks_so_far < num_attacks:
            for dir_name in os.listdir(syn_path):
                dataset_path = os.path.join(syn_path, dir_name, 'anonymeter')
                # Check if dataset_path exists
                if not os.path.exists(dataset_path):
                    continue
                tm = TablesManager(dir_path=dataset_path)
                columns = list(tm.df_orig.columns)
                pid_cols = tm.get_pid_cols()
                if len(pid_cols) > 0:
                    # We can't really run the attack on time-series data
                    continue
                for secret in columns:
                    # We are only setup to test on categorical columns
                    if tm.orig_meta_data['column_classes'][secret] == 'continuous':
                        continue
                    aux_cols = []
                    if num_known != -1:
                        aux_cols = [col for col in columns if col not in secret]
                        if len(aux_cols) < num_known:
                            # We can't make enough aux_cols for the experiment, so just
                            # skip this secret
                            attacks_so_far += num_attacks_per_job
                            continue
                        aux_cols = random.sample(aux_cols, num_known)
                    attack_jobs.append({
                        'dir_name': dir_name,
                        'secret': secret,
                        'num_runs': num_attacks_per_job,
                        'num_known': num_known,
                        'aux_cols': aux_cols,
                    })
                    attacks_so_far += num_attacks_per_job
    # randomize the order in which the attack_jobs are run
    random.shuffle(attack_jobs)
    for index, job in enumerate(attack_jobs):
        job['index'] = index

    # Write attack_jobs into a JSON file
    with open(os.path.join(attack_path, 'attack_jobs.json'), 'w') as f:
        json.dump(attack_jobs, f, indent=4)

    exe_path = os.path.join(code_path, 'anonymeter_attack.py')
    venv_path = os.path.join(base_path, 'sdx_venv', 'bin', 'activate')
    slurm_dir = os.path.join(attack_path, 'slurm_out')
    os.makedirs(slurm_dir, exist_ok=True)
    slurm_out = os.path.join(slurm_dir, 'out.%a.out')
    num_jobs = len(attack_jobs) - 1
    # Define the slurm template
    slurm_template = f'''#!/bin/bash
#SBATCH --job-name=anonymeter_attack
#SBATCH --output={slurm_out}
#SBATCH --time=7-0
#SBATCH --mem=16G
#SBATCH --cpus-per-task=1
#SBATCH --array=0-{num_jobs}
arrayNum="${{SLURM_ARRAY_TASK_ID}}"
source {venv_path}
python {exe_path} $arrayNum
'''
    # write the slurm template to a file attack.slurm
    with open(os.path.join(attack_path, 'attack.slurm'), 'w', encoding='utf-8') as f:
        f.write(slurm_template)

def get_valid_combs(tm, secret_col, aux_cols):
    # We want the column combinations that containt secret_col and have at least
    # one other column. tm.catalog contains every subset combination that was created.
    if tm.catalog is None:
        tm.build_catalog()
    valid_combs = []
    for catalog_entry in tm.catalog:
        # check to see if every column in catalog_entry is in aux_cols
        if not all(col in aux_cols + [secret_col] for col in catalog_entry['columns']):
            continue
        if secret_col in catalog_entry['columns'] and len(catalog_entry['columns']) > 1:
            valid_combs.append(catalog_entry['columns'])
    return valid_combs

def do_inference_attacks(tm, secret_col, secret_col_type, aux_cols, regression, df_original, df_control, df_syn, num_runs):
    ''' df_original and df_control have all columns.
        df_syn has only the columns in aux_cols and secret_col.

        df_syn is the (otherwise) full syndiffix synthetic data generated from df_original.
        df_control is disjoint from df_original
    '''
    # Because I'm modeling the control and syn dataframes, and because the models
    # don't play well with string or datetime types, I'm just going to convert everthing
    df_original = convert_datetime_to_timestamp(df_original)
    df_control = convert_datetime_to_timestamp(df_control)
    df_syn = convert_datetime_to_timestamp(df_syn)
    encoders = fit_encoders([df_original, df_control, df_syn])

    df_original = transform_df(df_original, encoders)
    df_control = transform_df(df_control, encoders)
    df_syn = transform_df(df_syn, encoders)
    attack_cols = aux_cols + [secret_col]
    # model_base is the baseline built from an ML model. This is our basis for ALS.
    print("build baseline model")
    model_base = build_and_train_model(df_control[attack_cols], secret_col, secret_col_type)
    # model_attack is used to generate a groundhog day type attack
    print("build attack model")
    model_attack = build_and_train_model(df_syn[attack_cols], secret_col, secret_col_type)
    # model_original is used simply to demonstrate the ineffectiveness of the groundhog attack
    print("build original model")
    model_original = build_and_train_model(df_original[attack_cols], secret_col, secret_col_type)

    num_model_base_correct = 0
    num_model_attack_correct = 0
    num_model_original_correct = 0
    num_syn_correct = 0
    num_meter_base_correct = 0
    attacks = []
    modal_value = df_original[secret_col].mode().iloc[0]
    num_modal_rows = df_original[df_original[secret_col] == modal_value].shape[0]
    modal_percentage = round(100*(num_modal_rows / len(df_original)), 2)
    print(f"start {num_runs} runs")
    for i in range(num_runs):
        init_variants()
        print(".", end='', flush=True)
        # There is a chance of replicas here, but small enough that we ignore it
        targets = df_original[attack_cols].sample(1)
        # Get the value of the secret column in the first row of targets
        secret_value = targets[secret_col].iloc[0]
        # Count the number of rows that contian secret_value in column secret_col
        num_secret_rows = df_original[secret_col].value_counts().get(secret_value, 0)
        secret_percentage = round(100*(num_secret_rows / len(df_original)), 2)
        this_attack = {
            'secret_value': str(secret_value),
            'secret_percentage': secret_percentage,
            'secret_col': secret_col,
            'secret_col_type': secret_col_type,
            'modal_value': str(modal_value),
            'modal_percentage': modal_percentage,
            'num_known_cols': len(aux_cols),
            'known_cols': str(aux_cols),
        }
        # Now get the model baseline prediction
        try:
            model_base_pred_value = model_base.predict(targets.drop(secret_col, axis=1))
            # proba[0] is a list of probability values, indexed by the column values
            proba = model_base.predict_proba(targets.drop(secret_col, axis=1))
            model_base_pred_value = model_base_pred_value[0]
        except Exception as e:
            print(f"A model.predict() Error occurred: {e}")
            sys.exit(1)
        # convert model_base_pred_value to a series
        model_base_pred_value_series = pd.Series(model_base_pred_value, index=targets.index)
        model_base_answer = anonymeter_mods.evaluate_inference_guesses(guesses=model_base_pred_value_series, secrets=targets[secret_col], regression=regression).sum()
        if model_base_answer not in [0,1]:
            print(f"Error: unexpected answer {model_base_answer}")
            sys.exit(1)
        num_model_base_correct += model_base_answer
        this_attack['model_base_pred_value'] = str(model_base_pred_value)
        this_attack['model_base_answer'] = int(model_base_answer)
        this_attack['model_base_probability'] = float(proba[0][int(model_base_pred_value)])

        # Now run the model (groundhog) attack
        try:
            model_attack_pred_value = model_attack.predict(targets.drop(secret_col, axis=1))
            model_attack_pred_value = model_attack_pred_value[0]
        except Exception as e:
            print(f"A model.predict() Error occurred: {e}")
            sys.exit(1)
        # convert model_attack_pred_value to a series
        model_attack_pred_value_series = pd.Series(model_attack_pred_value, index=targets.index)
        model_attack_answer = anonymeter_mods.evaluate_inference_guesses(guesses=model_attack_pred_value_series, secrets=targets[secret_col], regression=regression).sum()
        if model_attack_answer not in [0,1]:
            print(f"Error: unexpected answer {model_attack_answer}")
            sys.exit(1)
        num_model_attack_correct += model_attack_answer
        this_attack['model_attack_pred_value'] = str(model_attack_pred_value)
        this_attack['model_attack_answer'] = int(model_attack_answer)

        # Now run the model attack using the groundhog model
        try:
            model_original_pred_value = model_original.predict(targets.drop(secret_col, axis=1))
            model_original_pred_value = model_original_pred_value[0]
        except Exception as e:
            print(f"A model.predict() Error occurred: {e}")
            sys.exit(1)
        # convert model_original_pred_value to a series
        model_original_pred_value_series = pd.Series(model_original_pred_value, index=targets.index)
        model_original_answer = anonymeter_mods.evaluate_inference_guesses(guesses=model_original_pred_value_series, secrets=targets[secret_col], regression=regression).sum()
        if model_original_answer not in [0,1]:
            print(f"Error: unexpected answer {model_original_answer}")
            sys.exit(1)
        num_model_original_correct += model_original_answer
        this_attack['model_original_pred_value'] = str(model_original_pred_value)
        this_attack['model_original_answer'] = int(model_original_answer)

        # Run the anonymeter-style attack on the synthetic data
        syn_meter_pred_values = []
        ans = anonymeter_mods.run_anonymeter_attack(
                                        targets=targets,
                                        basis=df_syn[attack_cols],
                                        aux_cols=aux_cols,
                                        secret=secret_col,
                                        regression=regression)
        syn_meter_pred_value_series = ans['guess_series']
        syn_meter_pred_value = syn_meter_pred_value_series.iloc[0]
        syn_meter_pred_values.append(syn_meter_pred_value)
        syn_meter_answer = anonymeter_mods.evaluate_inference_guesses(guesses=syn_meter_pred_value_series, secrets=targets[secret_col], regression=regression).sum()
        if syn_meter_answer not in [0,1]:
            print(f"Error: unexpected answer {syn_meter_answer}")
            sys.exit(1)
        num_syn_correct += syn_meter_answer
        this_attack['syn_meter_pred_value'] = str(syn_meter_pred_value)
        this_attack['syn_meter_answer'] = int(syn_meter_answer)

        # Run the anonymeter-style attack on the control data for the baseline
        ans = anonymeter_mods.run_anonymeter_attack(
                                        targets=targets,
                                        basis=df_control[attack_cols],
                                        aux_cols=aux_cols,
                                        secret=secret_col,
                                        regression=regression)
        base_meter_pred_value_series = ans['guess_series']
        base_meter_pred_value = base_meter_pred_value_series.iloc[0]
        base_meter_answer = anonymeter_mods.evaluate_inference_guesses(guesses=base_meter_pred_value_series, secrets=targets[secret_col], regression=regression).sum()
        if base_meter_answer not in [0,1]:
            print(f"Error: unexpected answer {base_meter_answer}")
            sys.exit(1)
        num_meter_base_correct += base_meter_answer
        this_attack['base_meter_pred_value'] = str(base_meter_pred_value)
        this_attack['base_meter_answer'] = int(base_meter_answer)

        # Now, we want to run the anonymeter-style attack on every valid
        # synthetic dataset. We will use this additional information to decide
        # if the anonymeter-style attack on the full dataset is correct or not.
        num_subset_combs = 0
        num_subset_correct = 0
        col_combs = get_valid_combs(tm, secret_col, aux_cols)
        #print(f"Running with total {max_subsets} of {len(col_combs)} column combinations")
        if len(col_combs) > max_subsets:
            col_combs = random.sample(col_combs, max_subsets)
        this_attack['num_subsets'] = len(col_combs)
        for col_comb in col_combs:
            # First run attack on synthetic data
            df_syn_subset = tm.get_syn_df(col_comb)
            df_syn_subset = convert_datetime_to_timestamp(df_syn_subset)
            df_syn_subset = transform_df_with_update(df_syn_subset, encoders)
            subset_aux_cols = col_comb.copy()
            subset_aux_cols.remove(secret_col)
            ans_syn = anonymeter_mods.run_anonymeter_attack(
                                            targets=targets[col_comb],
                                            basis=df_syn_subset[col_comb],
                                            aux_cols=subset_aux_cols,
                                            secret=secret_col,
                                            regression=regression)
            # Compute an answer based on the vanilla anonymeter attack
            pred_value_series = ans_syn['guess_series']
            pred_value = pred_value_series.iloc[0]
            variants['syn_meter_vanilla'].append(pred_value)

            # Compute an answer based on the modal anonymeter attack
            variants['syn_meter_modal'].append(ans_syn['match_modal_value'])	

            # Compute an answer only if the modal value is more than 50% of the possible answers
            if ans_syn['match_modal_percentage'] > 50:
                variants['syn_meter_modal_50'].append(ans_syn['match_modal_value'])

            # Compute an answer only if the modal value is more than 90% of the possible answers
            if ans_syn['match_modal_percentage'] > 90:
                variants['syn_meter_modal_90'].append(ans_syn['match_modal_value'])

            # Then run attack on control data for the baseline
            ans_base = anonymeter_mods.run_anonymeter_attack(
                                            targets=targets[col_comb],
                                            basis=df_control[col_comb],
                                            aux_cols=subset_aux_cols,
                                            secret=secret_col,
                                            regression=regression)
            # Compute an answer based on the vanilla anonymeter attack
            pred_value_series = ans_base['guess_series']
            pred_value = pred_value_series.iloc[0]
            variants['base_meter_vanilla'].append(pred_value)

            # Compute an answer based on the modal anonymeter attack
            variants['base_meter_modal'].append(ans_base['match_modal_value'])	

            # Compute an answer only if the modal value is more than 50% of the possible answers
            if ans_base['match_modal_percentage'] > 50:
                variants['base_meter_modal_50'].append(ans_base['match_modal_value'])

            # Compute an answer only if the modal value is more than 90% of the possible answers
            if ans_base['match_modal_percentage'] > 90:
                variants['base_meter_modal_90'].append(ans_base['match_modal_value'])

        if debug:
            print(f"variants:")
            pp.pprint(variants)
        # We want to filter again according to the amount of agreement among the
        # different column combinations
        for v_label, pred_values in variants.items():
            if debug:
                print(f"v_label: {v_label}")
                print(f"pred_values: {pred_values}")
            for cc_label, cc_thresh in col_comb_thresholds.items():
                label = f"{v_label}_{cc_label}"
                pred_value = find_most_frequent_value(pred_values, cc_thresh/100)
                if pred_value is not None:
                    pred_value_series = pd.Series(pred_value, index=targets.index)
                    answer = anonymeter_mods.evaluate_inference_guesses(guesses=pred_value_series, secrets=targets[secret_col], regression=regression).sum()
                else:
                    answer = -1     # no prediction
                this_attack[f'{label}_value'] = str(pred_value)
                this_attack[f'{label}_answer'] = int(answer)

        if debug:
            print(f"this_attack:")
            pp.pprint(this_attack)
        attacks.append(this_attack)
        #print('---------------------------------------------------')
        #pp.pprint(attacks[-1])
    print(f"\nnum_model_base_correct: {num_model_base_correct}\nnum_syn_correct: {num_syn_correct}\nnum_meter_base_correct: {num_meter_base_correct}\nnum_model_attack_correct: {num_model_attack_correct}\nnum_model_original_correct: {num_model_original_correct}")
    return attacks


def run_attack(job_num):
    with open(os.path.join(attack_path, 'attack_jobs.json'), 'r') as f:
        jobs = json.load(f)

    # Make sure job_num is within the range of jobs, and if not, print an error message and exit
    if job_num < 0 or job_num >= len(jobs):
        print(f"Invalid job number: {job_num}")
        return

    # Get the job
    job = jobs[job_num]

    # Create 'instances' directory in attack_path if it isn't already there
    instances_path = os.path.join(attack_path, 'instances')
    os.makedirs(instances_path, exist_ok=True)

    # Make a file_name and file_path
    file_name = f"{job['dir_name']}.{job['secret']}.{max_subsets}.{job_num}.json"
    file_path = os.path.join(instances_path, file_name)

    if os.path.exists(file_path):
        if job['num_known'] == -1:
            print(f"File already exists: {file_path}")
            return
    dataset_path = os.path.join(syn_path, job['dir_name'], 'anonymeter')
    control_path = os.path.join(dataset_path, 'control.parquet')
    # read the control file into a DataFrame
    df_control = pd.read_parquet(control_path)
    # Make a TablesManager object
    tm = TablesManager(dir_path=dataset_path)
    # First, run the attack on the full synthetic dataset
    df_syn = tm.get_syn_df()
    print(f"df_syn has shape {df_syn.shape} and columns {df_syn.columns}")
    # set aux_cols to all columns except the secret column
    aux_cols = [col for col in df_syn.columns if col not in [job['secret']]]
    if job['num_known'] != -1:
        # select num_known columns from aux_cols
        aux_cols = random.sample(aux_cols, job['num_known'])
    if tm.orig_meta_data['column_classes'][job['secret']] == 'continuous':
        regression = True
        target_type = 'continuous'
        print(f"We are no longer doing continuous secret column attacks: {job}")
        sys.exit(1)
    else:
        regression = False
        target_type = 'categorical'
    attacks = do_inference_attacks(tm, job['secret'], target_type, aux_cols, regression, tm.df_orig, df_control, df_syn, job['num_runs'])
    with open(file_path, 'w') as f:
        json.dump(attacks, f, indent=4)

def gather(instances_path):
    attacks = []
    # check to see if attacks.parquet exists
    if os.path.exists(os.path.join(attack_path, 'attacks.parquet')):
        # read it as a DataFrame
        print("Reading attacks.parquet")
        df = pd.read_parquet(os.path.join(attack_path, 'attacks.parquet'))
    else:
        all_files = list(os.listdir(instances_path))
        # loop through the index and filename of all_files
        num_files_with_num_known = {3:0, 6:0, -1:0}
        for i, filename in enumerate(all_files):
            if not filename.endswith('.json'):
                print(f"!!!!!!!!!!!!!!! bad filename: {filename}!!!!!!!!!!!!!!!!!!!!!!!")
                continue
            # split the filename on '.'
            table = filename.split('.')[0]
            with open(os.path.join(instances_path, filename), 'r') as f:
                if i % 100 == 0:
                    print(f"Reading {i+1} of {len(all_files)} {filename}")
                res = json.load(f)
                if 'num_known_cols' not in res[0]:
                    print("old format")
                    pp.pprint(res[0])
                    print(filename)
                    continue
                for record in res:
                    record['dataset'] = table
                attacks += res
                if 'num_known_cols' not in res[0]:
                    pp.pprint(res[0])
                    print(filename)
                if res[0]['num_known_cols'] == 3:
                    num_files_with_num_known[3] += len(res)
                elif res[0]['num_known_cols'] == 6:
                    num_files_with_num_known[6] += len(res)
                else:
                    num_files_with_num_known[-1] += len(res)
        print(f"Total attacks: {len(attacks)}")
        pp.pprint(num_files_with_num_known)
        # convert attacks to a DataFrame
        df = pd.DataFrame(attacks)
        for col in df.columns:
            if df[col].dtype == object:
                try:
                    df[col] = df[col].astype(int)
                except ValueError:
                    try:
                        df[col] = df[col].astype(float)
                    except ValueError:
                        pass
        # print the dtypes of df
        pp.pprint(df.dtypes)
        # save the dataframe to a parquet file
        df.to_parquet(os.path.join(attack_path, 'attacks.parquet'))
        # save the dataframe to a csv file
        df.to_csv(os.path.join(attack_path, 'attacks.csv'))
    return df

def update_max_als(max_als, max_info, label, stats):
    als_label = f"{label}_als"
    prec_label = f"{label}_precision"
    cov_label = f"{label}_coverage"
    if stats[als_label] > max_als:
        max_als = stats[als_label]
        if cov_label in stats:
            cov = stats[cov_label]
        else:
            cov = 1.0
        max_info = {'label':label, 'als':max_als, 'precision':stats[prec_label], 'coverage':cov}
    return max_als, max_info

def get_base_pred(df, target_coverage):
    df_copy = df.copy()
    df_copy = df_copy.sort_values(by='model_base_probability', ascending=False)
    num_rows = int(round(target_coverage * len(df)))
    return df_copy.head(num_rows)

def add_to_dump(df, label, slice_name):
    cols = ['secret_value', 'secret_percentage', 'secret_col', 'secret_col_type', 'modal_value', 'modal_percentage', 'num_known_cols', 'dataset',]
    val_col = f"{label}_value"
    ans_col = f"{label}_answer"
    cols += [val_col, ans_col]
    df_filtered_1 = df[df[ans_col] == 1]
    path = os.path.join(attack_path, 'dumps')
    os.makedirs(path, exist_ok=True)
    # remove spaces and "'" from slice_name
    slice_name = slice_name.replace(' ', '')
    slice_name = slice_name.replace("'", '')
    # if the slice_name is too long, truncate it
    if len(slice_name) > 50:
        slice_name = slice_name[:25] + slice_name[-25:]
    file_path = os.path.join(path, f"{slice_name}.{label}.csv")
    # check if file_path already exists
    if os.path.exists(file_path):
        return
    df_filtered_1[cols].to_csv(file_path)

def get_basic_stats(stats, df, info, cov_basis, slice_type, dataset, attack_setup, slice_name = None):
    num_zero_coverage_attacks = 0
    #print(f"get_basic_stats: {slice_type}, {dataset}, {slice_name} ({len(df)})")
    als = ALScore()
    als_threshold = 0.5
    max_als = -1000
    max_info = {}
    stats['num_attacks'] = len(df)
    stats['slice_type'] = slice_type
    stats['dataset'] = dataset
    stats['attack_setup'] = attack_setup
    stats['avg_num_subsets'] = round(df['num_subsets'].mean(), 2)
    stats['avg_secret_percentage'] = round(df['secret_percentage'].mean(), 2)
    stats['avg_modal_percentage'] = round(df['modal_percentage'].mean(), 2)
    stats['avg_num_known_cols'] = round(df['num_known_cols'].mean(), 2)
    # All of these initial measures have coverage of 1.0
    # The model_base_precision and coverage are used for the ALS calculation of SynDiffix
    stats['model_base_precision'] = round(df['model_base_answer'].sum() / len(df), 6)
    stats['model_base_coverage'] = 1.0
    stats['model_base_pcc'] = als.pcc(stats['model_base_precision'], stats['model_base_coverage'])
    # Same thing applies to the base meter prediction
    stats['meter_base_precision'] = round(df['base_meter_answer'].sum() / len(df), 6)
    stats['meter_base_coverage'] = 1.0
    stats['meter_base_pcc'] = als.pcc(stats['meter_base_precision'], stats['meter_base_coverage'])
    # We want to select the best precision-coverage-coefficient between the two baselines, the model
    # baseline and the anonymeter-style baseline (since in any event at this point the coverage is 1.0,
    # this is equivalent to selecting the best precision, but still....)
    stats['model_meter_best_pcc'] = max(stats['model_base_pcc'], stats['meter_base_pcc'])
    # Compute the ALS for the model attack on syndiffix. This is the attack based on the
    # groundhog method part 2
    stats['model_attack_precision'] = round(df['model_attack_answer'].sum() / len(df), 6)
    stats['model_attack_coverage'] = 1.0
    stats['model_attack_als'] = als.alscore(pcc_base=stats['model_meter_best_pcc'],
                                        p_attack=stats['model_attack_precision'],
                                        c_attack=stats['model_attack_coverage'])
    if stats['model_attack_als'] > als_threshold:
        stats['model_attack_problem'] = True
    else:
        stats['model_attack_problem'] = False

    # Compute the ALS for the model attack run over the original data. This has nothing
    # to do with syndiffix, rather we just want to show how ineffective the model attack is
    # We also ignore the meter_base when doing this measure
    stats['model_original_precision'] = round(df['model_original_answer'].sum() / len(df), 6)
    stats['model_original_coverage'] = 1.0
    stats['model_original_als'] = als.alscore(p_base=stats['model_base_precision'],
                                              c_base=stats['model_base_coverage'],
                                              p_attack=stats['model_original_precision'],
                                              c_attack=stats['model_original_coverage'])
    if stats['model_original_als'] > als_threshold:
        stats['model_original_problem'] = True
    else:
        stats['model_original_problem'] = False
    # Now measure the anonymeter-style attack on syndiffix.
    stats['meter_attack_precision'] = round(df['syn_meter_answer'].sum() / len(df), 6)
    stats['meter_attack_coverage'] = 1.0
    stats['meter_attack_als'] = als.alscore(pcc_base=stats['model_meter_best_pcc'],
                                            p_attack=stats['meter_attack_precision'],
                                            c_attack=stats['meter_attack_coverage'])
    if stats['meter_attack_als'] > als_threshold:
        stats['meter_attack_problem'] = True
    else:
        stats['meter_attack_problem'] = False
    max_als, max_info = update_max_als(max_als, max_info, 'meter_attack', stats)
    for v_label in variant_labels:
        for cc_label in col_comb_thresholds.keys():
            syn_meter_label = f"syn_meter_{v_label}_{cc_label}"
            syn_answer = f"{syn_meter_label}_answer"
            syn_precision = f"{syn_meter_label}_precision"
            syn_coverage = f"{syn_meter_label}_coverage"
            syn_pcc = f"{syn_meter_label}_pcc"
            syn_base_used = f"{syn_meter_label}_base_used"
            syn_base_precision = f"{syn_meter_label}_used_base_precision"
            syn_base_coverage = f"{syn_meter_label}_used_base_coverage"
            syn_base_pcc = f"{syn_meter_label}_used_base_pcc"
            syn_als = f"{syn_meter_label}_als"
            syn_problem = f"{syn_meter_label}_problem"
            base_meter_label = f"base_meter_{v_label}_{cc_label}"
            base_meter_answer = f"{base_meter_label}_answer"
            base_meter_precision = f"{base_meter_label}_precision"
            base_meter_coverage = f"{base_meter_label}_coverage"
            base_meter_pcc = f"{base_meter_label}_pcc"
            base_model_label = f"base_model_{v_label}_{cc_label}"
            base_model_precision = f"{base_model_label}_precision"
            base_model_coverage = f"{base_model_label}_coverage"
            base_model_pcc = f"{base_model_label}_pcc"
            # df_pred contains only the rows where predictions were made
            df_syn_pred = df[df[syn_answer] != -1]
            # target_coverage is the coverage we'd like to match from the base model
            target_coverage = 0 if len(df_syn_pred) == 0 else len(df_syn_pred) / len(df)
            # zzzz
            # df_base_model_pred contains the appropriate number of rows for the given
            # target coverage. We use this to base PCC
            df_base_model_pred = get_base_pred(df, target_coverage)
            # df_base_meter_pred contains the predictions (without abstains) for the given
            # threshold variant of the anonymeter attack on syndiffix
            df_base_meter_pred = df[df[base_meter_answer] != -1]
            stats[base_meter_coverage] = 0
            stats[base_meter_precision] = 0
            stats[base_meter_pcc] = 0
            stats[base_model_coverage] = 0
            stats[base_model_precision] = 0
            stats[base_model_pcc] = 0
            stats[syn_base_coverage] = 0
            stats[syn_base_precision] = 0
            stats[syn_base_pcc] = 0
            stats[syn_coverage] = 0
            stats[syn_precision] = 0
            stats[syn_pcc] = 0
            stats[syn_als] = 0
            stats[syn_problem] = False
            if len(df_syn_pred) == 0:
                num_zero_coverage_attacks += 1
                print(f"Got 0 predictions of variant {v_label} and column combination {cc_label}, length {len(df)}")
                #print(f"{slice_type}, {dataset}, {slice_name} ({len(df)})")
                continue
            # Basing the base precision on the rows where the attack happened to make predictions
            # is not necessarily the right thing to do. What we really want is to find the best base
            # precision given a similar coverage. That would give us a more accurate base precision, but
            # this is close enough.
            if len(df_base_model_pred) > 0:
                stats[base_model_precision] = df_base_model_pred[base_meter_answer].sum() / len(df_base_model_pred)
                stats[base_model_coverage] = len(df_base_model_pred) / cov_basis
                stats[base_model_pcc] = als.pcc(stats[base_model_precision], stats[base_model_coverage])
            if len(df_base_meter_pred) > 0:
                stats[base_meter_precision] = df_base_meter_pred[base_meter_answer].sum() / len(df_base_meter_pred)
                stats[base_meter_coverage] = len(df_base_meter_pred) / cov_basis
                stats[base_meter_pcc] = als.pcc(stats[base_meter_precision], stats[base_meter_coverage])
            pcc_base_use = max(stats['model_meter_best_pcc'], stats[base_model_pcc], stats[base_meter_pcc])
            # We want to record how often each type of base measure was used
            pcc_index = [stats['model_base_pcc'], stats['meter_base_pcc'],
                       stats[base_model_pcc], stats[base_meter_pcc]].index(pcc_base_use)
            pcc_name = ['num_model_base', 'num_meter_base', 'num_subset_model_base', 'num_subset_meter_base'][pcc_index]
            info[pcc_name] += 1

            # We want to record the base precision and coverage associated with the best
            # base pcc (pcc_base_use)
            p_base_use = [stats['model_base_precision'], stats['meter_base_precision'],
                          stats[base_model_precision], stats[base_meter_precision]][pcc_index]
            c_base_use = [stats['model_base_coverage'], stats['meter_base_coverage'],
                          stats[base_model_coverage], stats[base_meter_coverage]][pcc_index]
            stats[syn_base_precision] = round(p_base_use, 4)
            stats[syn_base_coverage] = round(c_base_use, 6)
            stats[syn_base_pcc] = round(pcc_base_use, 4)

            stats[syn_base_used] = pcc_name
            stats[syn_precision] = df_syn_pred[syn_answer].sum() / len(df_syn_pred)
            stats[syn_coverage] = len(df_syn_pred) / cov_basis
            stats[syn_pcc] = als.pcc(stats[syn_precision], stats[syn_coverage])
            stats[syn_als] = als.alscore(pcc_base=pcc_base_use,
                                            p_attack=stats[syn_precision],
                                            c_attack=stats[syn_coverage])
            if stats[syn_als] > als_threshold:
                stats[syn_problem] = True
                add_to_dump(df_syn_pred, syn_meter_label, slice_name)
            else:
                stats[syn_problem] = False
            max_als, max_info = update_max_als(max_als, max_info, syn_meter_label, stats)
    stats['max_als_record'] = max_info
    stats['max_als'] = 0 if 'als' not in max_info else max_info['als']
    stats['max_precision'] = 0 if 'precision' not in max_info else max_info['precision']	
    stats['max_coverage'] = 0 if 'coverage' not in max_info else max_info['coverage']	
    return num_zero_coverage_attacks

def get_by_metric_from_by_slice(stats):
    problem_cases = {}
    num_problem_cases = 0
    for metric in stats['by_slice']['all_results'].keys():
        stats['by_metric'][metric] = {}	
        for slice_key, result in stats['by_slice'].items():
            if metric in result:
                stats['by_metric'][metric][slice_key] = result[metric]
            else:
                continue
            if metric[-3:] == 'als' and metric not in ['model_original_als', 'max_als']:
                if result[metric] > 0.5:
                    num_problem_cases += 1
                    if metric[:-3] not in problem_cases:
                        problem_cases[metric[:-3]] = []
                    problem_cases[metric[:-3]].append(str((slice_key, metric, result[metric])))
                    cov_metric = metric[:-3] + 'coverage'
                    coverage = stats['by_metric'][cov_metric][slice_key]
                    problem_cases[metric[:-3]].append(str((slice_key, cov_metric, coverage)))
                    prec_metric = metric[:-3] + 'precision'
                    precision = stats['by_metric'][prec_metric][slice_key]
                    problem_cases[metric[:-3]].append(str((slice_key, prec_metric, precision)))
                    num_predictions = int(round(coverage * stats['by_metric']['num_attacks'][slice_key]))
                    problem_cases[metric[:-3]].append(str(('num_predictions', num_predictions)))
    stats['problem_cases'] = problem_cases
    stats['info']['num_problem_cases'] = num_problem_cases

def digin(df):
    df = df.copy()
    print("---------------------------------------------------")
    for low, high in [[0,10], [10,20], [20,30], [30,40], [40,50], [50,60], [60,70], [70,80], [80,90], [90,100]]:
        num_rows_high_true = df[(df['modal_value'] == df['secret_value']) & (df['modal_percentage'] > low) & (df['modal_percentage'] < high) & (df['high_syn_meter_answer'] == 1)].shape[0]
        num_rows_high_false = df[(df['modal_value'] == df['secret_value']) & (df['modal_percentage'] > low) & (df['modal_percentage'] < high) & (df['high_syn_meter_answer'] == 0)].shape[0]
        frac_true = round(100*(num_rows_high_true / (num_rows_high_true + num_rows_high_false + 0.00001)), 2)
        print(f"{low}-{high} percent true = {frac_true} ({num_rows_high_true}, {num_rows_high_false})")
    print("---------------------------------------------------")
    for low, high in [[0,10], [10,20], [20,30], [30,40], [40,50], [50,60], [60,70], [70,80], [80,90], [90,100]]:
        num_rows_true = df[(df['modal_value'] == df['secret_value']) & (df['modal_percentage'] > low) & (df['modal_percentage'] <= high)].shape[0]
        num_rows = df[(df['modal_percentage'] > low) & (df['modal_percentage'] <= high)].shape[0]
        frac_true = round(100*(num_rows_true / (num_rows + 0.00001)), 2)
        print(f"{low}-{high} precision = {frac_true} ({num_rows_true}, {num_rows})")


def df_compare(df1: pd.DataFrame, df2: pd.DataFrame) -> int:
    # Ensure the dataframes have the same columns and index
    df1, df2 = df1.align(df2)

    # Create a boolean mask where each row is True if the row in df1 differs from the row in df2
    mask = ~(df1 == df2).all(axis=1)

    # Count the number of True values in the mask
    num_differing_rows = mask.sum()

    return num_differing_rows

def set_model_base_predictions(df, thresh):
    df_copy = df.copy()
    num_pred_value_changes = 0
    num_answer_changes_to_1 = 0
    num_answer_changes_to_0 = 0
    for index, row in df_copy.iterrows():
        if row['modal_percentage'] > thresh:
            if row['model_base_pred_value'] != row['modal_value']:
                num_pred_value_changes += 1
                #print(row['model_base_pred_value'], row['modal_value'])
                df_copy.at[index, 'model_base_pred_value'] = row['modal_value']
            if row['modal_value'] == row['secret_value']:
                if row['model_base_answer'] != 1:
                    num_answer_changes_to_1 += 1
                df_copy.at[index, 'model_base_answer'] = 1
            else:
                if row['model_base_answer'] != 0:
                    num_answer_changes_to_0 += 1
                df_copy.at[index, 'model_base_answer'] = 0
    return df_copy

def run_stats_for_subsets(stats, df):
    num_zero_coverage_attacks = 0
    for bin_value, df_bin in df.groupby('attack_setup'):
        dataset = df_bin['dataset'].iloc[0]
        slice_name = f"cat_attack_setup_{bin_value}"
        stats['by_slice'][slice_name] = {}
        num_zero_coverage_attacks += get_basic_stats(stats['by_slice'][slice_name], df_bin, stats['info'], len(df_bin), 'attack_setup', dataset, bin_value, slice_name=slice_name)
    print(f"num_zero_coverage_attacks: {num_zero_coverage_attacks}")
    #df_cat_copy['percentile_bin'] = pd.cut(df_cat_copy['modal_percentage'], bins=10, labels=False)
    #df_all['percentile_bin'] = pd.cut(df_all['modal_percentage'], bins=10, labels=False)
    #get_by_metric_from_by_slice(stats)

def do_stats_dict(stats_path):
    df = gather(instances_path=os.path.join(attack_path, 'instances'))
    # print distinct values of num_subsets
    print(f"Distinct values of num_subsets: {df['num_subsets'].unique()}")
    df['known_cols'] = df['known_cols'].astype(str)

    # Define a function to create the 'attack_setup' column
    def create_attack_setup(row):
        if row['num_subsets'] == 100:
            return row['secret_col'] + row['known_cols'] + row['dataset'] + '_100'
        else:
            return row['secret_col'] + row['known_cols'] + row['dataset'] + '_200'

    # Apply the function to each row to create the 'attack_setup' column
    df['attack_setup'] = df.apply(create_attack_setup, axis=1)
    print(f"df has shape {df.shape} and columns:")
    print(df.columns)
    # count unique combinations of secret_col and dataset
    num_combinations = df.groupby(['secret_col', 'dataset']).ngroups
    print(f"Number of unique combinations of secret_col and dataset: {num_combinations}")
    for col in df.columns:
        if df[col].dtype == object:
            try:
                df[col] = df[col].astype(int)
            except ValueError:
                try:
                    df[col] = df[col].astype(float)
                except ValueError:
                    pass
    pp.pprint(df.dtypes)
    stats = {}
    # num_known_cols is the measured number of known columns (not the configuration)
    # known_cols are the actual known column names. num_subsets are the number of
    # subset synthetic tables examined
    df_all_copy = df[(df['num_known_cols'] != 3) & (df['num_known_cols'] != 6)].copy()
    for sub_key, num_known in num_known_config.items():
        if num_known != 'all':
            df_copy = df[df['num_known_cols'] == int(num_known)].copy()
        else:
            df_copy = df_all_copy
        stats[sub_key] = {'by_slice': {}, 'by_metric': {},
                          'info': {'num_model_base': 0,
                                   'num_meter_base': 0,
                                   'num_subset_model_base': 0,
                                   'num_subset_meter_base': 0,}}
        run_stats_for_subsets(stats[sub_key], df_copy)
        print(f"Writing stats {sub_key} to {stats_path}")
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=4)
    return stats

def make_df_from_stats(stats):
    collects = {
                'model_base': 'other',
                'meter_base': 'other',
                'model_attack': 'other',
                'model_original': 'other',
                'meter_attack': 'attack',
                'base_meter_vanilla_thresh_0': 'other',
                'syn_meter_vanilla_thresh_0': 'attack',
                'base_meter_vanilla_thresh_50': 'other',
                'syn_meter_vanilla_thresh_50': 'attack',
                'base_meter_vanilla_thresh_90': 'other',
                'syn_meter_vanilla_thresh_90': 'attack',
                'base_meter_modal_thresh_0': 'other',
                'syn_meter_modal_thresh_0': 'attack',
                'base_meter_modal_thresh_50': 'other',
                'syn_meter_modal_thresh_50': 'attack',
                'base_meter_modal_thresh_90': 'other',
                'syn_meter_modal_thresh_90': 'attack',
                'base_meter_modal_50_thresh_0': 'other',
                'syn_meter_modal_50_thresh_0': 'attack',
                'base_meter_modal_50_thresh_50': 'other',
                'syn_meter_modal_50_thresh_50': 'attack',
                'base_meter_modal_50_thresh_90': 'other',
                'syn_meter_modal_50_thresh_90': 'attack',
                'base_meter_modal_90_thresh_0': 'other',
                'syn_meter_modal_90_thresh_0': 'attack',
                'base_meter_modal_90_thresh_50': 'other',
                'syn_meter_modal_90_thresh_50': 'attack',
                'base_meter_modal_90_thresh_90': 'other',
                'syn_meter_modal_90_thresh_90': 'attack',
        }
    dat = []
    for num_known_key, stuff in stats.items():
        num_known = num_known_config[num_known_key]
        for slice_key, slice_results in stuff['by_slice'].items():
            for collect, info_type in collects.items():
                row = {
                        'num_known': num_known,
                        'info_type': info_type,
                        'num_attacks': slice_results['num_attacks'],
                        'slice_type': slice_results['slice_type'], 
                        'dataset': slice_results['dataset'], 
                        'attack_setup': slice_results['attack_setup'], 
                        'avg_num_subsets': slice_results['avg_num_subsets'], 
                        'avg_secret_percentage': slice_results['avg_secret_percentage'], 
                        'avg_modal_percentage': slice_results['avg_modal_percentage'], 
                }
                row['metric'] = collect
                col_prec = f"{collect}_precision"
                col_als = f"{collect}_als"
                col_coverage = f"{collect}_coverage"
                col_use_prec = f"{collect}_used_base_precision"
                row['precision'] = slice_results[col_prec]
                row['als'] = 0.0
                row['coverage'] = 1.0
                if col_use_prec in slice_results:
                    row['base_precision'] = slice_results[col_use_prec]
                    #print(f"Found {col_use_prec} in slice_results")
                else:
                    row['base_precision'] = 0.0
                    #print(f"Not Found {col_use_prec} in slice_results")
                    #print(slice_results)
                if col_als in slice_results:
                    row['als'] = slice_results[col_als]
                if col_coverage in slice_results:
                    row['coverage'] = slice_results[col_coverage]
                dat.append(row)
    df = pd.DataFrame(dat)
    return df

def plot_by_slice(df, slice, note, hue='metric'):
    df = df[df['slice_type'] == slice].copy()
    plt.figure(figsize=(8, 4))
    sns.boxplot(data=df, y=hue, x='als', orient='h', hue=hue)
    plt.xlim(-1, 1)
    plt.axvline(x=0.0, color='black', linestyle='--')
    plt.axvline(x=0.5, color='black', linestyle='--')
    plt.ylabel(hue)
    plt.xlabel(f'Anonymity Loss Score {note}')
    plt.tight_layout()
    plt.savefig(os.path.join(plots_path, f'pi_cov_by_slice_{slice}_{note}.png'))
    plt.savefig(os.path.join(plots_path, f'pi_cov_by_slice_{slice}_{note}.pdf'))

def plot_by_num_known_complete(df, note):
    from matplotlib.ticker import FixedLocator
    # Define a function to modify 'num_known' based on 'avg_num_sub_subsets'
    def modify_num_known(row):
        if row['num_known'] == 'all':
            if row['avg_num_subsets'] > 100:
                return 'all_200'
            else:
                return 'all_100'
        else:
            return row['num_known']

    # Apply the function to each row to modify the 'num_known' column
    df['num_known'] = df.apply(modify_num_known, axis=1)
    df['num_known'] = pd.Categorical(df['num_known'], categories=['3', '6', 'all_100', 'all_200'], ordered=True)
    value_counts = df['num_known'].value_counts().reindex(df['num_known'].cat.categories)
    print(value_counts)
    labels = {'3': '3', '6': '6', 'all_100': 'all\n(0-100)', 'all_200': 'all\n(101-200)'}
    tick_labels = [f"{labels[known_value]}" for known_value in value_counts.index]
    print("Number of rows for each distinct value of 'num_known':")
    print(df['num_known'].value_counts())
    plt.figure(figsize=(6, 2.5))
    ax = sns.boxplot(data=df, y='num_known', x='als', orient='h', color='lightblue')
    plt.xlim(-1, 1)
    plt.axvline(x=0.0, color='black', linestyle='--')
    plt.axvline(x=0.5, color='black', linestyle='--')
    plt.ylabel(f'Known Attributes\n(synthetic tables)')
    plt.xlabel('Anonymity Loss Coefficient (ALC)')
    ax.yaxis.set_major_locator(FixedLocator(range(len(tick_labels))))
    ax.set_yticklabels(tick_labels)
    
    # Calculate and display the percentage of 'als' values greater than 0.5 for each 'num_known'
    for i, known_value in enumerate(df['num_known'].cat.categories):
        subset = df[df['num_known'] == known_value]
        if len(subset) > 0:
            percent_greater_than_0_5 = (subset['als'] > 0.5).mean() * 100
            ax.text(1, i, f"{percent_greater_than_0_5:.1f}%", va='center', ha='right', color='black', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(os.path.join(plots_path, f'als_by_num_known_{note}.png'))
    plt.savefig(os.path.join(plots_path, f'als_by_num_known_{note}.pdf'))

def plot_prec_cov(df):
    plt.figure(figsize=(5, 3))

    plt.scatter(df['coverage'], df['precision'], marker='o', s=4)
    plt.xscale('log')
    plt.xlabel('Coverage')
    plt.ylabel('Precision')
    plt.tight_layout()
    plt.savefig(os.path.join(plots_path, f'pi_cov.png'))
    plt.savefig(os.path.join(plots_path, f'pi_cov.pdf'))

def do_plots():
    stats_path = os.path.join(attack_path, 'stats.json')
    if os.path.exists(stats_path):
        print(f"read stats from {stats_path}")
        with open(stats_path, 'r') as f:
            stats = json.load(f)
    else:
        # make a json file with the collected stats
        stats = do_stats_dict(stats_path)
    df = make_df_from_stats(stats)
    stats_df_path = os.path.join(attack_path, 'stats.parquet')
    df.to_parquet(stats_df_path)
    # df_atk contains the attacks on syndiffix (not groundhog attack etc.)
    df_atk = df[df['info_type'] == 'attack'].copy()
    # count the number of distinct attack_setup values
    print(f"There are {df_atk['attack_setup'].nunique()} distinct attack setups")
    # Count the number of distinct attack_setup values for each num_known value
    print("Distinct attack setups per num_known:")
    print(df_atk.groupby('num_known')['attack_setup'].nunique())
    stats_summary = {}
    stats_summary['precision_by_num_known_all_slices'] = df_atk.groupby('num_known')['precision'].mean().to_dict()
    stats_summary['als_by_num_known_all_slices'] = df_atk.groupby('num_known')['als'].median().to_dict()
    stats_summary['count_by_num_known_all_slices'] = df_atk.groupby('num_known')['als'].count().to_dict()
    stats_summary['precision_by_metric_all_slices'] = df_atk.groupby('metric')['precision'].mean().to_dict()
    stats_summary['als_by_metric_all_slices'] = df_atk.groupby('metric')['als'].median().to_dict()
    stats_summary['count_by_metric_all_slices'] = df_atk.groupby('metric')['als'].count().to_dict()
    df_atk_slice_all = df_atk[df_atk['slice_type'] == 'all'].copy()
    stats_summary['precision_by_num_known_slice_all_only'] = df_atk_slice_all.groupby('num_known')['precision'].mean().to_dict()
    stats_summary['als_by_num_known_slice_all_only'] = df_atk_slice_all.groupby('num_known')['als'].median().to_dict()
    stats_summary['count_by_num_known_slice_all_only'] = df_atk_slice_all.groupby('num_known')['als'].count().to_dict()
    stats_summary['precision_by_metric_slice_all_only'] = df_atk_slice_all.groupby('metric')['precision'].mean().to_dict()
    stats_summary['als_by_metric_slice_all_only'] = df_atk_slice_all.groupby('metric')['als'].median().to_dict()
    stats_summary['count_by_metric_slice_all_only'] = df_atk_slice_all.groupby('metric')['als'].count().to_dict()

    df_atk_filtered = df_atk[df_atk['base_precision'] <= 0.95].copy()
    print(f"df_atk has shape {df_atk.shape}")
    print(f"df_atk_filtered has shape {df_atk_filtered.shape}")

    df_atk_not_filtered = df_atk[df_atk['base_precision'] > 0.95].copy()
    print("Rows in df_atk but not in df_atk_filtered:")
    #print(df_atk_not_filtered[['num_known', 'info_type', 'precision', 'als', 'coverage', 'base_precision']].to_string())
    for column in df_atk.columns:
        if df_atk[column].nunique() <= 20:
            print(f"Distinct values in {column}: {df_atk[column].unique()}")

    plot_prec_cov(df_atk)
    for df_loop, loop_note in [[df_atk, '']]:
        plot_by_num_known_complete(df_loop, loop_note)

    pp.pprint(stats_summary)
    stats_summ_path = os.path.join(attack_path, 'stats_summary.json')
    # save stats_summary to a json file
    with open(stats_summ_path, 'w') as f:
        json.dump(stats_summary, f, indent=4)
    # print the distinct values of slice_type
    print(df['slice_type'].value_counts())
    print(df.columns)

def do_tests():
    if find_most_frequent_value([1, 2, 2, 3, 3, 3], 0.5) != 3:
        print("failed 1")
    if find_most_frequent_value([1, 2, 2, 3, 3, 3], 0.6) is not None:
        print("failed 2")
    if find_most_frequent_value([], 0.5) is not None:
        print("failed 3")
    if find_most_frequent_value([1, 1, 1, 1, 1], 0.2) != 1:
        print("failed 4")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", help="'config' to run make_config(), or an integer to run run_attacks()")
    args = parser.parse_args()

    if args.command == 'config':
        make_config()
    elif args.command == 'stats':
        attack_stats()
    elif args.command == 'test':
        do_tests()
    elif args.command == 'plots':
        do_plots()
    else:
        try:
            job_num = int(args.command)
            run_attack(job_num)
        except ValueError:
            print(f"Unknown command: {args.command}")

if __name__ == "__main__":
    main()