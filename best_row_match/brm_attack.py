import argparse
import sys
import os
import json
import pandas as pd
import random
import matplotlib.pyplot as plt
import seaborn as sns
import pprint
import itertools
from anonymity_loss_coefficient import BrmAttack
from anonymity_loss_coefficient.utils import get_good_known_column_sets

pp = pprint.PrettyPrinter(indent=4)
random.seed(42)
clip_val = -0.2
max_tables = [1, 20, 50, 100, 5000]

if 'SDX_TEST_DIR' in os.environ:
    base_path = os.getenv('SDX_TEST_DIR')
else:
    base_path = os.getcwd()
if 'SDX_TEST_CODE' in os.environ:
    code_path = os.getenv('SDX_TEST_CODE')
    code_path = os.path.join(code_path, 'best_row_match')
    from syndiffix_tools.tables_manager import TablesManager
else:
    code_path = None
syn_path = os.path.join(base_path, 'synDatasets')
attack_path = os.path.join(base_path, 'best_row_match')
os.makedirs(attack_path, exist_ok=True)
jobs_path = os.path.join(attack_path, 'jobs.json')
work_files_dir_path = os.path.join(attack_path, 'work_files')
os.makedirs(work_files_dir_path, exist_ok=True)
plots_dir = os.path.join(attack_path, 'plots')

orig_files_dir = os.path.join(base_path, 'original_data_parquet')

def do_attack(job_num):
    with open(jobs_path, 'r') as f:
        jobs = json.load(f)
    # get the job from the jobs list
    if job_num >= len(jobs):
        print(f"Job number {job_num} is out of range. There are only {len(jobs)} jobs.")
        sys.exit()
    job = jobs[job_num]
    print(f"Job number {job_num} started.")
    pp.pprint(job)
    df_orig = pd.read_parquet(os.path.join(orig_files_dir, job['dataset']))
    file_name = job['dataset'].split('.')[0]
    anon_path = os.path.join(syn_path, file_name, 'syn')

    anon_df_list = []
    anon_files = [f for f in os.listdir(anon_path) if f.endswith('.parquet')]
    for anon_file in anon_files:
        anon_df_list.append(pd.read_parquet(os.path.join(anon_path, anon_file)))

    attack_dir_name = f"{file_name}.{job_num}"

    my_work_files_dir = os.path.join(work_files_dir_path, attack_dir_name)
    test_file_path = os.path.join(my_work_files_dir, 'summary_secret_known.csv')
    if os.path.exists(test_file_path):
        print(f"File {test_file_path} already exists. Skipping this job.")
        return
    os.makedirs(my_work_files_dir, exist_ok=True)

    brm = BrmAttack(df_original=df_orig,
                    anon=anon_df_list,
                    results_path=my_work_files_dir,
                    attack_name = attack_dir_name,
                    max_num_anon_datasets = job['max_tables'],
                    no_counter=True,
                    )
    brm.run_one_attack(
        secret_column=job['secret_column'],
        known_columns=job['known_columns'],
    )
    print(f"Job number {job_num} FINISHED for attack {attack_dir_name}.")

class PlotsStuff:
    def __init__(self, df: pd.DataFrame, label: str):
        '''
            df_max contains one entry per group, which is the max of all alc, paired and unpaired
            df_unpaired contains one entry per group, which is the unpaired record
            df_one contains one entry per group, which is the paired record with attack_recall == 1
        '''
        self.df = df
        self.label = label
        self.df['alc'] = self.df['alc'].clip(lower=clip_val)
        idx = self.df.groupby(['secret_column', 'known_columns'])['alc'].idxmax()
        self.df_max = self.df.loc[idx].reset_index(drop=True)
        self.df_unpaired = self.df[self.df['paired'] == False].reset_index(drop=True)
        idx = self.df_unpaired.groupby(['secret_column', 'known_columns'])['alc'].idxmax()
        self.df_unpaired = self.df_unpaired.loc[idx].reset_index(drop=True)
        self.df_one = self.df[self.df['attack_recall'] == 1].reset_index(drop=True)
        idx = self.df_one.groupby(['secret_column', 'known_columns'])['alc'].idxmax()
        self.df_one = self.df_one.loc[idx].reset_index(drop=True)

        self.max_clipped = len(self.df_max[self.df_max['alc'] <= clip_val])
        self.one_clipped = len(self.df_one[self.df_one['alc'] <= clip_val])
        self.unpaired_clipped = len(self.df_unpaired[self.df_unpaired['alc'] <= clip_val])
        r_val = 2
        while True:
            self.one_clipped_frac = round(self.one_clipped / len(self.df_one), r_val)
            if self.one_clipped_frac == 0.0:
                r_val += 1
                continue
            break
        r_val = 2
        while True:
            self.unpaired_clipped_frac = round(self.unpaired_clipped / len(self.df_unpaired), r_val)
            if self.unpaired_clipped_frac == 0.0:
                r_val += 1
                continue
            break
        r_val = 2
        while True:
            self.max_clipped_frac = round(self.max_clipped / len(self.df_max), r_val)
            if self.max_clipped_frac == 0.0:
                r_val += 1
                continue
            break

    def describe(self):
        print(f"Max ALC ({self.label}):")
        print(self.df_max['alc'].describe())
        print(f"Num and frac clipped: {self.max_clipped} ({self.max_clipped_frac})")
        print(f"Unpaired ALC ({self.label}):")
        print(self.df_unpaired['alc'].describe())
        print(f"Num and frac clipped: {self.unpaired_clipped} ({self.unpaired_clipped_frac})")
        print(f"One ALC ({self.label}):")
        print(self.df_one['alc'].describe())
        print(f"Num and frac clipped: {self.one_clipped} ({self.one_clipped_frac})")

def do_plots():
    pass

def do_gather():
    out_name = f'all_secret_known.parquet'
    # List to store dataframes
    dataframes = []
    
    # Recursively walk through the directory
    for root, _, files in os.walk(work_files_dir_path):
        for file in files:
            if file == "summary_secret_known.csv":
                file_path = os.path.join(root, file)
                print(f"Reading file: {file_path}")
                try:
                    df = pd.read_csv(file_path)
                except Exception as e:
                    print(f"Error reading file {file_path}: {e}")
                    continue
                dir_name = os.path.dirname(file_path)
                dir_name = os.path.basename(dir_name)
                dir_name = dir_name.split('.')[0]
                df['dataset'] = dir_name
                dataframes.append(df)
    
    print(f"Found {len(dataframes)} files named 'summary_secret_known.csv'.")
    # Concatenate all dataframes
    if dataframes:
        combined_df = pd.concat(dataframes, ignore_index=True)
        
        # Write the combined dataframe to a parquet file
        combined_df.to_parquet(out_name, index=False)
        print(f"Parquet file written: {out_name}")
    else:
        print("No files named 'summary_secret_known.csv' were found.")

def num_attackable_tables(
    anon: list[list[str]],
    known_columns: list[str],
    secret_column: str,) -> int:
    num_attackable = 0
    for columns in anon:
        # Check if df has the secret column and at least one known column
        if secret_column not in columns:
            continue
        if not any(col in columns for col in known_columns):
            continue
        num_attackable += 1
    return num_attackable

def do_config():
    jobs = []
    files_list = os.listdir(orig_files_dir)
    for file_name in files_list:
        # read in the file
        df_orig = pd.read_parquet(os.path.join(orig_files_dir, file_name))
        file_name_prefix = file_name.split('.')[0]

        print(f"Read in anon files for {file_name_prefix}")
        anon_path = os.path.join(syn_path, file_name_prefix, 'syn')
        anon_cols_list = []
        anon_files = [f for f in os.listdir(anon_path) if f.endswith('.parquet')]
        for anon_file in anon_files:
            df = pd.read_parquet(os.path.join(anon_path, anon_file))
            anon_cols_list.append(df.columns.tolist())

        print(f"    {file_name_prefix} has {len(anon_cols_list)} tables.")
        # First populate with the cases where all columns are known
        columns = list(df_orig.columns)
        random.shuffle(columns)
        for secret_column in columns[:10]:
            # make a list with all columns except column
            other_columns = [c for c in df_orig.columns if c != secret_column]
            num_attackable = num_attackable_tables(anon_cols_list, other_columns, secret_column)
            for max_table in max_tables:
                if max_table < num_attackable:
                    jobs.append({"approach": "ours", "dataset": file_name, "known_columns": other_columns, "secret_column": secret_column, "max_tables": max_table})
                else:
                    jobs.append({"approach": "ours", "dataset": file_name, "known_columns": other_columns, "secret_column": secret_column, "max_tables": num_attackable})
                    break

        # Next populate with 5 random known column pairs
        all_column_pairs = list(itertools.combinations(df_orig.columns, 2))
        random.shuffle(all_column_pairs)
        for known_column_pair in all_column_pairs[:5]:
            # randomly select 3 secret columns that are not in known_column_pair
            columns = list(df_orig.columns)
            random.shuffle(columns)
            secret_columns = [c for c in columns if c not in known_column_pair]
            for secret_column in secret_columns[:3]:
                # make a list with all columns except known_column_pair
                num_attackable = num_attackable_tables(anon_cols_list, known_column_pair, secret_column)
                for max_table in max_tables:
                    if max_table < num_attackable:
                        jobs.append({"approach": "ours", "dataset": file_name, "known_columns": known_column_pair, "secret_column": secret_column, "max_tables": max_table})
                    else:
                        jobs.append({"approach": "ours", "dataset": file_name, "known_columns": known_column_pair, "secret_column": secret_column, "max_tables": num_attackable})
                        break

        # Next populate with 5 random known column 3-column sets
        all_column_triples = list(itertools.combinations(df_orig.columns, 3))
        random.shuffle(all_column_triples)
        for known_column_triple in all_column_triples[:5]:
            columns = list(df_orig.columns)
            random.shuffle(columns)
            secret_columns = [c for c in columns if c not in known_column_triple]
            for secret_column in secret_columns[:3]:
                # make a list with all columns except known_column_triple
                num_attackable = num_attackable_tables(anon_cols_list, known_column_triple, secret_column)
                for max_table in max_tables:
                    if max_table < num_attackable:
                        jobs.append({"approach": "ours", "dataset": file_name, "known_columns": known_column_triple, "secret_column": secret_column, "max_tables": max_table})
                    else:
                        jobs.append({"approach": "ours", "dataset": file_name, "known_columns": known_column_triple, "secret_column": secret_column, "max_tables": num_attackable})
                        break
        
        # Finally, populate with attackable (because of uniques) known column sets
        print(f"    Finding good known column sets for {file_name_prefix}")
        known_column_sets = get_good_known_column_sets(df_orig, list(df_orig.columns), max_sets=100)
        for column_set in known_column_sets:
            columns = list(df_orig.columns)
            random.shuffle(columns)
            secret_columns = [c for c in columns if c not in column_set]
            for secret_column in secret_columns[:5]:
                num_attackable = num_attackable_tables(anon_cols_list, column_set, secret_column)
                for max_table in max_tables:
                    if max_table < num_attackable:
                        jobs.append({"approach": "ours", "dataset": file_name, "known_columns": column_set, "secret_column": secret_column, "max_tables": max_table})
                    else:
                        jobs.append({"approach": "ours", "dataset": file_name, "known_columns": column_set, "secret_column": secret_column, "max_tables": num_attackable})
                        break
    random.shuffle(jobs)
    # The following tells us the distribution on the number of known columns, only for
    # planning purposes
    num_known = [0 for _ in range(20)]
    for job in jobs:
        if len(job['known_columns']) >= len(num_known):
            num_known[0] += 1
        else:
            num_known[len(job['known_columns'])] += 1
    print("Number of jobs per known columns:")
    for i, num in enumerate(num_known):
        print(f"{i} known columns: {num} jobs")

    for i, job in enumerate(jobs):
        job['job_num'] = i
    # save jobs to a json file
    with open(jobs_path, 'w') as f:
        json.dump(jobs, f, indent=4)
    exe_path = os.path.join(code_path, 'brm_attack.py')
    venv_path = os.path.join(base_path, 'sdx_venv', 'bin', 'activate')
    slurm_dir = os.path.join(attack_path, 'slurm_out')
    os.makedirs(slurm_dir, exist_ok=True)
    slurm_out = os.path.join(slurm_dir, 'out.%a.out')
    num_jobs = len(jobs) - 1
    # Define the slurm template
    slurm_template = f'''#!/bin/bash
#SBATCH --job-name=brm_attack
#SBATCH --output={slurm_out}
#SBATCH --time=7-0
#SBATCH --mem=30G
#SBATCH --cpus-per-task=1
#SBATCH --array=0-{num_jobs}
arrayNum="${{SLURM_ARRAY_TASK_ID}}"
source {venv_path}
python {exe_path} attack $arrayNum
'''
    # write the slurm template to a file attack.slurm
    with open(os.path.join(attack_path, 'attack.slurm'), 'w') as f:
        f.write(slurm_template)

def main():
    parser = argparse.ArgumentParser(description="Run attacks, plots, or configuration.")
    parser.add_argument("command", choices=["attack", "plot", "gather", "config"], help="Command to execute")
    parser.add_argument("job_num", nargs="?", type=int, help="Job number (required for 'attack')")

    args = parser.parse_args()

    if args.command == "attack":
        if args.job_num is None:
            print("Error: 'attack' command requires a job number.")
            sys.exit(1)
        do_attack(args.job_num)
    elif args.command == "plot":
        do_plots()
    elif args.command == "gather":
        for max_table in max_tables:
            print(f"Gathering work files for max_table={max_table}")
            do_gather()
    elif args.command == "config":
        do_config()

if __name__ == "__main__":
    main()