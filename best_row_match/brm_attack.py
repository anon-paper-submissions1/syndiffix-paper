import argparse
import sys
import os
import json
import pandas as pd
import random
import matplotlib.pyplot as plt
import seaborn as sns
import itertools
import pprint
from anonymity_loss_coefficient import BrmAttack
from anonymity_loss_coefficient.utils import get_good_known_column_sets, prepare_anon_list

pp = pprint.PrettyPrinter(indent=4)
random.seed(42)
clip_val = -0.2
max_tables = [1, 20, 50, 100, 5000]

if 'SDX_TEST_DIR' in os.environ:
    base_path = os.getenv('SDX_TEST_DIR')
else:
    base_path = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
if 'SDX_TEST_CODE' in os.environ:
    code_path = os.getenv('SDX_TEST_CODE')
    code_path = os.path.join(code_path, 'best_row_match')
    from syndiffix_tools.tables_manager import TablesManager
else:
    code_path = None
syn_path = os.path.join(base_path, 'synDatasets')
attack_path = os.path.join(base_path, 'best_row_match')
os.makedirs(attack_path, exist_ok=True)
plots_dir = os.path.join(attack_path, 'plots')
orig_files_dir = os.path.join(base_path, 'original_data_parquet')

# These are global values
jobs_path = os.path.join(attack_path, 'jobs.json')
work_files_dir_path = os.path.join(attack_path, 'work_files')
os.makedirs(work_files_dir_path, exist_ok=True)
summary_data_file = f'all_secret_known.parquet'
attack_slurm_dir = os.path.join(attack_path, 'attack.slurm')
slurm_flag = ''

def set_paths_all():
    global jobs_path, work_files_dir_path, summary_data_file, attack_slurm_dir, slurm_flag
    jobs_path = os.path.join(attack_path, 'jobs_all.json')
    work_files_dir_path = os.path.join(attack_path, 'work_files_all')
    os.makedirs(work_files_dir_path, exist_ok=True)
    summary_data_file = f'all_secret_known_all.parquet'
    attack_slurm_dir = os.path.join(attack_path, 'attack_all.slurm')
    slurm_flag = '--all'

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
    file_name = job['dataset'].split('.')[0]
    attack_dir_name = f"{file_name}.{job_num}"
    my_work_files_dir = os.path.join(work_files_dir_path, attack_dir_name)
    test_file_path = os.path.join(my_work_files_dir, 'summary_secret_known.csv')
    if os.path.exists(test_file_path):
        print(f"File {test_file_path} already exists. Skipping this job.")
        return

    df_orig = pd.read_parquet(os.path.join(orig_files_dir, job['dataset']))
    anon_path = os.path.join(syn_path, file_name, 'syn')
    anon_df_list, num_skipped = prepare_anon_list(anon_path, job['secret_column'], job['known_columns'])
    print(f"Prepared {len(anon_df_list)} anonymized dataset, skipped {num_skipped} datasets that did not have the secret column or known columns.")

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

def save_plot(plt, name):
    plot_path_png = os.path.join(plots_dir, f"{name}.png")
    plot_path_pdf = os.path.join(plots_dir, f"{name}.pdf")
    os.makedirs(plots_dir, exist_ok=True)
    plt.savefig(plot_path_png, bbox_inches='tight')
    plt.savefig(plot_path_pdf, bbox_inches='tight')
    plt.close()

def plot_num_datasets(df):
    """
    Plots atk_max_num_anon_datasets (y-axis) vs. index (x-axis) after sorting df by atk_max_num_anon_datasets ascending.
    """
    df_sorted = df.sort_values('atk_max_num_anon_datasets', ascending=True).reset_index(drop=True)
    plt.figure(figsize=(8, 5))
    plt.plot(df_sorted.index, df_sorted['atk_max_num_anon_datasets'], marker='o', linestyle='-')
    plt.xlabel('Index (sorted)')
    plt.ylabel('atk_max_num_anon_datasets')
    plt.yscale('log')
    plt.title('atk_max_num_anon_datasets vs. Sorted Index')
    plt.tight_layout()
    save_plot(plt, 'num_datasets')

def plot_alc_per_max_table_bins(df):
    """
    Creates a horizontal boxplot of 'alc' for each bin of 'atk_max_num_anon_datasets'.
    The x-axis is 'alc', the y-axis is the binned 'atk_max_num_anon_datasets', ordered as specified.
    The y-tick labels include the bin label and the count of datapoints in each bin.
    For each boxplot, adds a right-justified label showing the count and percent of points with alc >= 0.75.
    """
    import numpy as np

    # Define bins and labels
    bins = [1, 2, 20, 50, 100, 200, 1000]
    labels = ['1', '[2,20]', '[21,50]', '[51,100]', '[101,200]', '[201,1000]']
    
    # Bin the atk_max_num_anon_datasets column
    df = df.copy()
    df['atk_max_num_anon_datasets_bin'] = pd.cut(
        df['atk_max_num_anon_datasets'],
        bins=bins,
        labels=labels,
        right=True,
        include_lowest=True
    )
    
    # Count the number of datapoints in each bin
    bin_counts = df['atk_max_num_anon_datasets_bin'].value_counts().reindex(labels, fill_value=0)
    yticklabels = [f"{label}\n({count})" for label, count in zip(labels, bin_counts)]
    
    plt.figure(figsize=(5, 3))
    ax = sns.boxplot(
        data=df,
        x='alc',
        y='atk_max_num_anon_datasets_bin',
        order=labels,
        orient='h'
    )
    ax.set_yticklabels(yticklabels)
    plt.xlabel('ALC')
    plt.ylabel('[Number of Datasets] (count)')
    ax.axvline(x=0, color='black', linestyle='--', linewidth=1)
    ax.axvline(x=0.5, color='green', linestyle='--', linewidth=1)
    ax.axvline(x=0.75, color='red', linestyle='--', linewidth=1)
    plt.tight_layout()

    # Add right-justified text labels for each boxplot
    xlim = ax.get_xlim()
    x_right = xlim[1]
    for i, label in enumerate(labels):
        bin_df = df[df['atk_max_num_anon_datasets_bin'] == label]
        total = len(bin_df)
        cnt = (bin_df['alc'] >= 0.75).sum()
        val = 0 if total == 0 else round(100 * cnt / total, 1)
        text = f"{cnt} ({val}%)"
        # Get the y position for this boxplot
        # In seaborn, y-ticks are at positions 0, 1, ..., len(labels)-1 from top to bottom
        ytick_pos = i
        # Place text just above the boxplot axis
        ax.text(
            x_right, ytick_pos - 0.14,
            text,
            ha='right',
            va='bottom',
            fontsize=9,
            color='black',
            clip_on=False
        )
    save_plot(plt, 'alc_per_max_table_bins')

def plot_alc_per_max_table(df):
    """
    Creates a horizontal boxplot of 'alc' for each specified value/group of 'atk_max_num_anon_datasets':
    1, 20, 50, 100, and >100. The x-axis is 'alc', the y-axis is the group label.
    The y-tick labels include the group label and the count of datapoints in each group.
    For each boxplot, adds a right-justified label showing the count and percent of points with alc >= 0.75.
    """
    import numpy as np

    # Define the groups and labels
    def group_func(val):
        if val == 1:
            return "1"
        elif val == 20:
            return "20"
        elif val == 50:
            return "50"
        elif val == 100:
            return "100"
        elif val > 100:
            return ">100"
        else:
            return None

    df = df.copy()
    df['atk_max_num_anon_datasets_group'] = df['atk_max_num_anon_datasets'].apply(group_func)
    group_labels = ["1", "20", "50", "100", ">100"]

    # Filter out rows not in any group
    df = df[df['atk_max_num_anon_datasets_group'].isin(group_labels)]

    # Count the number of datapoints in each group
    group_counts = df['atk_max_num_anon_datasets_group'].value_counts().reindex(group_labels, fill_value=0)
    yticklabels = [f"{label}\n({count})" for label, count in zip(group_labels, group_counts)]

    plt.figure(figsize=(5, 3))
    ax = sns.boxplot(
        data=df,
        x='alc',
        y='atk_max_num_anon_datasets_group',
        order=group_labels,
        orient='h'
    )
    ax.set_yticklabels(yticklabels)
    plt.xlabel('ALC')
    plt.ylabel('Number of Datasets')
    ax.axvline(x=0, color='black', linestyle='--', linewidth=1)
    ax.axvline(x=0.5, color='green', linestyle='--', linewidth=1)
    ax.axvline(x=0.75, color='red', linestyle='--', linewidth=1)
    plt.tight_layout()

    # Add right-justified text labels for each boxplot
    xlim = ax.get_xlim()
    x_right = xlim[1]
    for i, label in enumerate(group_labels):
        group_df = df[df['atk_max_num_anon_datasets_group'] == label]
        total = len(group_df)
        cnt = (group_df['alc'] >= 0.75).sum()
        val = 0 if total == 0 else round(100 * cnt / total, 1)
        text = f"{cnt} ({val}%)"
        ytick_pos = i
        ax.text(
            x_right, ytick_pos - 0.14,
            text,
            ha='right',
            va='bottom',
            fontsize=9,
            color='black',
            clip_on=False
        )
    save_plot(plt, 'alc_per_max_table')

def list_bad_rows(df):
    """
    Prints details for each row in df where alc >= 0.75, sorted by alc descending.
    Columns printed: alc, base_prec, base_recall, attack_prec, attack_recall, secret_column, known_columns
    """
    bad_rows = df[df['alc'] >= 0.75].sort_values('alc', ascending=False)
    for _, row in bad_rows.iterrows():
        print(
            f"alc: {row['alc']}, "
            f"base_prec: {row['base_prec']}, "
            f"base_recall: {row['base_recall']}, "
            f"attack_prec: {row['attack_prec']}, "
            f"attack_recall: {row['attack_recall']},\n"
            f"     {row['dataset']}, "
            f"secret: {row['secret_column']}, "
            f"known: {row['known_columns']}"
        )

def do_plots(attack_all):
    df_summ = pd.read_parquet(summary_data_file)
    print(df_summ.columns)
    ps = PlotsStuff(df_summ, '')
    print(ps.describe())
    df = ps.df_unpaired
    plot_alc_per_max_table_bins(df)
    plot_num_datasets(df)
    plot_alc_per_max_table(df)
    list_bad_rows(df)
    print("Description of atk_max_num_anon_datasets")
    print(df['atk_max_num_anon_datasets'].describe())
    # get count of each unique value in model_name
    print(df['model_name'].value_counts())

def do_gather(attack_all):
    # List to store dataframes
    dataframes = []
    
    # Recursively walk through the directory
    counter = 0
    print(f"Gathering files from {work_files_dir_path}")
    for root, _, files in os.walk(work_files_dir_path):
        for file in files:
            if file == "summary_secret_known.csv":
                counter += 1
                file_path = os.path.join(root, file)
                if counter % 1000 == 0:
                    print(f"Read {counter} files")
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
        combined_df.to_parquet(summary_data_file, index=False)
        print(f"Parquet file written: {summary_data_file}")
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

def prepare_random_known_column_sets(columns):
    max_examples_per_num_known = 15
    known_column_sets = []
    columns = list(columns)
    for num_known in range(1, 6):
        # All possible combinations
        all_combos = list(itertools.combinations(columns, num_known))
        # Shuffle and select up to max_examples_per_num_known
        random.shuffle(all_combos)
        selected = all_combos[:max_examples_per_num_known]
        for combo in selected:
            known_column_sets.append(list(combo))
    return known_column_sets

def do_config(attack_all):
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
        if not attack_all:
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

        if not attack_all:
            # Populate with attackable (because of uniques) known column sets
            print(f"    Finding good known column sets for {file_name_prefix}")
            known_column_sets = get_good_known_column_sets(df_orig, list(df_orig.columns), max_sets=100)
        else:
            known_column_sets = prepare_random_known_column_sets(list(df_orig.columns))

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
python {exe_path} {slurm_flag} attack $arrayNum
'''
    # write the slurm template to a file attack.slurm
    with open(attack_slurm_dir, 'w') as f:
        f.write(slurm_template)

def main():
    parser = argparse.ArgumentParser(description="Run attacks, plots, or configuration.")
    parser.add_argument("command", choices=["attack", "plot", "gather", "config"], help="Command to execute")
    parser.add_argument("job_num", nargs="?", type=int, help="Job number (required for 'attack')")
    parser.add_argument("-a", "--all", action="store_true", help="Set attack_all to True if present")

    args = parser.parse_args()
    # The attack_all flag generates attacks for randomly selected attack configs, not just
    # the ones that are attackable by virtue of having mostly uniques.
    attack_all = args.all
    if attack_all is True:
        set_paths_all()

    if args.command == "attack":
        if args.job_num is None:
            print("Error: 'attack' command requires a job number.")
            sys.exit(1)
        # attack_all is available here if needed
        do_attack(args.job_num)
    elif args.command == "plot":
        do_plots(attack_all)
    elif args.command == "gather":
        do_gather(attack_all)
    elif args.command == "config":
        do_config(attack_all)

if __name__ == "__main__":
    main()