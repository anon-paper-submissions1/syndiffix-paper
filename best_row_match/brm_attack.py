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

orig_files_dir = os.path.join(base_path, 'original_data_parquet')
work_files_dir = os.path.join(attack_path, 'work_files')
os.makedirs(work_files_dir, exist_ok=True)

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
    my_work_files_dir = os.path.join(work_files_dir, attack_dir_name)
    test_file_path = os.path.join(my_work_files_dir, 'summary_secret_known.csv')
    if os.path.exists(test_file_path):
        print(f"File {test_file_path} already exists. Skipping this job.")
        return
    os.makedirs(my_work_files_dir, exist_ok=True)

    brm = BrmAttack(df_original=df_orig,
                    anon=anon_df_list,
                    results_path=my_work_files_dir,
                    attack_name = attack_dir_name,
                    no_counter=True,
                    )
    if False:
        # for debugging
        brm.run_one_attack(
            secret_column='n_unique_tokens',
            known_columns=['n_tokens_title', 'n_tokens_content',],
        )
        quit()
    # get all columns in df_orig.columns but not in job['known_columns']
    secret_columns = [c for c in df_orig.columns if c not in job['known_columns']]
    random.shuffle(secret_columns)
    print(f"Secret columns: {secret_columns}")
    print(f"Known columns: {job['known_columns']}")
    # Select 5 random secret columns for the attacks
    for secret_column in secret_columns[:5]:
        print(f"Running attack for {secret_column}...")
        brm.run_one_attack(
            secret_column=secret_column,
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
    # Read the parquet files into dataframes
    try:
        df_ours_weak = pd.read_parquet(f"all_secret_known_weak.parquet")
        df_ours_strong = pd.read_parquet(f"all_secret_known_strong.parquet")
        df_prior_weak = pd.read_parquet(f"all_secret_known_prior_weak.parquet")
        df_prior_strong = pd.read_parquet(f"all_secret_known_prior_strong.parquet")
    except Exception as e:
        print(f"Error reading parquet files: {e}")
        return

    ps_ours_weak = PlotsStuff(df_ours_weak, 'ours_weak')
    ps_ours_strong = PlotsStuff(df_ours_strong, 'ours_strong')
    ps_prior_weak = PlotsStuff(df_prior_weak, 'prior_weak')
    ps_prior_strong = PlotsStuff(df_prior_strong, 'prior_strong')

    ps_ours_weak.describe()
    ps_ours_strong.describe()
    ps_prior_weak.describe()
    ps_prior_strong.describe()

    print("Difference in computation time, ours vs prior:")
    print("   Ours weak:")
    print(ps_ours_weak.df['elapsed_time'].describe())
    print("   Prior weak:")
    print(ps_prior_weak.df['elapsed_time'].describe())
    print("   Ours strong:")
    print(ps_ours_strong.df['elapsed_time'].describe())
    print("   Prior strong:")
    print(ps_prior_strong.df['elapsed_time'].describe())

    #plot_prior_versus_ours(ps_ours_weak, ps_prior_weak, 'weak')
    plot_alc_unpaired_vs_one(ps_ours_strong, 'strong')
    plot_alc_unpaired_vs_one(ps_ours_weak, 'weak')
    plot_recall_boxes(ps_ours_weak, ps_ours_strong, 'ours')
    plot_recall_boxes(ps_prior_weak, ps_prior_strong, 'prior')
    plot_alc_ours_vs_prior(ps_ours_strong, ps_prior_strong, 'strong')
    plot_alc_ours_vs_prior(ps_ours_weak, ps_prior_weak, 'weak')
    make_tables(ps_ours_weak, ps_prior_weak)
    make_prec_prc_boxplots(ps_ours_strong, ps_prior_strong, 'strong')
    make_prec_prc_boxplots(ps_ours_weak, ps_prior_weak, 'weak')

#
#    print(f"Number of groups in 'own': {len(df_ours_unpaired)}")
#    print(f"Number of groups in 'prior': {len(df_prior_unpaired)}")
#
#    # Create a dataframe for common groups
#    print(f"Number of common groups: {len(df_common)}") 
#
#    print(f"Strength: {strength}")
#    # Get the number of rows where alc_ours < 0.5 and alc_prior > 0.5
#    alc_ours_better = df_common[(df_common['alc_ours'] < 0.5) & (df_common['alc_prior'] > 0.5)]
#    print(f"Number of rows where ALC (ours) < 0.5 and ALC (prior) > 0.5: {len(alc_ours_better)}")
#    alc_prior_better = df_common[(df_common['alc_ours'] > 0.5) & (df_common['alc_prior'] < 0.5)]
#    print(f"Number of rows where ALC (prior) < 0.5 and ALC (ours) > 0.5: {len(alc_prior_better)}")
#
#    alc_ours_much_better = df_common[(df_common['alc_ours'] < 0.5) & (df_common['alc_prior'] > 0.75)]
#    print(f"Number of rows where ALC (ours) < 0.5 and ALC (prior) > 0.75: {len(alc_ours_much_better)}")
#    alc_prior_much_better = df_common[(df_common['alc_ours'] > 0.75) & (df_common['alc_prior'] < 0.5)]
#    print(f"Number of rows where ALC (prior) < 0.5 and ALC (ours) > 0.75: {len(alc_prior_much_better)}")
#
#    print(f"Description of base PRC difference:")
#    print(df_common['prc_diff'].describe())
#    print(f"Description of base ALC difference:")
#    print(df_common['alc_diff'].describe())

def plot_prior_versus_ours(ps_ours, ps_prior, strength):
    # Process df_ours
    print(f"plot_prior_versus_ours: {strength}")

    df_common = pd.merge(
        ps_ours.df_max[['secret_column', 'known_columns', 'alc', 'base_prc']],
        ps_prior.df_max[['secret_column', 'known_columns', 'alc', 'base_prc']],
        on=['secret_column', 'known_columns'],
        suffixes=('_ours', '_prior')
    )
    df_common['prc_diff'] = df_common['base_prc_ours'] - df_common['base_prc_prior']
    df_common['alc_diff'] = df_common['alc_ours'] - df_common['alc_prior']


    # Create a scatterplot for ALC
    plt.figure(figsize=(6, 4))
    sns.scatterplot(
        data=df_common,
        x='alc_prior',
        y='alc_ours',
        alpha=0.7,
        edgecolor=None
    )
    plt.xlabel('ALC (Prior)')
    plt.ylabel('ALC (Own)')
    plt.title(f'Comparison of ALC: Prior vs Own ({strength})')
    plt.axline((0, 0), slope=1, color='red', linestyle='--', linewidth=1)  # Diagonal line
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, f'prior_versus_ours_{strength}.png'))
    plt.savefig(os.path.join(plots_dir, f'prior_versus_ours_{strength}.pdf'))
    plt.close()

    # Create a scatterplot for Baseline PRC
    plt.figure(figsize=(6, 4))
    sns.scatterplot(
        data=df_common,
        x='base_prc_prior',
        y='base_prc_ours',
        alpha=0.7,
        edgecolor=None
    )
    plt.xlabel('Best Base PRC (Prior)')
    plt.ylabel('Best Base PRC (Own)')
    plt.title(f'Comparison of Base PRC: Prior vs Own ({strength})')
    plt.axline((0, 0), slope=1, color='red', linestyle='--', linewidth=1)  # Diagonal line
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, f'prior_versus_ours_base_{strength}.png'))
    plt.savefig(os.path.join(plots_dir, f'prior_versus_ours_base_{strength}.pdf'))
    plt.close()

    # Make a boxplot for ALC
    df_melted = df_common.melt(value_vars=["alc_prior", "alc_ours"], 
                            var_name="Test Type", 
                            value_name="ALC")

    plt.figure(figsize=(6, 3))
    sns.boxplot(data=df_melted, x="ALC", y="Test Type", orient="h")
    plt.xlabel("ALC", fontsize=12)
    plt.ylabel("Test Type", fontsize=12)
    plt.title("Comparison of ALC: Prior vs Ours", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, f'prior_versus_ours_box_{strength}.png'))
    plt.savefig(os.path.join(plots_dir, f'prior_versus_ours_box_{strength}.pdf'))
    plt.close()

    # Make a boxplot for Base PRC
    df_melted = df_common.melt(value_vars=["base_prc_prior", "base_prc_ours"], 
                            var_name="Test Type", 
                            value_name="Base PRC")

    plt.figure(figsize=(6, 3))
    sns.boxplot(data=df_melted, x="Base PRC", y="Test Type", orient="h")
    plt.xlabel("Base PRC", fontsize=12)
    plt.ylabel("Test Type", fontsize=12)
    plt.title("Comparison of Base PRC: Prior vs Ours", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, f'prior_versus_ours_base_box_{strength}.png'))
    plt.savefig(os.path.join(plots_dir, f'prior_versus_ours_base_box_{strength}.pdf'))
    plt.close()

def plot_recall_boxes(ps_weak, ps_strong, whose):
    print(f"plot_recall_boxes: {whose}")

    # For each value of column halt_code in ps_strong.df_unpaired, count the rows
    # and print the counts
    print(f"Counts of halt_code in ps_strong.df_unpaired:")
    print(ps_strong.df_unpaired['halt_code'].value_counts())
    print(f"Counts of halt_code in ps_weak.df_unpaired:")
    print(ps_weak.df_unpaired['halt_code'].value_counts())

    df_merged_strong = pd.merge(
        ps_strong.df_unpaired[['secret_column', 'known_columns', 'alc']],
        ps_strong.df_one[['secret_column', 'known_columns', 'alc']],
        on=['secret_column', 'known_columns'],
        suffixes=('_unpaired', '_one')
    )
    df_merged_weak = pd.merge(
        ps_weak.df_unpaired[['secret_column', 'known_columns', 'alc']],
        ps_weak.df_one[['secret_column', 'known_columns', 'alc']],
        on=['secret_column', 'known_columns'],
        suffixes=('_unpaired', '_one')
    )
    # compute the difference of alc_unpaired and alc_one
    df_merged_strong['alc_difference'] = df_merged_strong['alc_unpaired'] - df_merged_strong['alc_one']
    df_merged_weak['alc_difference'] = df_merged_weak['alc_unpaired'] - df_merged_weak['alc_one']

    df_prc_cnt_weak = (ps_weak.df.groupby(['secret_column', 'known_columns']).size().reset_index(name='num_prc'))
    df_prc_cnt_weak['num_prc'] -= 1
    df_prc_cnt_strong = (ps_strong.df.groupby(['secret_column', 'known_columns']).size().reset_index(name='num_prc'))
    df_prc_cnt_strong['num_prc'] -= 1

    print(f"Describe num_prc for strong:")
    print(df_prc_cnt_strong['num_prc'].describe())
    print(f"Describe num_prc for weak:")
    print(df_prc_cnt_weak['num_prc'].describe())

    df_prc_cnt_weak_low = ( ps_weak.df[ps_weak.df['halt_code'] == 'extreme_low'] .groupby(['secret_column', 'known_columns']) .size() .reset_index(name='num_prc'))
    df_prc_cnt_weak_high = ( ps_weak.df[ps_weak.df['halt_code'] != 'extreme_low'] .groupby(['secret_column', 'known_columns']) .size() .reset_index(name='num_prc'))
    df_prc_cnt_strong_low = ( ps_strong.df[ps_strong.df['halt_code'] == 'extreme_low'] .groupby(['secret_column', 'known_columns']) .size() .reset_index(name='num_prc'))
    df_prc_cnt_strong_high = ( ps_strong.df[ps_strong.df['halt_code'] != 'extreme_low'] .groupby(['secret_column', 'known_columns']) .size() .reset_index(name='num_prc'))
    print(f"Describe num_prc for weak extreme_low only:")
    print(df_prc_cnt_weak_low['num_prc'].describe())
    print(f"Describe num_prc for weak all but extreme_low:")
    print(df_prc_cnt_weak_high['num_prc'].describe())
    print(f"Describe num_prc for strong extreme_low only:")
    print(df_prc_cnt_strong_low['num_prc'].describe())
    print(f"Describe num_prc for strong all but extreme_low:")
    print(df_prc_cnt_strong_high['num_prc'].describe())
    # Prepare data for recall boxplots
    df_combined_recall = pd.DataFrame({
        "Value": (
            list(ps_strong.df_unpaired["base_recall"]) +
            list(ps_strong.df_unpaired["attack_recall"]) +
            list(ps_weak.df_unpaired["attack_recall"])
        ),
        "Category": (
            ["Base recall"] * len(ps_strong.df_unpaired["base_recall"]) +
            ["Attack recall,\nstrong anon"] * len(ps_strong.df_unpaired["attack_recall"]) +
            ["Attack recall,\nweak anon"] * len(ps_weak.df_unpaired["attack_recall"])
        )
    })

    # Prepare data for difference boxplots
    df_combined_diff = pd.DataFrame({
        "Value": (
            list(df_merged_strong["alc_difference"]) +
            list(df_merged_weak["alc_difference"])
        ),
        "Category": (
            ["ALC difference,\nstrong anon"] * len(df_merged_strong["alc_difference"]) +
            ["ALC difference,\nweak anon"] * len(df_merged_weak["alc_difference"])
        )
    })

    # Create the subplots
    fig, axes = plt.subplots(2, 1, figsize=(5.5, 3.2), gridspec_kw={'height_ratios': [3, 2]}, sharex=False)

    # Plot recall boxplots
    sns.boxplot(data=df_combined_recall, x="Value", y="Category", orient="h", ax=axes[0])
    axes[0].set_xlabel("Recall (experiments with recall)", fontsize=10)
    axes[0].set_ylabel("")  # Remove y-axis label

    # Plot difference boxplots
    sns.boxplot(data=df_combined_diff, x="Value", y="Category", orient="h", ax=axes[1])
    if whose == 'ours':
        axes[1].set_xlabel("ALC Difference ours (with recall - no recall)", fontsize=10)
    else:
        axes[1].set_xlabel("ALC Difference (ours - prior)", fontsize=10)
    axes[1].set_ylabel("")  # Remove y-axis label

    # Adjust layout and save the plot
    plt.tight_layout()
    plt.savefig(f"plots/recall_and_diff_boxes_{whose}.png", dpi=300)
    plt.savefig(f"plots/recall_and_diff_boxes_{whose}.pdf", dpi=300)
    plt.close()

def make_prec_prc_boxplots(ps_ours, ps_prior, strength):
    df_merged = pd.merge(
        ps_ours.df_unpaired[['secret_column', 'known_columns', 'alc', 'base_prc', 'attack_prc', 'base_prec', 'attack_prec']],
        ps_prior.df_one[['secret_column', 'known_columns', 'alc',  'base_prc', 'attack_prc', 'base_prec', 'attack_prec']],
        on=['secret_column', 'known_columns'],
        suffixes=('_ours', '_prior')
    )
    df_merged['base_prec_diff'] = df_merged['base_prec_ours'] - df_merged['base_prec_prior']
    df_merged['base_prc_diff'] = df_merged['base_prc_ours'] - df_merged['base_prc_prior']
    df_merged['attack_prc_diff'] = df_merged['attack_prc_ours'] - df_merged['attack_prc_prior']

    # Prepare data for plotting
    df_box = pd.DataFrame({
        "Value": (
            list(df_merged["base_prec_ours"]) +
            list(df_merged["base_prc_ours"]) +
            list(df_merged["base_prc_prior"]) +
            list(df_merged["base_prc_diff"]) +
            list(df_merged["attack_prc_ours"]) +
            list(df_merged["attack_prc_prior"]) +
            list(df_merged["attack_prc_diff"]) +
            list(df_merged["alc_ours"]) +
            list(df_merged["alc_prior"])
        ),
        "Category": (
            ["Base Prec Ours"] * len(df_merged["base_prec_ours"]) +
            ["Base PRC Ours"] * len(df_merged["base_prc_ours"]) +
            ["Base Prec/PRC Prior"] * len(df_merged["base_prc_prior"]) +
            ["Base PRC Diff"] * len(df_merged["base_prc_diff"]) +
            ["Attack PRC Ours"] * len(df_merged["attack_prc_ours"]) +
            ["Attack PRC Prior"] * len(df_merged["attack_prc_prior"]) +
            ["Attack PRC Diff"] * len(df_merged["attack_prc_diff"]) +
            ["ALC Ours"] * len(df_merged["alc_ours"]) +
            ["ALC Prior"] * len(df_merged["alc_prior"])
        )
    })

    palette = {
        "Base Prec Ours": "#1f78b4",      # darker blue
        "Base PRC Ours": "#1f78b4",       # darker blue
        "Attack PRC Ours": "#1f78b4",     # darker blue
        "ALC Ours": "#1f78b4",            # darker blue
        "Base Prec/PRC Prior": "#e31a1c", # darker red
        "Attack PRC Prior": "#e31a1c",    # darker red
        "ALC Prior": "#e31a1c",           # darker red
        "Base PRC Diff": "#ff7f00",       # darker orange
        "Attack PRC Diff": "#ff7f00",     # darker orange
    }

    # Create the horizontal boxplots
    plt.figure(figsize=(5, 3))
    sns.boxplot(data=df_box, x="Value", y="Category", orient="h", palette=palette)

    plt.axvline(x=0.0, color='black', linestyle='dotted', linewidth=1.0)
    plt.xlim(-0.2, 1.05)
    plt.xlabel(f"Value ({strength} anon)")
    plt.ylabel("")  # Remove y-axis label, keep category labels

    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, f'boxplots_{strength}.png'))
    plt.savefig(os.path.join(plots_dir, f'boxplots_{strength}.pdf'))
    plt.close()


def make_tables(ps_ours_weak, ps_prior_weak):
    df_merged_recall = pd.merge(
        ps_ours_weak.df_unpaired[['secret_column', 'known_columns', 'alc']],
        ps_ours_weak.df_one[['secret_column', 'known_columns', 'alc']],
        on=['secret_column', 'known_columns'],
        suffixes=('_unpaired', '_one')
    )
    df_merged_weak = pd.merge(
        ps_ours_weak.df_unpaired[['secret_column', 'known_columns', 'alc']],
        ps_prior_weak.df_one[['secret_column', 'known_columns', 'alc']],
        on=['secret_column', 'known_columns'],
        suffixes=('_unpaired', '_one')
    )
    num_ours_safe_prior_unsafe_recall = len(df_merged_recall[(df_merged_recall['alc_unpaired'] < 0.5) & (df_merged_recall['alc_one'] >= 0.75)])
    num_ours_unsafe_prior_safe_recall = len(df_merged_recall[(df_merged_recall['alc_unpaired'] >= 0.75) & (df_merged_recall['alc_one'] < 0.5)])
    num_ours_risk_prior_safe_recall = len(df_merged_recall[(df_merged_recall['alc_unpaired'] >= 0.5) & (df_merged_recall['alc_unpaired'] < 0.75) & (df_merged_recall['alc_one'] < 0.5)])
    num_ours_safe_prior_risk_recall = len(df_merged_recall[(df_merged_recall['alc_one'] >= 0.5) & (df_merged_recall['alc_one'] < 0.75) & (df_merged_recall['alc_unpaired'] < 0.5)])

    num_ours_safe_prior_unsafe_weak = len(df_merged_weak[(df_merged_weak['alc_unpaired'] < 0.5) & (df_merged_weak['alc_one'] >= 0.75)])
    num_ours_unsafe_prior_safe_weak = len(df_merged_weak[(df_merged_weak['alc_unpaired'] >= 0.75) & (df_merged_weak['alc_one'] < 0.5)])
    num_ours_risk_prior_safe_weak = len(df_merged_weak[(df_merged_weak['alc_unpaired'] >= 0.5) & (df_merged_weak['alc_unpaired'] < 0.75) & (df_merged_weak['alc_one'] < 0.5)])
    num_ours_safe_prior_risk_weak = len(df_merged_weak[(df_merged_weak['alc_one'] >= 0.5) & (df_merged_weak['alc_one'] < 0.75) & (df_merged_weak['alc_unpaired'] < 0.5)])


    factor = 100/len(df_merged_recall)
    rnd = 2
    ours_safe_prior_unsafe_recall = round(factor * len(df_merged_recall[(df_merged_recall['alc_unpaired'] < 0.5) & (df_merged_recall['alc_one'] >= 0.75)]), rnd)
    ours_unsafe_prior_safe_recall = round(factor * len(df_merged_recall[(df_merged_recall['alc_unpaired'] >= 0.75) & (df_merged_recall['alc_one'] < 0.5)]), rnd)
    ours_risk_prior_safe_recall = round(factor * len(df_merged_recall[(df_merged_recall['alc_unpaired'] >= 0.5) & (df_merged_recall['alc_unpaired'] < 0.75) & (df_merged_recall['alc_one'] < 0.5)]), rnd)
    ours_safe_prior_risk_recall = round(factor * len(df_merged_recall[(df_merged_recall['alc_one'] >= 0.5) & (df_merged_recall['alc_one'] < 0.75) & (df_merged_recall['alc_unpaired'] < 0.5)]), rnd)

    ours_safe_prior_unsafe_weak = round(factor * len(df_merged_weak[(df_merged_weak['alc_unpaired'] < 0.5) & (df_merged_weak['alc_one'] >= 0.75)]), rnd)
    ours_unsafe_prior_safe_weak = round(factor * len(df_merged_weak[(df_merged_weak['alc_unpaired'] >= 0.75) & (df_merged_weak['alc_one'] < 0.5)]), rnd)
    ours_risk_prior_safe_weak = round(factor * len(df_merged_weak[(df_merged_weak['alc_unpaired'] >= 0.5) & (df_merged_weak['alc_unpaired'] < 0.75) & (df_merged_weak['alc_one'] < 0.5)]), rnd)
    ours_safe_prior_risk_weak = round(factor * len(df_merged_weak[(df_merged_weak['alc_one'] >= 0.5) & (df_merged_weak['alc_one'] < 0.75) & (df_merged_weak['alc_unpaired'] < 0.5)]), rnd)

    print(f"ours_safe_prior_unsafe_recall:{num_ours_safe_prior_unsafe_recall} ({ours_safe_prior_unsafe_recall})")
    print(f"ours_unsafe_prior_safe_recall:{num_ours_unsafe_prior_safe_recall} ({ours_unsafe_prior_safe_recall})")
    print(f"ours_risk_prior_safe_recall:{num_ours_risk_prior_safe_recall} ({ours_risk_prior_safe_recall})")
    print(f"ours_safe_prior_risk_recall:{num_ours_safe_prior_risk_recall} ({ours_safe_prior_risk_recall})")
    print(f"ours_safe_prior_unsafe_weak:{num_ours_safe_prior_unsafe_weak} ({ours_safe_prior_unsafe_weak})")
    print(f"ours_unsafe_prior_safe_weak:{num_ours_unsafe_prior_safe_weak} ({ours_unsafe_prior_safe_weak})")
    print(f"ours_risk_prior_safe_weak:{num_ours_risk_prior_safe_weak} ({ours_risk_prior_safe_weak})")
    print(f"ours_safe_prior_risk_weak:{num_ours_safe_prior_risk_weak} ({ours_safe_prior_risk_weak})")

    # Let's make a nice table!
    tab = f'''
\\begin{{table}}[t]
\\begin{{center}}
\\begin{{small}}
\\begin{{tabular}}{{cc|cc}}
\\toprule
 Ours & Prior & No recall & Complete \\\\
\\midrule
At risk & Safe & {ours_risk_prior_safe_recall}\\% ({num_ours_risk_prior_safe_recall}) & {ours_risk_prior_safe_weak}\\% ({num_ours_risk_prior_safe_weak}) \\\\
Serious & Safe & {ours_unsafe_prior_safe_recall}\\% ({num_ours_unsafe_prior_safe_recall}) & {ours_unsafe_prior_safe_weak}\\% ({num_ours_unsafe_prior_safe_weak}) \\\\
Safe & At risk & {ours_safe_prior_risk_recall}\\% ({num_ours_safe_prior_risk_recall}) & {ours_safe_prior_risk_weak}\\% ({num_ours_safe_prior_risk_weak}) \\\\
Safe & Serious & {ours_safe_prior_unsafe_recall}\\% ({num_ours_safe_prior_unsafe_recall}) & {ours_safe_prior_unsafe_weak}\\% ({num_ours_safe_prior_unsafe_weak}) \\\\
\\bottomrule
\\end{{tabular}}
\\caption{{Percentage (number) of attacks where the prior approach classifies anonymity incorrectly.}}
\\label{{tab:wrong_conclusion}}
\\end{{small}}
\\end{{center}}
\\end{{table}}
'''
    # Write tab to a file
    with open(f'plots/wrong_conclusion.tex', 'w') as f:
        f.write(tab)

def plot_alc_ours_vs_prior(ps_ours, ps_prior, strength):
    print(f"plot_alc_ours_vs_prior: {strength}")
    df_merged = pd.merge(
        ps_ours.df_unpaired[['secret_column', 'known_columns', 'alc', 'attack_recall']],
        ps_prior.df_one[['secret_column', 'known_columns', 'alc']],
        on=['secret_column', 'known_columns'],
        suffixes=('_ours', '_prior')
    )
    print(f"Number of rows in merged dataframe __use__: {len(df_merged)}")
    # Compute the difference of alc values
    df_merged['alc_difference'] = abs(df_merged['alc_ours'] - df_merged['alc_prior'])
    print("ps.df_ours alc:")
    print(ps_ours.df_unpaired['alc'].describe())
    print("ps_prior.df_one alc:")
    print(ps_prior.df_one['alc'].describe())
    # describe alc_difference
    print("ps_merged alc_difference (all):")
    print(df_merged['alc_difference'].describe())
    df_merged_wrong = df_merged[(df_merged['alc_ours'] > 0.5) & (df_merged['alc_prior'] < 0.5)]
    print("ps_merged alc_difference (wrong) __use__:")
    print(df_merged_wrong['alc_difference'].describe())
    print("ps_merged alc_difference (90th percentile) __use__:")
    print(df_merged['alc_difference'].quantile(0.9))

    df_merged = df_merged.sort_values(by='attack_recall', ascending=False)
    clipped_percent_ours = 100 * ps_ours.unpaired_clipped_frac
    clipped_percent_ours = int(clipped_percent_ours) if clipped_percent_ours >= 1.0 else clipped_percent_ours
    clipped_percent_prior = 100 * ps_prior.one_clipped_frac
    clipped_percent_prior = int(clipped_percent_prior) if clipped_percent_prior >= 1.0 else clipped_percent_prior
    # make a seaborn scatterplot from df_merged with alc_ours on x and alc_prior on y
    plt.figure(figsize=(5, 3))
    scatter = sns.scatterplot(
        data=df_merged,
        y='alc_ours',
        x='alc_prior',
        hue='attack_recall',  # Color points by 'attack_recall'
        palette='viridis',   # Use a colormap
        edgecolor=None,
        s=10,
        legend=False  # Remove the legend
    )

    plt.ylabel('Clipped ALC our approach')
    plt.xlabel('Clipped ALC prior approach')
    #plt.ylim(0.4, 1.05)
    #plt.xlim(-0.55, 1.05)

    # Add the shaded boxes
    plt.fill_betweenx(
        y=[0.75, 1.0], x1=0, x2=0.5, color='red', alpha=0.1, edgecolor=None
    )
    plt.fill_betweenx(
        y=[0.5, 0.75], x1=0, x2=0.5, color='orange', alpha=0.1, edgecolor=None
    )

    # draw a diagonal line from (0,0) to (1,1)
    plt.plot([0, 1], [0, 1], color='black', linestyle='dotted', linewidth=0.5)
    # draw a horizontal line at y=0.5
    plt.axhline(y=0.0, color='black', linestyle='--', linewidth=1.0)
    plt.axhline(y=0.5, color='green', linestyle='--', linewidth=1.0)
    plt.axhline(y=0.75, color='red', linestyle='--', linewidth=1.0)
    # draw a vertical line at x=0.5
    plt.axvline(x=0, color='black', linestyle='--', linewidth=1.0)
    plt.axvline(x=0.5, color='green', linestyle='--', linewidth=1.0)
    plt.axvline(x=0.75, color='red', linestyle='--', linewidth=1.0)
    plt.grid(True)
    # Add a colorbar directly from the scatterplot
    norm = plt.Normalize(df_merged['attack_recall'].min(), df_merged['attack_recall'].max())
    sm = scatter.collections[0]  # Use the scatterplot's PathCollection
    cbar = plt.colorbar(sm, label='Attack Recall our approach', orientation='vertical', pad=0.02)

    # Add clipped percentages to the plot
    plt.text(clip_val, 1, f"{clipped_percent_prior}% clipped", fontsize=8, color='black', ha='left', va='center')
    plt.text(1.05, clip_val, f"{clipped_percent_ours}% clipped", fontsize=8, color='black', ha='right', va='center')

    # tighten
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, f'alc_ours_vs_prior_{strength}.png'))
    plt.savefig(os.path.join(plots_dir, f'alc_ours_vs_prior_{strength}.pdf'))
    plt.close()

    print(f"Description of base_recall for ours PRC:")
    print(ps_ours.df_unpaired['base_recall'].describe())
    print(f"Description of attack_recall for ours PRC:")
    print(ps_ours.df_unpaired['attack_recall'].describe())

def plot_alc_unpaired_vs_one(ps, strength):
    print(f"plot_alc_unpaired_vs_one: {strength}")
    # make a new column called alc where all alc values less than -0.5 are set to -0.5
    print(f"Number of rows in 'unpaired': {len(ps.df_unpaired)}")
    print(f"Number of unpaired clipped alc values: {ps.unpaired_clipped}")
    print(f"Fraction of unpaired alc values that are clipped: {ps.unpaired_clipped_frac}")
    print(f"Number of rows in 'unpaired' after grouping: {len(ps.df_unpaired)}")
    print(f"Number of rows in 'one': {len(ps.df_one)}")
    print(f"ps.df_one after grouping: {len(ps.df_one)}")
    print(f"Number of one clipped alc values: {ps.one_clipped}")
    print(f"Fraction of one alc values that are clipped: {ps.one_clipped_frac}")
    # take the sum of base_count in ps.df_one
    print(f"Total number of predictions: {ps.df_one['base_count'].sum()}")
    print(f"Average predictions per attack __use__: {ps.df_one['base_count'].sum() / len(ps.df_one)}")
    # count the number of rows in df where both base_si and attack_si are <= 0.1, and paired is True
    df_paired = ps.df[(ps.df['base_si'] <= 0.1) & (ps.df['attack_si'] <= 0.1) & (ps.df['paired'] == True)]
    print(f"Number of significant PRC scores __use__: {len(df_paired)}")
    print(f"Average significant PRC scores per attack: {len(df_paired) / len(ps.df_one)}")
    # compute the average of num_known_columns in ps.df_unpaired
    print(f"Average number of known columns in unpaired: {ps.df_unpaired['num_known_columns'].mean()}")
    # for each combination of secret_column and known_columns, that is in both
    # ps.df_unpaired and ps.df_one, make a new dataframe df_diff that contains the difference
    # of 'alc' between ps.df_unpaired and ps.df_one
    # Merge ps.df_unpaired and ps.df_one on 'secret_column' and 'known_columns'
    df_merged = pd.merge(
        ps.df_unpaired[['secret_column', 'known_columns', 'alc', 'attack_recall']],
        ps.df_one[['secret_column', 'known_columns', 'alc']],
        on=['secret_column', 'known_columns'],
        suffixes=('_unpaired', '_one')
    )
    print(f"Number of rows in merged dataframe: {len(df_merged)}")
    # Compute the difference of alc values
    df_merged['alc_difference'] = df_merged['alc_unpaired'] - df_merged['alc_one']
    print("ps.df_unpaired alc:")
    print(ps.df_unpaired['alc'].describe())
    print("ps.df_one alc:")
    print(ps.df_one['alc'].describe())
    # describe alc_difference
    print(df_merged['alc_difference'].describe())

    # Count the number of rows where alc_unpaired > 0.75 and alc_one < 0.5
    alc_unpaired_much_better = df_merged[(df_merged['alc_unpaired'] > 0.75) & (df_merged['alc_one'] < 0.5)]
    print(f"Number of rows where ALC (unpaired) > 0.75 and ALC (one) < 0.5 ({strength}): {len(alc_unpaired_much_better)}, {len(alc_unpaired_much_better) / len(df_merged)}")
    # Count the number of rows where alc_unpaired is between 0.5 and 0.75 and alc_one < 0.5
    alc_unpaired_better = df_merged[(df_merged['alc_unpaired'] > 0.5) & (df_merged['alc_unpaired'] < 0.75) & (df_merged['alc_one'] < 0.5)]
    print(f"Number of rows where ALC (unpaired) between 0.5-0.75 and ALC (one) < 0.5 ({strength}): {len(alc_unpaired_better)}, {len(alc_unpaired_better) / len(df_merged)}")

    df_merged = df_merged.sort_values(by='attack_recall', ascending=False)
    # make a seaborn scatterplot from df_merged with alc_unpaired on x and alc_one on y
    clipped_percent_unpaired = 100 * ps.unpaired_clipped_frac
    clipped_percent_unpaired = int(clipped_percent_unpaired) if clipped_percent_unpaired >= 1.0 else clipped_percent_unpaired
    clipped_percent_one = 100 * ps.one_clipped_frac
    clipped_percent_one = int(clipped_percent_one) if clipped_percent_one >= 1.0 else clipped_percent_one
    plt.figure(figsize=(5, 3))
    scatter = sns.scatterplot(
        data=df_merged,
        y='alc_unpaired',
        x='alc_one',
        hue='attack_recall',  # Color points by 'attack_recall'
        palette='viridis',   # Use a colormap
        edgecolor=None,
        s=10,
        legend=False  # Remove the legend
    )

    plt.ylabel('Clipped ALC with recall')
    plt.xlabel('Clipped ALC no recall')
    #plt.ylim(0.4, 1.05)
    #plt.xlim(-0.55, 1.05)

    # Add the shaded boxes
    plt.fill_betweenx(
        y=[0.75, 1.0], x1=0, x2=0.5, color='red', alpha=0.1, edgecolor=None
    )
    plt.fill_betweenx(
        y=[0.5, 0.75], x1=0, x2=0.5, color='orange', alpha=0.1, edgecolor=None
    )

    # draw a diagonal line from (0,0) to (1,1)
    plt.plot([0, 1], [0, 1], color='black', linestyle='dotted', linewidth=0.5)
    # draw a horizontal line at y=0.5
    plt.axhline(y=0.0, color='black', linestyle='--', linewidth=1.0)
    plt.axhline(y=0.5, color='green', linestyle='--', linewidth=1.0)
    plt.axhline(y=0.75, color='red', linestyle='--', linewidth=1.0)
    # draw a vertical line at x=0.5
    plt.axvline(x=0, color='black', linestyle='--', linewidth=1.0)
    plt.axvline(x=0.5, color='green', linestyle='--', linewidth=1.0)
    plt.axvline(x=0.75, color='red', linestyle='--', linewidth=1.0)
    plt.grid(True)
    # Add a colorbar directly from the scatterplot
    norm = plt.Normalize(df_merged['attack_recall'].min(), df_merged['attack_recall'].max())
    sm = scatter.collections[0]  # Use the scatterplot's PathCollection
    cbar = plt.colorbar(sm, label='Attack Recall our approach', orientation='vertical', pad=0.02)

    # Add clipped percentages to the plot
    plt.text(clip_val, 1, f"{clipped_percent_one}% clipped", fontsize=8, color='black', ha='left', va='center')
    plt.text(1.05, clip_val, f"{clipped_percent_unpaired}% clipped", fontsize=8, color='black', ha='right', va='center')

    # tighten
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, f'alc_unpaired_vs_one_{strength}.png'))
    plt.savefig(os.path.join(plots_dir, f'alc_unpaired_vs_one_{strength}.pdf'))
    plt.close()

    print(f"Description of base_recall for unpaired PRC:")
    print(ps.df_unpaired['base_recall'].describe())
    print(f"Description of attack_recall for unpaired PRC:")
    print(ps.df_unpaired['attack_recall'].describe())

    # Make a boxplot for recall
    df_melted = ps.df_unpaired.melt(value_vars=["base_recall", "attack_recall"], 
                            var_name="Test Type", 
                            value_name="Recall")

    plt.figure(figsize=(6, 3))
    sns.boxplot(data=df_melted, x="Recall", y="Test Type", orient="h")
    plt.xlabel("Recall", fontsize=12)
    plt.ylabel("Test Type", fontsize=12)
    plt.title("Comparison of Recall: Baseline versus Attack", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, f'recalls_box_{strength}.png'))
    plt.savefig(os.path.join(plots_dir, f'recalls_box_{strength}.pdf'))
    plt.close()

def do_gather():
    out_name = 'all_secret_known.parquet'
    # List to store dataframes
    dataframes = []
    
    # Recursively walk through the directory
    for root, _, files in os.walk(work_files_dir):
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

def do_config():
    jobs = []
    files_list = os.listdir(orig_files_dir)
    for file_name in files_list:
        # read in the file
        df_orig = pd.read_parquet(os.path.join(orig_files_dir, file_name))
        print(f"Get known column sets for {file_name}")
        # First populate with the cases where all columns are known
        columns = list(df_orig.columns)
        random.shuffle(columns)
        for secret_column in columns[:10]:
            # make a list with all columns except column
            other_columns = [c for c in df_orig.columns if c != secret_column]
            jobs.append({"approach": "ours", "dataset": file_name, "known_columns": other_columns, "secret_column": secret_column})

        # Next populate with 25 random known column pairs
        all_column_pairs = list(itertools.combinations(df_orig.columns, 2))
        random.shuffle(all_column_pairs)
        for known_column_pair in all_column_pairs[:25]:
            # randomly select 5 secret columns that are not in known_column_pair
            columns = list(df_orig.columns)
            random.shuffle(columns)
            secret_columns = [c for c in columns if c not in known_column_pair]
            for secret_column in secret_columns[:5]:
                # make a list with all columns except known_column_pair
                jobs.append({"approach": "ours", "dataset": file_name, "known_columns": known_column_pair, "secret_column": secret_column})

        # Next populate with 25 random known column 3-column sets
        all_column_triples = list(itertools.combinations(df_orig.columns, 3))
        random.shuffle(all_column_triples)
        for known_column_triple in all_column_triples[:25]:
            columns = list(df_orig.columns)
            random.shuffle(columns)
            secret_columns = [c for c in columns if c not in known_column_triple]
            for secret_column in secret_columns[:5]:
                # make a list with all columns except known_column_triple
                jobs.append({"approach": "ours", "dataset": file_name, "known_columns": known_column_triple, "secret_column": secret_column})
        
        # Finally, populate with attackable (because of uniques) known column sets
        known_column_sets = get_good_known_column_sets(df_orig, list(df_orig.columns), max_sets=100)
        for column_set in known_column_sets:
            columns = list(df_orig.columns)
            random.shuffle(columns)
            secret_columns = [c for c in columns if c not in column_set]
            for secret_column in secret_columns[:5]:
                # make a list with all columns except column_set
                jobs.append({"approach": "ours", "dataset": file_name, "known_columns": column_set, "secret_column": secret_column})
    random.shuffle(jobs)
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
        do_gather()
    elif args.command == "config":
        do_config()

if __name__ == "__main__":
    main()