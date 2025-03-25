import argparse
import os
import pandas as pd
import json
import sys
import random
import numpy as np
import seaborn as sns
import alscore
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve, auc, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import itertools
import pprint

pp = pprint.PrettyPrinter(indent=4)
als = alscore.ALScore()

remove_bad_files = False
#sample_for_model = 200000
sample_for_model = None
do_comb_3_and_4 = False
num_bins = 40
win = 5000
account_for_unattackable_configurations = False

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
attack_path = os.path.join(base_path, 'suppress_attacks')
os.makedirs(attack_path, exist_ok=True)
max_attacks = 100000

def compute_metrics(df, column_name):
    # Map the labels to binary values
    mapping_true = {'tp': 1, 'tn': 0, 'fp': 0, 'fn': 1}  # 'tp' and 'fn' are positive class (1), 'tn' and 'fp' are negative class (0)
    mapping_pred = {'tp': 1, 'tn': 0, 'fp': 1, 'fn': 0}  # 'tp' and 'fp' are predicted as positive class (1), 'tn' and 'fn' are predicted as negative class (0)
    
    y_true = df[column_name].map(mapping_true)
    y_pred = df[column_name].map(mapping_pred)

    # Compute metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    return accuracy, precision, recall, f1

def prob_guess(frac_tar):
    # select a random float between 0 and 1.0
    rand = random.random()
    if rand < frac_tar:
        return 'tp'
    else:
        return 'fp'

def naive_decision(c, nkwt, nkwot):
    if c == 1 and nkwt > 0 and nkwot == 0:
        return 'tp'
    elif c == 0 and nkwt > 0 and nkwot == 0:
        return 'fp'
    elif c == 1 and (nkwt == 0 or nkwot > 0):
        return 'fn'
    elif c == 0 and (nkwt == 0 or nkwot > 0):
        return 'tn'

def model_decision(c, pos_prob):
    if c == 1 and pos_prob > 0.5:
        return 'tp'
    if c == 0 and pos_prob > 0.5:
        return 'fp'
    if c == 1 and pos_prob <= 0.5:
        return 'fn'
    elif c == 0 and  pos_prob <= 0.5:
        return 'tn'

def get_unneeded(X, needed_columns):
    unneeded = []
    for column in X.columns:
        if column not in needed_columns:
            unneeded.append(column)
    return unneeded


def build_and_add_model(X_train, X_test, y_train, y_test, X_test_all, model_stats, unneeded_columns, model_name):
    encoder = OneHotEncoder()
    scaler = StandardScaler()

    X_train_encoded = pd.get_dummies(X_train.drop(columns=unneeded_columns))
    columns = X_train_encoded.columns
    X_train_scaled = scaler.fit_transform(X_train_encoded)
    X_train = pd.DataFrame(X_train_scaled, columns=columns)

    X_test_encoded = pd.get_dummies(X_test.drop(columns=unneeded_columns))
    columns = X_test_encoded.columns
    X_test_scaled = scaler.fit_transform(X_test_encoded)
    X_test = pd.DataFrame(X_test_scaled, columns=columns)

    # Train the model
    model = LogisticRegression()
    model.fit(X_train, y_train)

    # Get the feature importance
    importance = model.coef_[0]
    # Map feature numbers to names
    feature_names = X_train.columns
    feature_importance = pd.DataFrame({'Feature': feature_names, 'Importance': importance})

    # Sort by absolute value of importance
    feature_importance['abs_importance'] = feature_importance['Importance'].abs()
    feature_importance = feature_importance.sort_values(by='abs_importance', ascending=False)
    print(model_name, feature_importance[['Feature', 'Importance']])

    # save feature_importance as a dictionary
    model_stats[model_name] = {}
    model_stats[model_name]['feature_importance'] = feature_importance.set_index('Feature')['Importance'].to_dict()

    # Predict on the test data
    y_pred = model.predict(X_test)

    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred)

    # Save metrics
    model_stats[model_name]['accuracy'] = accuracy
    model_stats[model_name]['precision'] = precision
    model_stats[model_name]['recall'] = recall
    model_stats[model_name]['f1'] = f1
    model_stats[model_name]['roc_auc'] = roc_auc
    print(f"Model: {model_name}")
    print(f"Accuracy: {accuracy}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1 Score: {f1}")
    print(f"ROC AUC: {roc_auc}")

    # Get the probability of positive class
    y_score = model.predict_proba(X_test)[:,1]

    # Add y_score into the retained copy as an additional column
    prob_col = f'prob_{model_name}'
    pred_col = f'pred_{model_name}'
    X_test_all[prob_col] = y_score

    # Apply model_decision function to get model predictions
    X_test_all[pred_col] = X_test_all.apply(lambda row: model_decision(row['c'], row[prob_col]), axis=1)

    accuracy, precision, recall, f1 = compute_metrics(X_test_all, pred_col)
    print(pred_col)
    print(f"Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1: {f1}")
    model_stats[model_name]['compute_metrics'] = {'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1': f1}

def do_model():
    # Read in the parquet file
    model_stats = {}
    res_path = os.path.join(attack_path, 'results.parquet')
    df = pd.read_parquet(res_path)
    print(f"Columns in df: {df.columns}")
    model_stats['attack_columns'] = list(df.columns)

    if sample_for_model is not None:
        df = df.sample(n=sample_for_model, random_state=42)

    # Convert 'c' column to binary
    df['c'] = df['c'].map({'positive': 1, 'negative': 0})

    # Separate features and target
    #X = df.drop(columns=['c'])
    # We are not dropping the target column c, because we want to use it later
    # when computing the non-ml approach. Rather we ignore it a scaling time
    X = df
    y = df['c']

    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

    # Retain a copy of X_test which includes all columns
    X_test_all = X_test.copy()

    # At this point, we have the following columns in X_train and X_test:
    # 'nrtv',      number rows with target value
    # 'ndtv',      number of distinct target values
    # 'c',         correct prediction (positive/negative)
    # 'nkwt',      number of synthetic rows with known values and target value
    # 'nkwot',     number of synthetic rows with known values and not target value
    # 'bs',        whether the synthetic table is the best one
    # 'nkc',       number of known columns
    # 'tp',        whether simple critieria yielded true positive
    # 'table',     the name of the synthetic table
    # 'capt',      coverage assuming specific victim and specific target attribute
    # 'cap',       coverage assuming only specific victim (any target attribute)
    # 'frac_tar',  fraction of rows with target value

    # Let's get basic stats for the naive model though
    model_stats['naive'] = {}
    X_test_all['pred_naive'] = X_test_all.apply(lambda row: naive_decision(row['c'], row['nkwt'], row['nkwot']), axis=1)
    accuracy, precision, recall, f1 = compute_metrics(X_test_all, 'pred_naive')
    print('naive')
    print(f"Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1: {f1}")
    model_stats['naive']['compute_metrics'] = {'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1': f1}

    # We are going to make three models. One model is for the purpose of establishing
    # a baseline. This model knows nrtv, ndtv, bs, nkc, and frac_tar.
    baseline_columns = ['ndtv', 'nkc', 'frac_tar', 'table']
    baseline_unneeded = get_unneeded(X, baseline_columns)
    print(f"baseline_columns: {baseline_columns}")
    print(f"baseline_unneeded: {baseline_unneeded}")
    # A second model is for an attack that only considers the attack information
    narrow_attack_columns = ['nkwt', 'nkwot']
    narrow_unneeded = get_unneeded(X, narrow_attack_columns)
    print(f"narrow_attack_columns: {narrow_attack_columns}")
    print(f"narrow_unneeded: {narrow_unneeded}")
    # A third model is for an attack that takes into account all relevant columns.
    # This includes the baseline columns plus nkwt and nkwot (the attack results).
    full_attack_columns = baseline_columns + narrow_attack_columns + ['bs']
    full_unneeded = get_unneeded(X, full_attack_columns)
    print(f"full_attack_columns: {full_attack_columns}")
    print(f"full_unneeded: {full_unneeded}")

    for unneeded_columns, model_name in [(baseline_unneeded, 'baseline'), (narrow_unneeded, 'narrow_attack'), (full_unneeded, 'full_attack')]:
        build_and_add_model(X_train.copy(), X_test.copy(), y_train.copy(), y_test.copy(), X_test_all, model_stats, unneeded_columns, model_name)
        pass

    print(X_test_all[['nrtv', 'prob_full_attack', 'prob_baseline']].head(10))
    X_test_all.to_parquet(os.path.join(attack_path, 'X_test.parquet'))
    # write model_stats to json file
    with open(os.path.join(attack_path, 'model_stats.json'), 'w') as f:
        json.dump(model_stats, f, indent=4)

def make_bin_scatterplot(df_bin, color_by, label, filename, pi_floor):
    plt.figure(figsize=(8, 4))
    plt.scatter(df_bin['frac_perfect'], df_bin['pi_fl_mid'], c=df_bin[color_by], cmap='viridis', marker='o')
    plt.scatter(df_bin['frac_capt'], df_bin['pi_fl_mid'], c=df_bin[color_by], cmap='viridis', marker='x')
    #plt.scatter(df_bin['frac_perfect'], df_bin['pi_fl_mid'], c=df_bin[color_by], cmap='viridis')
    plt.colorbar(label=label)
    plt.xscale('log')
    plt.hlines(0.5, 0.001, 1, colors='black', linestyles='--')
    plt.vlines(0.001, 0.5, 1.0, colors='black', linestyles='--')
    plt.xlabel('Coverage (log scale)', fontsize=13, labelpad=10)
    plt.ylabel(f'Precision Improvement\n(floored at {pi_floor})', fontsize=13, labelpad=10)

    # Create custom legend
    legend_elements = [
        Line2D([0], [0], marker='o', color='black', markerfacecolor='black', markersize=8, label='Attack conditions\nhappen to exist', linestyle='None'),
        Line2D([0], [0], marker='x', color='black', markerfacecolor='black', markersize=8, label='Attack specific person\nand target', linestyle='None')]
    plt.legend(handles=legend_elements, loc='lower left', fontsize=7)
    plt.tight_layout()
    filename += f"_{num_bins}.png"
    plot_path = os.path.join(attack_path, filename)
    plt.savefig(plot_path)
    plt.close()

def plot_move_avg(df):
    # Sort the DataFrame by the 'pi_fl' column in descending order
    df_sorted = df.sort_values('pi_fl', ascending=False).reset_index(drop=True)

    df_sorted['moving_avg_pi_fl'] = df_sorted['pi_fl'].rolling(window=win).mean()
    df_sorted['moving_avg_capt'] = df_sorted['capt'].rolling(window=win).mean()
    df_sorted['moving_avg_frac_tar'] = df_sorted['frac_tar'].rolling(window=win).mean()

    # Drop rows with NaN values in 'moving_avg' columns
    df_sorted = df_sorted.dropna(subset=['moving_avg_pi_fl', 'moving_avg_capt', 'moving_avg_frac_tar'])

    # Compute the CDF
    df_sorted['cdf'] = (df_sorted.index + 1) / len(df_sorted)
    df_sorted['cdf_to_capt'] = df_sorted['cdf'] * df_sorted['moving_avg_capt']

    # Plot the pi_fl moving average against the CDF
    plt.figure(figsize=(6, 3.5))
    scatter1 = plt.scatter(df_sorted['cdf'], df_sorted['moving_avg_pi_fl'], c=df_sorted['moving_avg_frac_tar'], label="Attack conditions\nhappen to exist", s=2)
    scatter2 = plt.scatter(df_sorted['cdf_to_capt'], df_sorted['moving_avg_pi_fl'], c=df_sorted['moving_avg_frac_tar'], label="Attacker has specific\nvictim and target", s=2)
    #plt.plot(df_sorted['cdf'], df_sorted['moving_avg_pi_fl'], label="Attack conditions\nhappen to exist")
    #plt.plot(df_sorted['cdf_to_capt'], df_sorted['moving_avg_pi_fl'], label="Attacker has specific\nvictim and target")
    plt.hlines(0.5, 0.001, 1, colors='black', linestyles='--')
    plt.vlines(0.001, 0.5, 1.0, colors='black', linestyles='--')
    plt.text(0.03, 0.35, "Attack conditions\nhappen to exist", fontsize=8, bbox=dict(facecolor='white', alpha=0.5, edgecolor='none'))
    plt.text(0.000002, 0.05, "Attacker has specific\nvictim and target", fontsize=8, bbox=dict(facecolor='white', alpha=0.5, edgecolor='none'))
    cbar = plt.colorbar(scatter1)
    cbar.set_label('Target Value Fraction', rotation=270, labelpad=15)
    plt.xscale('log')
    plt.xlabel('Coverage as Cumulative Probability', fontsize=13, labelpad=10)
    plt.ylabel(f'Precision Improvement\n(rolling window={win})', fontsize=13, labelpad=10)
    plt.tight_layout()
    plt.savefig(os.path.join(attack_path, 'pi_fl_mv_avg.png'))
    plt.savefig(os.path.join(attack_path, 'pi_fl_mv_avg.pdf'))
    plt.close()
    print("done with pi_fl_mv_avg.png")

    # Plot the frac_tar moving average against the CDF
    plt.figure(figsize=(10, 6))
    plt.plot(df_sorted['cdf'], df_sorted['moving_avg_frac_tar'])
    plt.xscale('log')
    plt.xlabel('Coverage')
    plt.ylabel(f'Moving Average Fraction Target Rows\n(floored at 0, window={win})')
    plt.tight_layout()
    plt.savefig(os.path.join(attack_path, 'frac_tar_mv_avg.png'))
    plt.savefig(os.path.join(attack_path, 'frac_tar_mv_avg.pdf'))
    plt.close()

def alc_cap(df):
    # Sort the DataFrame by 'cap' ascending
    df = df.sort_values(by='cap', ascending=True)

    # Initialize the list to store alc values
    alc_list = []

    # Loop through the DataFrame in groups of 1000 rows
    group_size = 1000
    for i in range(0, len(df), group_size):
        group = df.iloc[i:i+group_size]

        # Compute the required values
        cov_avg = group['cap'].mean()
        p_base = len(group[group['pred_baseline'] == 'tp']) / len(group[(group['pred_baseline'] == 'tp') | (group['pred_baseline'] == 'fp')])
        cov_base = len(group[(group['pred_baseline'] == 'tp') | (group['pred_baseline'] == 'fp')]) / group_size
        c_base = cov_base
        p_atk = len(group[group['pred_full_attack'] == 'tp']) / len(group[(group['pred_full_attack'] == 'tp') | (group['pred_full_attack'] == 'fp')])
        cov_atk = len(group[(group['pred_full_attack'] == 'tp') | (group['pred_full_attack'] == 'fp')]) / group_size
        if account_for_unattackable_configurations:
            c_atk = cov_atk * cov_avg
        else:
            c_atk = cov_atk

        # Compute alc
        alc = als.alscore(p_base=p_base, c_base=c_base, p_attack=p_atk, c_attack=c_atk)

        # Add alc to the list
        alc_list.append(alc)

    # Display alc_list as a boxplot
    plt.figure(figsize=(6.0, 1.5))
    sns.boxplot(data=alc_list, orient='h', color='lightblue')
    plt.xlim(-0.05, 1.05)
    plt.axvline(x=0.5, color='black', linestyle='--')
    plt.xlabel('Anonymity Loss Coefficient (ALC)')
    plt.ylabel('Real\ndatasets')
    plt.tight_layout()
    plot_path = os.path.join(attack_path, 'alc_cap.png')
    plt.savefig(plot_path)
    plot_path = os.path.join(attack_path, 'alc_cap.pdf')
    plt.savefig(plot_path)
    plt.close()
    print("done with alc_cap.png")


def alc_capt(df):
    # Sort the DataFrame by 'frac_tar' ascending
    #df = df.sort_values(by='capt', ascending=True)
    df = df.sort_values(by='frac_tar', ascending=True)

    # Initialize the list to store alc values
    alc_list = []
    alc_list2 = []

    # Loop through the DataFrame in groups of 10000 rows
    group_size = 1000
    for i in range(0, len(df), group_size):
        group = df.iloc[i:i+group_size]

        # Compute the required values
        cov_avg = group['capt'].mean()
        num_base_predicts = len(group[(group['pred_baseline'] == 'tp') | (group['pred_baseline'] == 'fp')])
        if num_base_predicts == 0:
            continue
        p_base = len(group[group['pred_baseline'] == 'tp']) / num_base_predicts
        cov_base = len(group[(group['pred_baseline'] == 'tp') | (group['pred_baseline'] == 'fp')]) / group_size
        c_base = cov_base
        num_atk_predicts = len(group[(group['pred_full_attack'] == 'tp') | (group['pred_full_attack'] == 'fp')])
        if num_atk_predicts == 0:
            continue
        p_atk = len(group[group['pred_full_attack'] == 'tp']) / num_atk_predicts
        cov_atk = len(group[(group['pred_full_attack'] == 'tp') | (group['pred_full_attack'] == 'fp')]) / group_size
        c_atk2 = cov_atk * cov_avg
        c_atk = cov_atk
        alc2 = als.alscore(p_base=p_base, c_base=c_base, p_attack=p_atk, c_attack=c_atk2)
        alc = als.alscore(p_base=p_base, c_base=c_base, p_attack=p_atk, c_attack=c_atk)
        alc_list2.append(alc2)
        alc_list.append(alc)
        if alc > 0.35:
            frac_tar_avg = group['frac_tar'].mean()
            print(f"frac_tar: {frac_tar_avg}, p_base: {p_base}, c_base: {c_base}, p_attack: {p_atk}, c_attack: {c_atk}, alc: {alc}")

    print(f"There are {len(alc_list)} datapoints in the alc_capt boxplot")
    # Display alc_list as a boxplot
    data = {
        'ALC': alc_list + alc_list2,
        'Dataset': ['Attacks with\ncondition only'] * len(alc_list) + ['All attacks'] * len(alc_list2)
    }
    df = pd.DataFrame(data)
    
    plt.figure(figsize=(7.0, 2.0))
    sns.boxplot(data=df, x='ALC', y='Dataset', orient='h', palette='light:#5A9', hue='Dataset')
    plt.xlim(-0.05, 1.05)
    plt.axvline(x=0.5, color='black', linestyle='--')
    plt.xlabel('Anonymity Loss Coefficient (ALC)')
    plt.ylabel('Real datasets')
    plt.tight_layout()
    plot_path = os.path.join(attack_path, 'alc_capt.png')
    plt.savefig(plot_path)
    plot_path = os.path.join(attack_path, 'alc_capt.pdf')
    plt.savefig(plot_path)
    plt.close()
    print("done with alc_capt.png")

def do_plots():
    # Read in the parquet files
    X_test_all = pd.read_parquet(os.path.join(attack_path, 'X_test.parquet'))

    X_test_all['pi'] = (X_test_all['prob_full_attack'] - X_test_all['prob_baseline']) / (1.000001 - X_test_all['prob_baseline'])
    # This makes up for the use of 1.000001 in the above line
    X_test_all.loc[X_test_all['pi'] >= 0.9999, 'pi'] = 1.0
    pi_floor = 0
    X_test_all['pi_fl'] = X_test_all['pi'].clip(lower=pi_floor)
    print(X_test_all[['nrtv', 'prob_full_attack', 'prob_baseline', 'pi', 'pi_fl']].head(10))

    # print distributions
    print("Distribution of capt:")
    print(X_test_all['capt'].describe())
    # Number of distinct capt values
    print(f"Number of distinct capt values: {X_test_all['capt'].nunique()}")
    print("Distribution of cap:")
    print(X_test_all['cap'].describe())
    print("Distribution of pi_fl:")
    print(X_test_all['pi_fl'].describe())
    print("Distribution of prob_full_attack:")
    print(X_test_all['prob_full_attack'].describe())
    print("Distribution of prob_baseline:")
    print(X_test_all['prob_baseline'].describe())
    avg_capt = X_test_all['capt'].mean()
    print(f"Average capt: {avg_capt}")
    avg_cap = X_test_all['cap'].mean()
    print(f"Average cap: {avg_cap}")

    print("X_test:")
    print(X_test_all.head())
    print(f"Total rows: {X_test_all.shape[0]}")
    print(X_test_all.columns)
    # make a copy of X_test_all where column sp is True
    X_test_all_sp = X_test_all[X_test_all['bs'] == True].copy()
    print(f"There are {X_test_all.shape[0]} rows in X_test, and {X_test_all_sp.shape[0]} rows where bs is True")

    # Make a copy of X_test_all_sp where frac_tar < 0.2
    X_test_all_sp_lt_02 = X_test_all_sp[X_test_all_sp['frac_tar'] < 0.2].copy()
    base_tp = X_test_all_sp_lt_02[X_test_all_sp_lt_02['pred_baseline'] == 'tp'].shape[0]
    base_fp = X_test_all_sp_lt_02[X_test_all_sp_lt_02['pred_baseline'] == 'fp'].shape[0]
    prec_base = base_tp / (base_tp + base_fp)
    full_tp = X_test_all_sp_lt_02[X_test_all_sp_lt_02['pred_full_attack'] == 'tp'].shape[0]
    full_fp = X_test_all_sp_lt_02[X_test_all_sp_lt_02['pred_full_attack'] == 'fp'].shape[0]
    prec_full = full_tp / (full_tp + full_fp)
    print(f"Precision of full_attack for frac_tar < 0.2: {prec_full}")
    print(f"Precision of baseline for frac_tar < 0.2: {prec_base}")

    alc_capt(X_test_all_sp.copy())
    alc_cap(X_test_all_sp.copy())

    # count 'tp' in 'prob_guess'
    X_test_all_sp_lt_02['prob_guess'] = X_test_all_sp_lt_02.apply(lambda row: prob_guess(row['frac_tar']), axis=1)
    prob_tp = X_test_all_sp_lt_02[X_test_all_sp_lt_02['prob_guess'] == 'tp'].shape[0]
    prec_prob = prob_tp / X_test_all_sp_lt_02.shape[0]
    print(f"Precision of prob_guess: {prec_prob}")
    print(f"Precision of baseline: {prec_base}")

    # Count the number of rows where pi_fl == 1
    count_pi_fl_1 = X_test_all[X_test_all['pi_fl'] == 1].shape[0]
    print(f"Count of rows where pi_fl == 1: {count_pi_fl_1}")

    X_test_all_sorted = X_test_all.sort_values(by='pi_fl', ascending=False).reset_index(drop=True)

    X_test_all_sorted['prob_perfect'] = (X_test_all_sorted.index + 1) / len(X_test_all_sorted)
    X_test_all_sorted['prob_combs_targets'] = X_test_all_sorted['prob_perfect'] * avg_capt
    X_test_all_sorted['prob_combs'] = X_test_all_sorted['prob_perfect'] * avg_cap
    df_plot = X_test_all_sorted.reset_index(drop=True)

    # Plot 'probability' vs 'pi_fl'
    plt.figure(figsize=(6, 3))
    plt.plot(df_plot['prob_perfect'], df_plot['pi_fl'], label='Attack conditions exist')
    plt.plot(df_plot['prob_combs'], df_plot['pi_fl'], label='Attacker knowledge, any target ok')
    plt.plot(df_plot['prob_combs_targets'], df_plot['pi_fl'], label='Attacker knowledge, only specific target')
    plt.xscale('log')
    plt.hlines(0.5, 0.001, 1, colors='black', linestyles='--', linewidth=0.5)
    plt.vlines(0.001, 0.5, 1.0, colors='black', linestyles='--', linewidth=0.5)
    plt.xlabel('Coverage (log)')
    plt.ylabel(f'Precision Improvement\n(floored at {pi_floor})')
    plt.legend(loc="lower left", prop={'size': 8})
    plt.tight_layout()
    plt.savefig(os.path.join(attack_path, 'pi_cov.png'))
    plt.savefig(os.path.join(attack_path, 'pi_cov.pdf'))
    plt.close()

    plot_move_avg(X_test_all.copy())

def gather(instances_path):
    all_entries = []
    
    datasets = set()
    gather_stats = {'num_possible_combs_targets': 0,
                    'num_possible_combs': 0,
                    'num_datasets': 0,
                    'num_attacks': 0,
                    'num_positive': 0,
                    'num_negative': 0,
                    'num_best_syn': 0,
                    'num_known_columns': [0,0,0,0,0,0]
                    }
    num_fail = 0
    # Step 1: Read in all of the json files in the directory at instances_path
    all_files = list(os.listdir(instances_path))
    # loop through the index and filename of all_files
    for i, filename in enumerate(all_files):
        if filename.endswith('.json'):
            with open(os.path.join(instances_path, filename), 'r') as f:
                print(f"Reading {i+1} of {len(all_files)} {filename}")
                file_length = os.path.getsize(f.name)
                if file_length < 10:
                    # Can happen if the file is still under construction
                    print(f"---- File {filename} is too short, skipping")
                    continue
                try:
                    res = json.load(f)
                    capt = res['summary']['coverage_all_combs_targets']
                    gather_stats['num_possible_combs_targets'] += res['summary']['num_possible_combs_targets']
                    gather_stats['num_possible_combs'] += res['summary']['num_possible_combs']
                    cap = res['summary']['coverage_all_combs']
                    gather_stats['num_attacks'] += res['summary']['num_attacks']
                    datasets.add(res['summary']['job']['dir_name'])
                    num_rows = res['summary']['num_rows']
                    for entry in res['attack_results']:
                        entry['table'] = res['summary']['job']['dir_name']
                        entry['capt'] = capt
                        entry['cap'] = cap
                        entry['frac_tar'] = entry['nrtv'] / num_rows
                        if entry['c'] == 'positive':
                            gather_stats['num_positive'] += 1
                        else:
                            gather_stats['num_negative'] += 1
                        if entry['bs'] is True:
                            gather_stats['num_best_syn'] += 1
                        gather_stats['num_known_columns'][entry['nkc']] += 1
                        all_entries.append(entry)
                except json.JSONDecodeError:
                    num_fail += 1
                    if remove_bad_files:
                        print(f"---- Error reading {filename}, deleting")
                        os.remove(os.path.join(instances_path, filename))
                    else:
                        print(f"---- Error reading {filename}, skipping")
                    continue
                
    # Step 4: Make a dataframe df where each key in each entry of all_entries is a column
    df = pd.DataFrame(all_entries)
    print(df.head())
    
    # Step 5: Store df as a parquet file called results.parquet
    file_path = os.path.join(attack_path, 'results.parquet')
    df.to_parquet(file_path)
    print(f"{num_fail} files were corrupted and deleted")

    gather_stats['num_datasets'] = len(datasets)
    gather_stats['percent_positive'] = round(100 * (gather_stats['num_positive'] / gather_stats['num_attacks']),2)
    gather_stats['percent_best_syn'] = round(100 * (gather_stats['num_best_syn'] / gather_stats['num_attacks']),2)
    gather_stats['percent_known_columns'] = [round(100 * (x / gather_stats['num_attacks']),2) for x in gather_stats['num_known_columns']]
    with open(os.path.join(attack_path, 'gather_stats.json'), 'w') as f:
        json.dump(gather_stats, f, indent=4)

def make_config():
    # Initialize attack_jobs
    attack_jobs = []

    # Loop over each directory name in syn_path
    for dir_name in os.listdir(syn_path):
        # Create a TablesManager object with the directory path
        tm = TablesManager(dir_path=os.path.join(syn_path, dir_name))
        columns = list(tm.df_orig.columns)
        # Get the protected ID columns
        pid_cols = tm.get_pid_cols()
        if len(pid_cols) > 0:
            # We can't really run the attack on time-series data
            continue
        # Remove the protected ID columns from columns
        columns = [col for col in columns if col not in pid_cols]
        for col1, col2 in itertools.combinations(columns, 2):
            attack_jobs.append({
                'index': len(attack_jobs),
                'dir_name': dir_name,
                'columns': [col1, col2],
            })
    # randomize the order in which the attack_jobs are run
    random.shuffle(attack_jobs)

    # Write attack_jobs into a JSON file
    with open(os.path.join(attack_path, 'attack_jobs.json'), 'w') as f:
        json.dump(attack_jobs, f, indent=4)

    exe_path = os.path.join(code_path, 'suppress_threshold_attack.py')
    venv_path = os.path.join(base_path, 'sdx_venv', 'bin', 'activate')
    slurm_dir = os.path.join(attack_path, 'slurm_out')
    os.makedirs(slurm_dir, exist_ok=True)
    slurm_out = os.path.join(slurm_dir, 'out.%a.out')
    num_jobs = len(attack_jobs) - 1
    # Define the slurm template
    slurm_template = f'''#!/bin/bash
#SBATCH --job-name=suppress_attack
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
    with open(os.path.join(attack_path, 'attack.slurm'), 'w') as f:
        f.write(slurm_template)

def to_list(value):
    if isinstance(value, str) or not hasattr(value, '__iter__'):
        return [value]
    else:
        return list(value)

def get_attack_info(df_syn, comb, known_val_comb, target_col, target_val):
    ''' Find the synthetic data that correpsonds to the columns in comb+target_col.
        This might be the full synthetic table (for instance, if comb has 3 columns).
        Prior to this, we have already determined that there are either 2 or 3 rows
        that contain the known_val_comb values in the columns in comb. We have also
        determined that 2 of those rows have value target_val in target_col. We want
        to see if the corresponding rows appear in the synthetic data.
    '''
    if df_syn is None:
        print(f"Could not find a synthetic table for columns {comb}")
        sys.exit(1)
    if len(list(df_syn.columns)) == len(comb+[target_col]):
        best_syn = True
    else:
        best_syn = False
    mask = (df_syn[comb] == known_val_comb).all(axis=1)
    subset = df_syn[mask]
    mask_with_target = subset[target_col] == target_val
    num_with_target = int(mask_with_target.sum())
    mask_without_target = subset[target_col] != target_val
    num_without_target = int(mask_without_target.sum())
    return num_with_target, num_without_target, best_syn

def run_attacks(tm, file_path, job):
    max_attack_instances = 10
    attack_summary = {'summary': {'num_samples':[0,0,0,0,0],
                              'num_possible_combs_targets': 0,
                              'num_possible_combs': 0,
                              'num_rows': tm.df_orig.shape[0],
                              'num_cols': tm.df_orig.shape[1],
                              'num_attacks': 0,
                              'tp': 0,
                              'fp': 0,
                              'tn': 0,
                              'fn': 0,
                              'finished': True,
                              'job':job},
                       'sample_instances': [[],[],[],[],[]],
                       'attack_results': []}

    columns = list(tm.df_orig.columns)
    if do_comb_3_and_4:
        combinations = list(itertools.combinations(columns, 3)) + list(itertools.combinations(columns, 4))
        combinations = [comb for comb in combinations if all(col in comb for col in job['columns'])]
        combs = [[job['columns'][0]], [job['columns'][1]], job['columns']] + combinations
    else:
        combs = [[job['columns'][0]], [job['columns'][1]], job['columns']]

    # For each comb, find all values that appear exactly 3 times in tm.df_orig
    sum_base_probs = 0
    do_summarize = False
    for comb in combs:
        if len(attack_summary['attack_results']) >= max_attacks:
            attack_summary['summary']['finished'] = False
            break
        if do_summarize:
            summarize_and_write(attack_summary, file_path, sum_base_probs)
            do_summarize = False
        if len(comb) == len(columns):
            # Can't have a target unknown column in this case
            continue
        comb = list(comb)
        num_known_columns = len(comb)
        # Group the DataFrame by the columns in comb and count the number of rows for each group
        grouped = tm.df_orig.groupby(comb).size()
        attack_summary['summary']['num_possible_combs'] += grouped.shape[0]
        attack_summary['summary']['num_possible_combs_targets'] += grouped.shape[0] * (tm.df_orig.shape[1] - num_known_columns)

        # Filter the groups to only include those with exactly 3 rows
        known_val_combs = grouped[grouped == 3].index.tolist()
        for target_col in list(tm.df_orig.columns):
            # We loop through the columns first so that we only need to pull in the
            # relevant df_syn once per target_col
            if target_col in comb:
                continue
            df_syn = None
            for known_val_comb in known_val_combs:
                # Find all columns where at least two of the 3 rows have the same value
                known_val_comb = to_list(known_val_comb)
                mask = (tm.df_orig[comb] == known_val_comb).all(axis=1)
                known_rows = tm.df_orig[mask]
                target_val = None
                victim_val = None
                correct_pred = None
                if known_rows[target_col].nunique() == 1:
                    # set target_val to the mode of known_rows[target_col]
                    target_val = known_rows[target_col].mode()[0]
                    victim_val = target_val
                    correct_pred = 'positive'
                elif known_rows[target_col].nunique() == 2:
                    # set target_val to the mode value of known_rows[target_col]
                    target_val = known_rows[target_col].mode()[0]
                    # set victim_val to the other value
                    victim_val = known_rows[target_col][known_rows[target_col] != target_val].values[0]
                    correct_pred = 'negative'
                if target_val is None:
                    continue
                num_rows_with_target_val = len(tm.df_orig[tm.df_orig[target_col] == target_val])
                num_distinct_values = len(tm.df_orig[target_col].unique())
                attack_summary['summary']['num_samples'][num_known_columns] += 1
                if len(attack_summary['sample_instances'][num_known_columns]) < max_attack_instances:
                    # The stringify's are needed to avoid json serialization issues
                    attack_instance = {
                        'target_col': target_col,
                        'num_rows_with_target_val': num_rows_with_target_val,
                        'num_distinct_target_col_vals': num_distinct_values,
                        'target_val': str(target_val),
                        'victim_val': str(victim_val),
                        'known_cols': comb,
                        'known_vals': known_val_comb,
                        'correct_pred': correct_pred,
                        'file_path': file_path,
                        #'known_rows': str(known_rows.to_dict(orient='records')),
                    }
                    attack_summary['sample_instances'][num_known_columns].append(attack_instance)
                if df_syn is None:
                    df_syn = tm.get_best_syn_df(columns=comb+[target_col])
                num_with_target, num_without_target, best_syn = get_attack_info(df_syn, comb, known_val_comb, target_col, target_val)
                got_tp = False
                # probability that simply guessing positive would be correct
                sum_base_probs += num_rows_with_target_val / tm.df_orig.shape[0]
                if correct_pred == 'positive' and num_with_target > 0 and num_without_target == 0:
                    attack_summary['summary']['tp'] += 1
                    got_tp = True
                elif correct_pred == 'negative' and num_with_target > 0 and num_without_target == 0:
                    attack_summary['summary']['fp'] += 1
                elif correct_pred == 'positive' and (num_with_target == 0 or num_without_target > 0):
                    attack_summary['summary']['fn'] += 1
                elif correct_pred == 'negative' and (num_with_target == 0 or num_without_target > 0):
                    attack_summary['summary']['tn'] += 1
                attack_result = {
                    # number rows with target value
                    'nrtv': num_rows_with_target_val,
                    # number of distinct target values
                    'ndtv': num_distinct_values,
                    # What a correct prediction would be
                    'c': correct_pred,
                    # number of synthetic rows with known values and target value
                    'nkwt': num_with_target,
                    # number of synthetic rows with known values and not target value
                    'nkwot': num_without_target,
                    # whether the synthetic table is the best one
                    'bs': best_syn,
                    # the number of known columns
                    'nkc': num_known_columns,
                    # whether simple critieria yielded true positive
                    'tp': got_tp,
                }
                attack_summary['attack_results'].append(attack_result)
                if len(attack_summary['attack_results']) % 10000 == 0:
                    do_summarize = True
            df_syn = None
    summarize_and_write(attack_summary, file_path, sum_base_probs)

def summarize_and_write(attack_summary, file_path, sum_base_probs):
    tp = attack_summary['summary']['tp']
    fp = attack_summary['summary']['fp']
    tn = attack_summary['summary']['tn']
    fn = attack_summary['summary']['fn']
    num_attacks = tp + fp + tn + fn
    avg_base = sum_base_probs / num_attacks if num_attacks > 0 else 0
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0
    pi = (prec - avg_base) / (1 - avg_base) if (1 - avg_base) != 0 else 0
    if (tp + fp + tn + fn) != len(attack_summary['attack_results']):
        print(f"Error: {tp} + {fp} + {tn} + {fn} != {len(attack_summary['attack_results'])}")
        sys.exit(1)
    attack_summary['summary']['num_attacks'] = num_attacks
    attack_summary['summary']['precision'] = prec
    attack_summary['summary']['precision_improvement'] = pi
    attack_summary['summary']['coverage_known_naive'] = (tp + fp) / num_attacks if num_attacks != 0 else 0
    attack_summary['summary']['coverage_all_combs'] = (tp + fp) / attack_summary['summary']['num_possible_combs'] if attack_summary['summary']['num_possible_combs'] != 0 else 0
    attack_summary['summary']['coverage_all_combs_targets'] = (tp + fp) / attack_summary['summary']['num_possible_combs_targets'] if attack_summary['summary']['num_possible_combs_targets'] != 0 else 0

    # Write attack_summary to file_path
    with open(file_path, 'w') as f:
        json.dump(attack_summary, f, indent=4)

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
    # make a string that contains the column names in job['columns'] separated by '_'
    column_str = '_'.join(job['columns'])
    file_name = f"{job['dir_name']}.{column_str}.json"
    file_path = os.path.join(instances_path, file_name)

    if os.path.exists(file_path):
        print(f"File already exists: {file_path}")
        return

    # Make a TablesManager object
    tm = TablesManager(dir_path=os.path.join(syn_path, job['dir_name']))
    run_attacks(tm, file_path, job)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", help="'config' to run make_config(), or an integer to run run_attacks()")
    args = parser.parse_args()

    if args.command == 'config':
        make_config()
    elif args.command == 'gather':
        gather(instances_path=os.path.join(attack_path, 'instances'))
    elif args.command == 'model':
        do_model()
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