import json
import math
import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np
import seaborn as sns

window = 500

def heatmap_data_combs(df):
    # Define the combinations of the three variables
    combinations = [('agg_size', 'num_val'), ('agg_size', 'num_col'), ('num_val', 'num_col')]

    # For each combination, create a heatmap
    for dim in [1, 2, 3]:
        df_dim = df.copy()
        df_dim = df_dim[df_dim['dim'] == dim]
        for i, (var1, var2) in enumerate(combinations):
            # Create bins for var1 and var2
            df_dim[var1 + '_bin'] = pd.qcut(df_dim[var1], 10, duplicates='drop')
            df_dim[var2 + '_bin'] = pd.qcut(df_dim[var2], 10, duplicates='drop')

            # Calculate precision for each bin
            precision_per_bin = df_dim.groupby([var1 + '_bin', var2 + '_bin'], observed=False)['correct'].mean().reset_index()

            # Convert the interval categories to strings
            #precision_per_bin[var1 + '_bin'] = precision_per_bin[var1 + '_bin'].astype(str)
            #precision_per_bin[var2 + '_bin'] = precision_per_bin[var2 + '_bin'].astype(str)

            # Pivot the data for the heatmap
            precision_pivot = precision_per_bin.pivot(index=var1 + '_bin', columns=var2 + '_bin', values='correct')

            # Create the heatmap
            plt.figure(i)
            sns.heatmap(precision_pivot, cmap='YlGnBu', linewidths=0.5)
            plt.title(f'Precision for {var1} and {var2}, dim {dim}')
            plt.xlabel(var2)
            plt.ylabel(var1)
            plt.tight_layout()
            file_path = os.path.join('results', 'exact_count', f'exact_count_heatmap.dim{dim}.{var1}.{var2}.png')
            plt.savefig(file_path, bbox_inches='tight', dpi=300)	
            plt.close()

def heatmap_data_3way(df):
    # Define the number of bins for each column
    num_bins = 6

    # For each 'dim' value, create a plot
    for dim in [1, 2, 3]:
        # Filter the DataFrame for the current 'dim' value
        df_dim = df[df['dim'] == dim]

        # Create bins for 'agg_size', 'num_val', and 'num_col'
        for var in ['agg_size', 'num_val', 'num_col']:
            df_dim[var + '_bin'] = pd.cut(df_dim[var], bins=num_bins)

        # Create a figure with 6 subplots
        fig, axs = plt.subplots(2, 3, figsize=(15, 10))
        axs = axs.flatten()

        # For each 'agg_size' bin, create a heatmap
        for i, (agg_size_bin, df_agg_size_bin) in enumerate(df_dim.groupby('agg_size_bin', observed=False)):
            # Calculate precision for each bin
            precision_per_bin = df_agg_size_bin.groupby(['num_val_bin', 'num_col_bin'], observed=False)['correct'].mean().reset_index()

            # Pivot the data for the heatmap
            precision_pivot = precision_per_bin.pivot(index='num_val_bin', columns='num_col_bin', values='correct')

            # Create the heatmap
            sns.heatmap(precision_pivot, cmap='YlGnBu', linewidths=0.5, ax=axs[i], vmin=0, vmax=0.8, annot=True, fmt=".2f")
            axs[i].set_title(f'agg_size: {agg_size_bin}')
            if i < 3:  # If it's a top heatmap, remove the X labels
                axs[i].set_xlabel('')
                axs[i].set_xticklabels([])
            if i % 3 != 0:  # If it's not a leftmost heatmap, remove the Y labels
                axs[i].set_ylabel('')
                axs[i].set_yticklabels([])

        # Add a colorbar
        #fig.colorbar(axs[0].collections[0], ax=axs, location="right", use_gridspec=False, pad=0.2)

        # Show the plot
        plt.tight_layout()
        file_path = os.path.join('results', 'exact_count', f'exact_count_3way.dim{dim}.png')
        plt.savefig(file_path, bbox_inches='tight', dpi=300)	
        plt.close()


def bin_by_dim(df):
    # Set the order of the categories in 'dim'
    df['dim'] = pd.Categorical(df['dim'], categories=[1, 2, 3], ordered=True)
    
    # Create the boxplot
    plt.figure(figsize=(5, 2))
    sns.boxplot(data=df, y='dim', x='precision', orient='h', color='lightblue')
    plt.xlabel('Precision')
    plt.ylabel('Number of columns\nper sample')
    plt.tight_layout()
    file_path = os.path.join('results', 'exact_count', 'exact_by_dim_box.png')
    plt.savefig(file_path, bbox_inches='tight', dpi=300)	
    file_path = os.path.join('results', 'exact_count', 'exact_by_dim_box.pdf')
    plt.savefig(file_path, bbox_inches='tight', dpi=300)	
    plt.close()

def print_grouped_stats(df_sort):
    # List of columns to group by
    columns = ['rows_per_val', 'num_col', 'agg_size', 'num_val']

    for column in columns:
        print(f"Statistics for column: {column}")
        # Group by the specified column and calculate the number of rows and sum of 'count'
        grouped = df_sort.groupby(column).agg(
            num_rows=('count', 'size'),
            count_sum=('count', 'sum')
        )

        # Print the results
        for value, stats in grouped.iterrows():
            print(f'{column}: {value}, num_rows: {stats["num_rows"]}, count_sum: {stats["count_sum"]}')
        print()

def bin_data(df, num_bins=4):
    # Create bins for each continuous variable
    bins = []
    total_count = 0
    bin_edges = {
        'rows_per_val': num_bins,  # Replace with your desired bin edges
        'num_col': num_bins,  # Replace with your desired bin edges
        'agg_size': [0, 15, 50, 100, 200],  # Replace with your desired bin edges
        'num_val': num_bins  # Replace with your desired bin edges
    }
    for dim in [1, 2, 3]:
        df_dim = df[df['dim'] == dim]
        for var in ['rows_per_val', 'num_col', 'agg_size', 'num_val']:
            df_dim[var + '_bin'] = pd.cut(df_dim[var], bins=bin_edges[var])

        # Group by the bin labels and calculate precision and count for each group
        grouped = df_dim.groupby(['rows_per_val_bin', 'num_col_bin', 'agg_size_bin', 'num_val_bin'])
        result = grouped['correct'].agg(['mean', 'count']).reset_index()

        # Create a list of dicts for each group
        for _, row in result.iterrows():
            if row['mean'] is not None and not math.isnan(row['mean']) and row['count'] > 20:
                bin_dict = {
                    'precision': row['mean'],
                    'count': row['count'],
                    'dim': dim,
                    'rows_per_val': str(row['rows_per_val_bin']),
                    'num_col': str(row['num_col_bin']),
                    'agg_size': str(row['agg_size_bin']),
                    'num_val': str(row['num_val_bin']),
                }
                bins.append(bin_dict)
                total_count += row['count']

    # Sort the list of dicts by precision in descending order
    bins.sort(key=lambda x: x['precision'], reverse=True)
    print("--------------------------------------------------")
    print(f"bins {len(bins)}")
    print(f"total count {total_count}")
    print(f"average bin size is {total_count/len(bins)}")
    print("--------------------------------------------------")

    # Write the list of dicts to a JSON file
    file_path = os.path.join('results', 'exact_count', f'exact_count_bins.json')
    with open(file_path, 'w') as f:
        json.dump(bins, f, indent=4)

    # From bins, generate a dataframe df_sort that has three columns: 'precision', 'cum_count', and 'dim'. The order of rows in df_sort is the same as bins. cum_count is the cumulative sum of all prior 'count' values in the table. dim is the 'dim' value of the corresponding row in bins.
    df_sort = pd.DataFrame(bins)
    print(f"df_sort\n{df_sort.shape}")
    print(f"df_sort\n{df_sort.head()}")
    # Display the unique values in columns rows_per_val, num_col, agg_size, and num_val
    print(f"rows_per_val: {df_sort['rows_per_val'].unique()}")
    print(f"num_col: {df_sort['num_col'].unique()}")
    print(f"agg_size: {df_sort['agg_size'].unique()}")
    print(f"num_val: {df_sort['num_val'].unique()}")
    print("Rows per dim value:")
    print(df_sort['dim'].value_counts())
    print("Sum of column 'count' per dim value:")
    print(df_sort.groupby('dim')['count'].sum())
    print("Average of column 'count' per dim value:")
    print(df_sort.groupby('dim')['count'].mean())
    bin_by_dim(df_sort)

    # Group by 'num_col' and 'dim' and calculate mean and standard deviation of 'precision'
    grouped = df_sort.groupby(['num_col', 'dim'])['precision'].agg(['mean', 'std', 'max', 'count'])

    # Print the results
    for (num_col, dim), stats in grouped.iterrows():
        print(f'num_col: {num_col}, dim: {dim}, mean: {stats["mean"]:.2f}, std: {stats["std"]:.2f}, max:{stats["max"]:.2f}, count: {stats["count"]}')

    print_grouped_stats(df_sort)

    df_sort['cum_count'] = df_sort.groupby('dim')['count'].cumsum()
    quit()

    # Create a scatterplot with 'cum_count' on the x-axis and 'precision' on the y-axis. The color of each point is determined by the 'dim' value.
    plt.figure(figsize=(6, 3))
    colors = ['blue', 'orange', 'green']  # Specify colors for each 'dim' value
    for i, dim in enumerate([1, 2, 3]):
        df_dim = df_sort[df_sort['dim'] == dim]
        # Extract 'agg_size' values and remove the brackets and spaces
        agg_size = df_dim['agg_size'].str[1:-1].str.split(',', expand=True).astype(float)
        
        # Calculate the average of the bin boundaries
        agg_size_avg = agg_size.mean(axis=1)
        agg_size_max = agg_size.max(axis=1)
        
        # Normalize 'agg_size_avg' to the range [5, 15]
        min_value = agg_size_avg.min()
        max_value = agg_size_avg.max()
        s = ((agg_size_avg - min_value) / (max_value - min_value)) * (20 - 2) + 2
        thresh = 15
        plt.plot(df_dim[agg_size_max > thresh]['cum_count'], df_dim[agg_size_max > thresh]['precision'], label=f'Attack dim {dim}', markersize=2, marker='o', color=colors[i], linestyle='-')
        plt.scatter(df_dim[agg_size_max <= thresh]['cum_count'], df_dim[agg_size_max <= thresh]['precision'], label=f'Attack dim {dim}', s=50, marker='+', color=colors[i])
        #plt.scatter(df_dim['cum_count'], df_dim['precision'], label=f'Attack dim {dim}', s=5)
    # Create a custom legend for 'dim'
    legend_elements_dim = [mlines.Line2D([0], [0], color=colors[i], marker='o', linestyle='None', markersize=5, label=f'Attack dim {dim}') for i, dim in enumerate([1, 2, 3])]
    legend_dim = plt.legend(handles=legend_elements_dim, loc='upper right', fontsize='small')

    # Add the legend manually to the current Axes.
    plt.gca().add_artist(legend_dim)

    # Create a custom legend for marker types
    legend_elements_marker = [mlines.Line2D([0], [0], color='black', marker='o', linestyle='None', markersize=2, label='Aggregate > 15'),
                            mlines.Line2D([0], [0], color='black', marker='+', linestyle='None', markersize=7, label='Aggregate <= 15')]
    plt.legend(handles=legend_elements_marker, loc='lower left', fontsize='small')
    plt.xlabel('Cumulative Number of Attacks', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    # Shrink the font size of the tick labels for both x and y axis
    plt.tick_params(axis='both', which='major', labelsize='small')
    plt.tight_layout()
    file_path = os.path.join('results', 'exact_count', f'exact_count_cum.png')
    plt.savefig(file_path, bbox_inches='tight', dpi=300)
    file_path = os.path.join('results', 'exact_count', f'exact_count_cum.pdf')
    plt.savefig(file_path, bbox_inches='tight', dpi=300)
    plt.close()


def apply_limits(df, limits, sort_col):
    xlabel = '\n'
    for col, limit in limits.items():
        if col == sort_col:
            continue
        if limit[1] is None:
            continue
        if limit[0] == 'min':
            xlabel += f'{col}>={limit[1]}\n'
            df = df[df[col] >= limit[1]]
        else:
            xlabel += f'{col}<={limit[1]}\n'
            df = df[df[col] <= limit[1]]
    xlabel = xlabel[:-1]
    return df, xlabel

def split_sort_and_roll(df, sort_col, all_col, smooth=False, limits=None):
    # make a copy of all_col
    other_col = all_col.copy()
    # remove sort_col from all_col
    other_col.remove(sort_col)
    # make dataframe df_agg_size with columns agg_size, dim, and correct 
    df_temp = df.copy()
    df_temp = df_temp[[sort_col, 'dim', 'correct']+other_col]
    min = df_temp[sort_col].min()
    # Assign a random float to each row in the sort_col column between the sort_col value and sort_col value + 1
    if smooth:
        df_temp[sort_col] = df_temp[sort_col].astype(float)
        for i in df_temp.index:
            # We don't randomize the minimum value so that the rolling average
            # shows the minimum value in the plot
            if df_temp.loc[i, sort_col] != min:
                df_temp.loc[i, sort_col] = df_temp.loc[i, sort_col] + np.random.rand()
    df_list = [None, None, None, None]
    if limits is None:
        xlabel = ''
    else:
        df_temp, xlabel = apply_limits(df_temp, limits, sort_col)
    # make a dataframe per attack dimension
    for dim in [1, 2, 3]:
        df_list[dim] = df_temp[df_temp['dim'] == dim]
        # sort agg_size
        df_list[dim] = df_list[dim].sort_values(sort_col)
        # add two new columns, precision and roll, where precision is the rolling average of correct with a window size of window and roll is the rolling average of agg_size with a window size of window
        df_list[dim]['precision'] = df_list[dim]['correct'].rolling(window=window).mean()
        df_list[dim]['roll'] = df_list[dim][sort_col].rolling(window=window).mean()
        # drop rows with NaN values
        df_list[dim] = df_list[dim].dropna()
    # concatinate the three dataframes into one
    df_concat = pd.concat([df_list[1], df_list[2], df_list[3]])
    # shuffle df_concat
    df_concat = df_concat.sample(frac=1)
    # sort df_concat by roll
    df_concat = df_concat.sort_values('roll')
    return df_concat, xlabel

def make_line_plots(df, all_cols, limits=None, tag=None):
    # Create a line plot for each variable
    df_agg_size, agg_size_label = split_sort_and_roll(df, 'agg_size', all_cols, smooth=True, limits=limits)
    df_num_val, num_val_label = split_sort_and_roll(df, 'num_val', all_cols, smooth=True, limits=limits)
    df_num_col, num_col_label = split_sort_and_roll(df, 'num_col', all_cols, smooth=True, limits=limits)
    df_rows_per_val, rows_per_val_label = split_sort_and_roll(df, 'rows_per_val', all_cols, smooth=True, limits=limits)
    df_frac_leaf_under, _ = split_sort_and_roll(df, 'frac_leaf_under', all_cols)

    # Define the color map for 'dim'
    color_map = {1: 'red', 2: 'green', 3: 'blue'}

    # Calculate the global minimum and maximum precision values
    min_precision = min(df['precision'].min() for df in [df_agg_size, df_num_val, df_num_col, df_rows_per_val, df_frac_leaf_under])
    max_precision = max(df['precision'].max() for df in [df_agg_size, df_num_val, df_num_col, df_rows_per_val, df_frac_leaf_under])

    print(f"max precision {max_precision} for rolling average window {window}")

    # Create a 2x2 subplot
    fig, axs = plt.subplots(3, 2, figsize=(8, 12))

    # Create line plots for df_agg_size, df_num_val, and df_num_col
    for df, ax, xlabel in zip([df_agg_size, df_num_val, df_num_col, df_rows_per_val], axs.flatten()[:4], ['Target aggregate size'+agg_size_label, 'Distinct values'+num_val_label, 'Columns'+num_col_label, 'Rows per value'+rows_per_val_label]):
        for dim, color in color_map.items():
            df_dim = df[df['dim'] == dim]
            ax.plot(df_dim['roll'], df_dim['precision'], color=color)
        ax.set_xlabel(xlabel)
        ax.set_ylabel('Precision')
        ax.set_ylim([min_precision, max_precision])  # Set the y-axis range

    # Create a scatter plot for df_frac_leaf_under
    for dim, color in color_map.items():
        df_dim = df_frac_leaf_under[df_frac_leaf_under['dim'] == dim]
        axs[2, 0].scatter(df_dim['roll'], df_dim['precision'], color=color, s=5)
    axs[2, 0].set_xlabel('Fraction suppressed nodes')
    axs[2, 0].set_ylabel('Precision')
    axs[2, 0].set_ylim([min_precision, max_precision])  # Set the y-axis range

    # Create a legend for 'dim'
    legend_elements = [mlines.Line2D([], [], color=color, label=f"Attack dims {dim}") for dim, color in color_map.items()]
    fig.legend(handles=legend_elements, loc='upper center', ncol=3, bbox_to_anchor=(0.5, 1.05))

    plt.tight_layout()

    if tag is None:
        file_name = f'exact_count.win{window}.png'
    else:
        file_name = f'exact_count.{tag}.win{window}.png'
    file_path = os.path.join('results', 'exact_count', file_name)
    plt.savefig(file_path, bbox_inches='tight', dpi=300)	

    plt.close()

json_files = [pos_json for pos_json in os.listdir("./exact_count_results") if pos_json.endswith('.json')]

data = []

for file in json_files:
    with open(f"./exact_count_results/{file}", "r") as json_file:
        print(file)
        dat1 = json.load(json_file)
        print(f"read {len(dat1)} records")
        for thing in dat1:
            data.append(thing)

print(f"total records {len(data)}")

df = pd.DataFrame.from_records(data)
# make a new column which is 1 - frac_leaf_over
df['frac_leaf_under'] = 1 - df['frac_leaf_over']
df['rows_per_val'] = (df['total_table_rows'] - df['agg_size']) / df['num_val']
print(f"Columns: {df.columns}")
for column in df.columns:
    print(f"{column}:\n{df[column].describe()}")

if True:
    bin_data(df)
if True:
    heatmap_data_3way(df)
if True:
    heatmap_data_combs(df)

if True:
    all_cols = ['agg_size', 'num_val', 'num_col', 'rows_per_val', 'frac_leaf_under']
    limits = {'agg_size': ['min', 100], 'num_val': ['max', 10], 'num_col': ['min', None], 'rows_per_val': ['max', 25]}
    make_line_plots(df, all_cols, limits=limits, tag='lim1')
    limits = {'agg_size': ['min', 100], 'num_val': ['max', 5], 'num_col': ['min', None], 'rows_per_val': ['max', 40]}
    make_line_plots(df, all_cols, limits=limits, tag='lim2')
    make_line_plots(df, all_cols)