import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import seaborn as sns
import json

def make_plot(df, xlabel, xcolumn, file_name, ylim=None):
    # Create a new column labeled "num_row_num_val"
    df['num_row_num_val'] = df['num_row'].astype(str) + ':' + df['num_val'].astype(str)

    # Create a color and marker map for the 'dim' column
    dim_styles = {dim_value: (color, marker) for dim_value, color, marker in zip(df['dim'].unique(), sns.color_palette(n_colors=df['dim'].nunique()), ['v', '^', 'o'])}
    df['color'] = df['dim'].map(lambda x: dim_styles[x][0])
    df['marker'] = df['dim'].map(lambda x: dim_styles[x][1])

    # Create a size map for the 'num_col' column
    size_map = {20: 10, 40: 23, 80: 36, 160: 50}
    df['size'] = df['num_col'].map(size_map)
    #print(df[['color', 'marker', 'size','dim']].to_string())

    # Specify the order of x-axis values
    order = ['10:8', '10:4', '10:2', '20:8', '20:4', '20:2', '40:8', '40:4', '40:2', '80:8', '80:4', '80:2', '160:8', '160:4', '160:2']
    order = ['10:10', '10:5', '10:2', '20:10', '20:5', '20:2', '40:10', '40:5', '40:2', '80:10', '80:5', '80:2', '160:10', '160:5', '160:2']
    df['num_row_num_val'] = pd.Categorical(df['num_row_num_val'], categories=order, ordered=True)

    # Sort the DataFrame by 'num_row_num_val'
    df = df.sort_values('num_row_num_val')

    # Create the scatter plot
    plt.figure(figsize=(8, 4))
    #for (color, marker, size), dim in zip(df[['color', 'marker', 'size']].drop_duplicates().values, sorted(df['dim'].unique())):
    #print(df[['color', 'marker', 'size']].drop_duplicates().values)
    #for (color, marker, size) in zip(df[['color', 'marker', 'size']].drop_duplicates().values):
    for thing in zip(df[['color', 'marker', 'size']].drop_duplicates().values):
        color = thing[0][0]
        marker = thing[0][1]
        size = thing[0][2]
        #print(color, marker, size)
        #df_dim = df[df['dim'] == dim]
        df_dim = df[(df['color'] == color) & (df['marker'] == marker) & (df['size'] == size)]
        # Remove all rows from df_dim where any column has a NaN value
        df_dim = df_dim.dropna()
        print(df_dim[['num_row_num_val',xcolumn]].to_string())
        plt.scatter(df_dim['num_row_num_val'], df_dim[xcolumn], color=color, marker=marker, alpha=0.8, s=size)

    plt.xlabel('Aggregate True Count : Number of Other Column Values', fontsize=12)
    plt.ylabel(xlabel, fontsize=12)

    # Set the range of the y-axis
    if ylim is not None:
        plt.ylim(ylim)  # Replace with your desired range

    # Add horizontal grid lines
    plt.grid(axis='y')

    # Reduce the font size of the tick labels
    plt.tick_params(axis='both', which='major', labelsize=8)

    # Create legend
    legend_elements = [mlines.Line2D([0], [0], color=dim_styles[dim][0], marker=dim_styles[dim][1], linestyle='None', markersize=5, label=f"Attack dims {dim}") for dim in dim_styles]
    legend1 = plt.legend(handles=legend_elements, fontsize='small', loc='lower left', bbox_to_anchor=(0, 0.75))
    plt.legend([mlines.Line2D([0], [0], color='black', marker='o', markersize=size/10, linestyle='None') for num_col, size in size_map.items()], ['Columns: {}'.format(num_col) for num_col in size_map.keys()], title='', loc='lower left', bbox_to_anchor=(0, 0.5), fontsize='small')
    plt.gca().add_artist(legend1)

    # Save the plot
    file_path = os.path.join('results', file_name)
    plt.savefig(file_path, bbox_inches='tight', dpi=300)	

    plt.close()

# Read in the file
with open('./results/exact_count_data.json', 'r') as f:
    data = json.load(f)

# Convert the list of dicts to a DataFrame
df = pd.DataFrame(data)
#df = df[['num_row', 'num_val', 'dim', 'num_col', 'correct_averages', 'leaf_over_frac_avg']]
#print(df.to_string())
make_plot(df, 'Precision', 'correct_averages', 'exact_count_prec_agg.png', ylim=[-0.05, 1.05])
make_plot(df, 'Fraction Non-suppressed Leafs', 'leaf_over_frac_avg', 'exact_count_leaf_agg.png', ylim=[-0.05, 1.05])
make_plot(df, 'Fraction Non-suppressed Branch', 'branch_over_frac_avg', 'exact_count_branch_agg.png', ylim=[-0.05, 1.05])
make_plot(df, 'Avg Num Leaf', 'num_leaf_avg', 'exact_count_num_leaf_agg.png')