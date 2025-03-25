import pandas as pd
from itertools import combinations

# Assuming df is your DataFrame
cols = df.columns

# Iterate over all combinations of columns
for r in range(1, len(cols) + 1):
    for combo in combinations(cols, r):
        # Group by the columns in the combo
        groups = df.groupby(list(combo))
        # Filter groups with exactly 3 rows
        groups = groups.filter(lambda x: len(x) == 3)
        # For each remaining column, check if at least two of the rows share a value
        for col in df.columns.drop(list(combo)):
            groups[f'{col}_unique_count'] = groups.groupby(list(combo))[col].transform('nunique')
            # Filter groups where the column has less than 3 unique values
            result = groups[groups[f'{col}_unique_count'] < 3]
            if not result.empty:
                print(f"Found matching rows for columns {combo} with shared values in column {col}:")
                print(result)