import my_utilities as mu
import zipfile
import os

''' Read in csv files, detect datetime fields and set appropriately, 
write as parquet dataframes
'''

max_rows = 30000
base_dir = os.environ['SDX_TEST_DIR']
csv_path = os.path.join(base_dir, 'original_data_csv')
pq_path = os.path.join(base_dir, 'original_data_parquet')
os.makedirs(pq_path, exist_ok=True)

for filename in [filename for filename in os.listdir(csv_path) if filename.endswith('.zip')]:
    inpath = os.path.join(csv_path, filename)
    print(f"Found file {inpath}")
    with zipfile.ZipFile(inpath, 'r') as zip_ref:
        print(f"Writing to {csv_path}")
        zip_ref.extractall(csv_path)

for filename in [filename for filename in os.listdir(csv_path) if filename.endswith('.csv')]:
    inpath = os.path.join(csv_path, filename)
    print(f"Found file {inpath}")
    df = mu.load_csv(inpath)
    if df.shape[0] > max_rows:
        # shuffle the rows
        print(f"Truncating {filename} to {max_rows} rows")
        df = df.sample(n=max_rows, random_state=42).reset_index(drop=True)
    outfile = filename.replace('.csv', '.parquet')
    outpath = os.path.join(pq_path, outfile)
    # create outpath if it does not exist
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    print(f"Writing to {outpath}")
    mu.dump_pq(outpath, df)