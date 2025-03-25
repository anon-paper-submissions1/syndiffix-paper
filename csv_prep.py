import my_utilities as mu
import zipfile
import os

''' Read in csv files, detect datetime fields and set appropriately, 
write as parquet dataframes
'''

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
    outfile = filename.replace('.csv', '.parquet')
    outpath = os.path.join(pq_path, outfile)
    print(f"Writing to {outpath}")
    mu.dump_pq(outpath, df)