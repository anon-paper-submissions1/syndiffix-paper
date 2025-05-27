import my_utilities as mu
import os
from pathlib import Path
from syndiffix_tools.tables_manager import TablesManager

'''
Go through all of the original datasets and create syndiffix_tools directories
for them.
'''

baseDir = os.environ['SDX_TEST_DIR']
pqDirPath = Path(baseDir, 'original_data_parquet')
synDatasetsDir = Path(baseDir, 'synDatasets')
os.makedirs(synDatasetsDir, exist_ok=True)

for fileName in [fileName for fileName in os.listdir(pqDirPath) if fileName.endswith('.parquet')]:
    baseName = fileName.replace('.parquet','')
    pqFilePath = Path(pqDirPath, fileName)
    print(f"Read file {pqFilePath}")
    df = mu.load_pq(pqFilePath.as_posix())
    thisSdxDir = Path(synDatasetsDir, baseName)
    # check if thisSdxDir already exists
    if thisSdxDir.exists():
        print(f"Directory {thisSdxDir} already exists. Skipping.")
        continue
    os.makedirs(thisSdxDir, exist_ok=True)
    # Create the TablesManager object
    tm = TablesManager(dir_path=thisSdxDir)
    tm.put_df_orig(df, baseName)