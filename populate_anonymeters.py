import pandas as pd
import itertools
import json
import my_utilities as mu
import os
import sys
import random
import targets
from syndiffix_tools.tables_manager import TablesManager
from pathlib import Path

baseDir = os.environ['SDX_TEST_DIR']
synDataPath = Path(baseDir, 'synDatasets')

for dir in os.listdir(synDataPath):
    thisDataPath = Path(synDataPath, dir)
    anonPath = os.path.join(thisDataPath, 'anonymeter')
    if not os.path.exists(anonPath):
        print(f"anonymeter path {anonPath} does not exist")
        continue
    trainingFile = os.path.join(thisDataPath, 'anonymeter', f'training.parquet')
    if not os.path.exists(trainingFile):
        print(f"trainingFile {trainingFile} does not exist")
        continue
    df_orig = pd.read_parquet(trainingFile)
    baseName = dir
    tm_from = TablesManager(dir_path=thisDataPath)
    if len(tm_from.get_pid_cols()) > 0:
        print(f"Skipping because time series data in {dir}")
        continue
    tm_to = TablesManager(dir_path=anonPath)
    tm_to.put_df_orig(df_orig, baseName)