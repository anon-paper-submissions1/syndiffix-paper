import pandas as pd
import itertools
import json
import my_utilities as mu
import os
import sys
from syndiffix_tools.tables_manager import TablesManager
from pathlib import Path

'''
Split the datasets into control and training parts
'''

baseDir = os.environ['SDX_TEST_DIR']
synDataPath = Path(baseDir, 'synDatasets')

for dir in os.listdir(synDataPath):
    thisDataPath = Path(synDataPath, dir)
    tm = TablesManager(dir_path=thisDataPath)
    pid_cols = tm.get_pid_cols()
    if len(pid_cols) > 0:
        continue
    anonymeterPath = Path(thisDataPath, 'anonymeter')
    os.makedirs(anonymeterPath, exist_ok=True)
    # select 70% of the rows for training
    df = tm.df_orig
    df = df.sample(frac=0.7)
    # create a control dataset with the remaining rows
    df_control = tm.df_orig.drop(df.index)
    # save the training data
    trainingPath = Path(anonymeterPath, 'training.parquet')
    mu.dump_pq(trainingPath, df)
    # save the control data
    controlPath = Path(anonymeterPath, 'control.parquet')
    mu.dump_pq(controlPath, df_control)
