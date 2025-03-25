import os
import sys
import json
from pathlib import Path
from syndiffix_tools.tables_manager import TablesManager
import my_utilities as mu

baseDir = os.environ['SDX_TEST_DIR']
# make sinDatasetsDir using Path
synDatasetsDir = Path(baseDir, 'synDatasets')
os.makedirs(synDatasetsDir, exist_ok=True)
jobsFile = 'allSynCombs.json'
allCombsPath = Path(baseDir, jobsFile)
with allCombsPath.open('r') as f:
    allCombs = json.load(f)
myJobNum = int(sys.argv[1])
if myJobNum > len(allCombs)-1:
    print(f"Bad job number {myJobNum}")
    quit()
job = allCombs[myJobNum]

tm = TablesManager(dir_path=job['synDir'])
print(f"Synthesize {job}")
tm.synthesize(columns=job['cols'], also_save_stats=True)