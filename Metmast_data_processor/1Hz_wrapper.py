__author__ = 'Elliot I. Simon'
__email__ = 'ellsim@dtu.dk'
__version__ = 'January 31, 2023'

import sys
import os
import glob
from pathlib import Path

inpath = Path("D:\ellsim\Globe\metmast_data\\1Hz_raw")
outpath = Path("D:\ellsim\Globe\metmast_data\\1Hz_processed")
path_8Hz = Path('D:\ellsim\Globe\metmast_data\8Hz_processed')

files = sorted(inpath.glob('*1Hz*'), key=lambda p: os.path.getctime(p))

for f in files:
    print(f)
    os.system("python 1Hz_main.py {0} {1} {2}".format(f, outpath, path_8Hz))
print('All done!')