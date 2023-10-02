__author__ = 'Elliot I. Simon'
__email__ = 'ellsim@dtu.dk'
__version__ = 'January 31, 2023'

import sys
import os
import glob
from pathlib import Path

inpath = Path("D:\ellsim\Globe\metmast_data\\10min_raw")
outpath = Path("D:\ellsim\Globe\metmast_data\\10min_processed")
path_1Hz = Path("D:\ellsim\Globe\metmast_data\\1Hz_processed")
path_iwes_92m = Path("D:\ellsim\Globe\metmast_data\iwes_data\\NSO_92m_v1.txt")
path_iwes_32m = Path("D:\ellsim\Globe\metmast_data\iwes_data\\NSO_32m_v1.txt")

files = sorted(inpath.glob('*10min*'), key=lambda p: os.path.getctime(p))

for f in files:
    print(f)
    os.system("python 10min_main.py {0} {1} {2} {3} {4}".format(f, outpath, path_1Hz, path_iwes_92m, path_iwes_32m))
print('All done!')