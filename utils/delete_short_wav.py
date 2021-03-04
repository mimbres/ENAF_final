# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import glob, os

DATA_ROOT_DIR = './'
MIN_FILE_SZ = 450000 # 450 Kb

fps = glob.glob(DATA_ROOT_DIR + '**/*.wav', recursive=True)

for fp in fps:
    file_sz = os.path.getsize(fp)
    if file_sz < MIN_FILE_SZ:
        # delete file
        print('deleted ', fp, ' {}Kb'.format(int(file_sz/1000)))
        os.remove(fp)
        
print('completed!!')