#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  2 17:41:32 2019

@author: sungkyun
"""

import numpy as np
d = 64                           # dimension
nb = 1000000                      # database size
nq = 10000                       # nb of queries
np.random.seed(1234)             # make reproducible
xb = np.random.random((nb, d)).astype('float32')
xb[:, 0] += np.arange(nb) / 1000.
xq = np.random.random((nq, d)).astype('float32')
xq[:, 0] += np.arange(nq) / 1000.

# xb: (100000,64)
# xq: (10000, 64)

#%% Building an index and adding the vectors to it
import faiss                   # make faiss available
index = faiss.IndexFlatL2(d)   # build the index
print(index.is_trained)
index.add(xb)                  # add vectors to the index
print(index.ntotal)

#%% Search
k = 4                          # we want to see 4 nearest neighbors
D, I = index.search(xb[:5], k) # sanity check
print(I)
print(D)
D, I = index.search(xq, k)     # actual search
print(I[:5])                   # neighbors of the 5 first queries
print(I[-5:])                  # neighbors of the 5 last queries


#%% Fingerprint DB
import numpy as np
import faiss 
# spec, batch-hard
#x_db_train = np.load('./logs/fp_output/20190705-0604/train/embs.npy')
#x_db_test = np.load('./logs/fp_output/20190705-0604/test/embs.npy')

## spec, batch-all
#x_db_train = np.load('./logs/fp_output/20190705-2133/train/embs.npy')
#x_db_test = np.load('./logs/fp_output/20190705-2133/test/embs.npy')


# melspec, batch-hard
x_db_train = np.load('./logs/fp_output/20190708-1431/train/embs.npy')
x_db_test = np.load('./logs/fp_output/20190708-1431/test/embs.npy')


# melspec, batch-all
x_db_train = np.load('./logs/fp_output/20190709-1439/train/embs.npy')
x_db_test = np.load('./logs/fp_output/20190709-1439/test/embs.npy')

x_db_train = np.load('./logs/fp_output/20190709-1439-ep-1/all/embs.npy')
x_db_test = None


x_db_train = np.load('./logs/fp_output/20190715-1921-ep-final/all/embs.npy')

x_db_train = np.load('./logs/fp_output/20190716-2008-ep-final/all/embs.npy')

x_db_train = np.load('./logs/fp_output/20190720-1613-ep-final/all/embs.npy')

#index = faiss.IndexFlatL2(128)
#index.add(x_db)   
#print(index.ntotal)
#%% Search
x_utube = np.load('./logs/fp_output/20190715-1921-ep-final/utube/embs.npy')
x_utube_info = np.load('./logs/fp_output/20190715-1921-ep-final/utube/songid_frame_idx.npy', allow_pickle=True)

x_utube = np.load('./logs/fp_output/20190716-2008-ep-final/utube/embs.npy')
x_utube_info = np.load('./logs/fp_output/20190716-2008-ep-final/utube/songid_frame_idx.npy', allow_pickle=True)

x_utube = np.load('./logs/fp_output/20190720-1613-ep-final/utube/embs.npy')
x_utube_info = np.load('./logs/fp_output/20190720-1613-ep-final/utube/songid_frame_idx.npy', allow_pickle=True)


x_db_ext = x_utube[:96, :]
x_q = x_utube[96:,:]

#x_db = np.vstack((x_db_ext, x_db_train, x_db_test)) # BTS: [:96]
x_db = np.vstack((x_db_ext, x_db_train)) # BTS: [:96]
index = faiss.IndexFlatL2(128)
index.add(x_db)   

# sanity check
D, I = index.search(x_db_ext[:16,:], 5)
print(D)
print(I)

#D, I = index.search(np.expand_dims(x_q[:16], axis=0), 500000)
D, I = index.search(x_q[:96,:], 50000)
print(D)
print(I)

np.where(I==15)

#%%
import matplotlib.pyplot as plt
plt.figure()
plt.plot(x_q[15,:])
plt.plot(x_db[15,:])
plt.plot(x_db[18389,:])

plt.plot(x_db[243,:])
plt.plot(x_db[79904,:])
