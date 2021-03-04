#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 21 20:38:45 2019

@author: sunkyun
"""
import faiss
import numpy as np
"""
index: FAISS index object
mode: {'r','w'}
"""
def index_ivf_read_write_ondisk(index=faiss.swigfaiss_avx2.Index,
                                mode='w',
                                emb_dir=str(),
                                emb_all=np.ndarray,
                                n_part=20):
    
    if (mode=='w'):
        index.train(emb_all)                                   
        # index.add(emb_all) # <-- in-memory
        faiss.write_index(index, emb_dir + "/trained.index") # <-- On-disk index, https://github.com/facebookresearch/faiss/blob/master/demos/demo_ondisk_ivf.py#L49
        for i in range(n_part): # diviede into n blocks..
            start = int(i * emb_all.shape[0] / n_part)
            end = int((i + 1) * emb_all.shape[0] / n_part)
            index = faiss.read_index(emb_dir + "/trained.index")
            index.add_with_ids(emb_all[start:end, :], np.arange(start, end))
            print("write " + emb_dir + "/block_{}.index".format(i))
            faiss.write_index(index, emb_dir + "/block_{}.index".format(i))
    
    ivfs = []
    for i in range(n_part):
        # the IO_FLAG_MMAP is to avoid actually loading the data thus
        # the total size of the inverted lists can exceed the
        # available RAM
        print("read " + emb_dir + "/block_{}.index".format(i))
        index = faiss.read_index(emb_dir + "/block_{}.index".format(i), faiss.IO_FLAG_MMAP)
        ivfs.append(index.invlists)
        index.own_invlists = False # avoid that the invlists get deallocated with the index
    
    index = faiss.read_index(emb_dir + "/trained.index") # construct the output index
    # prepare the output inverted lists. They will be written
    # to merged_index.ivfdata
    invlists = faiss.OnDiskInvertedLists(index.nlist, index.code_size,
                                         emb_dir + "/merged_index.ivfdata")
    
    # merge all the inverted lists
    ivf_vector = faiss.InvertedListsPtrVector()
    for ivf in ivfs:
        ivf_vector.push_back(ivf)
    print("merge {} inverted lists".format(ivf_vector.size()))
    ntotal = invlists.merge_from(ivf_vector.data(), ivf_vector.size())  
    
    # now replace the inverted lists in the output index
    index.ntotal = ntotal
    index.replace_invlists(invlists)
    print("write " + emb_dir + "/populated.index")
    faiss.write_index(index, emb_dir + "/populated.index")
    
    
    # perform a search from disk
    print("read " + emb_dir + "/populated.index")
    index = faiss.read_index(emb_dir + "/populated.index") # import DB--> OK!
    return index