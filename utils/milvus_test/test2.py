from milvus import Milvus, DataType
from pprint import pprint
import numpy as np
from tqdm import tqdm
import time

N = 45000000#560000#0#0#600000#10000 # (systhetic) DB size

INDEX_TYPE = 'IVF_PQ' # 'IVF_SQ8'
D = 128
CODE_SZ = 64#32#64 # 64 # 32 #64 # m
N_CENTROIDS = 1640 #100 #256 # 4096 # 262144-fail, 65536 is max #256 # nlist
N_PROBE_CENTROIDS = 20
SEGMENT_ROW_LIMIT = 100000#100000 #524288 #1048576 # default is 524,288(=512*1024), and max is 4,194,304 (=4*1024*1024) https://www.milvus.io/docs/storage_concept.md

collection_name = 'fp'
partition_name = 'test1'
host = '127.0.0.1'
port = '19530'

# Connect
client = Milvus(host, port)
#client.drop_collection('fp')
#client.drop_partition('fp', 'test1')

# Create collection
p_collection = {'fields':[{'name': 'emb',
                'type': DataType.FLOAT_VECTOR,
                'params': {'dim': D}
                }],
                'segment_row_limit': SEGMENT_ROW_LIMIT, #4096,
                'auto_id': False}
if len(client.list_collections()) > 0:
    client.drop_collection(collection_name)
client.create_collection(collection_name, p_collection)


# Info & stats
pprint(client.get_collection_info(collection_name))
pprint(client.get_collection_stats(collection_name))


# Delete previous parition and Create partition
if len(client.list_partitions(collection_name)) > 1:
    for tag in client.list_partitions(collection_name)[1:]:
        client.drop_partition(collection_name, tag)
client.create_partition(collection_name, partition_name)


# Generate data & insert entities
#vector_ids = list(range(N))
dump_sz = 262144 #(about 120mb)
data = np.random.rand(dump_sz, D).astype(np.float32) # store only first part for query

for i in tqdm(range(0, N, dump_sz), desc="Insert"):
    if i==0:
        d_part = data[i:(i+dump_sz)]
        i_part = list(range(i, i+dump_sz))
    elif i < N: 
        d_part = np.random.rand(dump_sz, D).astype(np.float32)
        i_part = list(range(i, i+dump_sz)) # vector_ids[i:(i+dump_sz)]
        
    p_entities = [{"name": "emb",
                   "values": d_part,
                   "type":DataType.FLOAT_VECTOR}]
    
    _ = client.insert(collection_name, p_entities, i_part, partition_name)
#result = client.get_entity_by_id(collection_name, [0,1,2,3,4]).dict()[0]['values']


# Create index
t0=time.time()
p_index = {"index_type": INDEX_TYPE,
           "metric_type": "L2",
           "params": {"nlist": N_CENTROIDS, "m": CODE_SZ}}
client.create_index(collection_name, 'emb', p_index)
print('Index training done - {} sec'.format(time.time()-t0))


# Search
q = data[:20]
dsl = {"bool": {"must": [{"vector": { "emb": {"topk": 10,
                                              "query": q,
                                              "metric_type": "L2",
                                              "params": {"nprobe": N_PROBE_CENTROIDS}
                                              }
                                     }}]}}
res = client.search(collection_name, dsl, [partition_name], fields=["emb"])
ids = np.asarray([r.ids for r in res])
dists = np.asarray([r.distances for r in res]) 
pprint(ids); pprint(dists)

# %timeit -n 1 client.search(collection_name, dsl, [partition_name], fields=["emb"])
# IVF_SQ8
# GCP GPU   train:60 s, search (20 query, 8M DB/nlist=4096/m=32) 2.18 s
# GCP GPU   train:?? s, search (20 query, 16M DB/nlist=256/m=64) 7.28 s
# GCP GPU   train:59 s, search (20 query, 16M DB/nlist=512/m=32) 7.01 s
# GCP GPU   train:117 s, search (20 query, 16M DB/nlist=4096/m=32) 7.92 s
# GCP GPU   train:117 s, search (2000 query, 16M DB/nlist=4096/m=32) 12.3 s
# GCP GPU train:? s, search (2000 query, 16M DB/nlist=256/m=64): ?s 
# GCP GPU train:? s, search (20 query, 16M DB/nlist=256/m=64): ?s 
# AMD 3900X train:25 s, search (20 query, 16M DB/nlist=256/m=64): 3.12 s 
# AMD 3900X train:108 s, search (20 query, 16M DB/nlist=1024/m=32): 2.97 s 
# AMD 3900X train:1127 s, search (20 query, 16M DB/nlist=4096/m=32): 2.91 s 
# AMD 3900X train:147 s, search (2000 query, 16M DB/nlist=256/m=64): 58.6 s 
# AMD 3900X train:147 s, search (20 query, 16M DB/nlist=256/m=64): 25.3 s 
#
# IVFPQ
# AMD 3900X train:704 s, search (20 query, 56M DB/nlist=256/m=64): 19.5 s
# GPU 5.6M 256/64 --> BaseException: <BaseException: (code=1, message=Failed to build segment 28 for collection fp, reason: Knowhere failed to build index: Error in void faiss::gpu::GpuIndexIVFPQ::verifySettings_() const at gpu/GpuIndexIVFPQ.cu:454: Error: 'requiredSmemSize <= getMaxSharedMemPerBlock(device_)' failed: Device 0 has 49152 bytes of shared memory, while 8 bits per code and 64 sub-quantizers requires 65536 bytes. Consider useFloat16LookupTables and/or reduce parameters)>