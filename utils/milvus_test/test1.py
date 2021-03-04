# sudo docker run -d --name milvus_gpu_0.11.0 --gpus all \
# -p 19530:19530 \
# -p 19121:19121 \
# -v /home/skchang/work \
# -v /home/$USER/milvus/db:/var/lib/milvus/db \
# -v /home/$USER/milvus/conf:/var/lib/milvus/conf \
# -v /home/$USER/milvus/logs:/var/lib/milvus/logs \
# -v /home/$USER/milvus/wal:/var/lib/milvus/wal \
# milvusdb/milvus:0.11.0-gpu-d101620-4c44c0


# docker run -dti -v /home/skchang/work:/home/work -p 6600:6600 --network="host" --gpus all --shm-size 4G --ulimit memlock=819200000:819200000 --name skchang2 0216c564d1c7 /bin/bash

#python3 -m pip install pymilvus
#%%
from milvus import Milvus, DataType
from pprint import pprint
host = '127.0.0.1'
port = '19530'
client = Milvus(host, port)

# Delete
#client.drop_collection('fp')

# Create collection
collection_name = 'fp'
field_name = 'emb'

collection_param = {
    "fields": [
        {
            "name": "emb",
            "type": DataType.FLOAT_VECTOR,
            "params": {"dim": 128}
        },
    ],
    "segment_row_limit": 4096,
    "auto_id": True
}

client.create_collection(collection_name, collection_param)

# Info & stats
pprint(client.get_collection_info('fp'))
pprint(client.get_collection_stats('fp'))


# Create index
m=64
nlist=100
client.create_index(collection_name, field_name, {"index_type": "IVF_PQ", "metric_type": "L2", "params": {"nlist": nlist, "m": m}})



# Insert data
import numpy as np
data = np.random.random((10000,128))
hybrid_entities = [
    {
        "name": "emb",
        "type": DataType.FLOAT_VECTOR,
        "values": data
    },
]

client.insert(collection_name, hybrid_entities)

