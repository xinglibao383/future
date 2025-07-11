from pymilvus import *

# 如果你在容器里用 --network=host 运行，host 就是 localhost
connections.connect("default", host="localhost", port="19530")

# 测试是否连接成功
print(connections.get_connection_addr("default"))

# 定义字段
fields = [
    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=False),
    FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=128)
]

# 创建 Collection Schema
schema = CollectionSchema(fields, description="Example collection")

# 创建 Collection
collection = Collection(name="example_collection", schema=schema)

# 插入数据
import numpy as np

ids = [i for i in range(100)]
vectors = np.random.random((100, 128)).tolist()

collection.insert([ids, vectors])

# 创建索引
collection.create_index(field_name="embedding", index_params={
    "metric_type": "L2",
    "index_type": "IVF_FLAT",
    "params": {"nlist": 128}
})

# 加载 collection 到内存
collection.load()

# 搜索
search_result = collection.search(
    data=[vectors[0]], 
    anns_field="embedding", 
    param={"metric_type": "L2", "params": {"nprobe": 10}},
    limit=5,
    output_fields=["id"]
)

for hits in search_result:
    for hit in hits:
        print(f"id: {hit.id}, distance: {hit.distance}")