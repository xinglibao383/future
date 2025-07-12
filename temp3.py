import torchvision
torchvision.disable_beta_transforms_warning()

import warnings
warnings.filterwarnings("ignore")

from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection
from sentence_transformers import SentenceTransformer
import torch

# ========== 参数配置 ==========
MILVUS_HOST = "localhost"
MILVUS_PORT = "19530"
COLLECTION_NAME = "doc_demo"
EMBEDDING_DIM = 1024  # bge-large-zh 维度

# ========== 初始化向量模型 ==========
device = "cuda" if torch.cuda.is_available() else "cpu"
model = SentenceTransformer("/home/xinglibao/workspace/future/embedding/bge-large-zh-v1.5")

def embed_text(text):
    text = "为这个句子生成表示以用于检索: " + text
    embedding = model.encode(text, normalize_embeddings=True)
    return embedding

# ========== 连接 Milvus ==========
connections.connect("default", host=MILVUS_HOST, port=MILVUS_PORT)

# ========== 创建 Collection ==========
fields = [
    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
    FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=EMBEDDING_DIM),
    FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=1000)
]
schema = CollectionSchema(fields, description="Demo collection")
collection = Collection(name=COLLECTION_NAME, schema=schema)
collection.create_index(field_name="embedding", index_params={"index_type": "IVF_FLAT", "metric_type": "L2", "params": {"nlist": 128}})
collection.load()

# ========== 加载文档并插入 ==========
docs = [
    "注册会计师考试包括《会计》《审计》《税法》《财务成本管理》《经济法》《公司战略与风险管理》六门科目。",
    "考试时间一般安排在每年的十月。",
    "考生需要通过所有科目才能获得全科合格证书。",
    "一般程序员喜欢吃火锅",
]

embeddings = [embed_text(d) for d in docs]
insert_data = [embeddings, docs]
collection.insert(insert_data)
collection.flush()

print("✅ 向量已写入 Milvus！")

# ========== 向量查询 ==========
query_text = "注册会计师考什么内容？"
query_vector = embed_text(query_text)

results = collection.search(
    data=[query_vector],
    anns_field="embedding",
    param={"metric_type": "L2", "params": {"nprobe": 10}},
    limit=3,
    output_fields=["text"]
)

print("\n🔍 查询结果：")
for hit in results[0]:
    print(f"- 相似内容: {hit.entity.get('text')}（距离: {hit.distance:.4f}）")