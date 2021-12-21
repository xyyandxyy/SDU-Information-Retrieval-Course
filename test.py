
# bert-serving-start -model_dir /tmp/xyy/bert_large -num_worker=2
from bert_serving.client import BertClient
bc = BertClient()
result = bc.encode(['I am your dad', 'yes', 'no'])
print(type(result))