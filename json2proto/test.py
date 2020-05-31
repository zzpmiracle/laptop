from person_3_pb2 import ComplexMessage
from json2pb import dict2pb
import json
with open('a.json','r') as f:
	des_dic = json.load(f)
target = dict2pb(ComplexMessage,des_dic)
print(target)
Serialized_data = target.SerializeToString()
print(Serialized_data)
with open('ComplexMessage.txt','wb+') as f:
	f.write(Serialized_data)