from addressbook_pb2 import AddressBook
from json2pb import dict2pb
import json
with open('a.json','r') as f:
	des_dic = json.load(f)
target = dict2pb(AddressBook,des_dic)
Serialized_data = target.SerializeToString()
print(Serialized_data)
with open('AddressBook.txt','wb+') as f:
	f.write(Serialized_data)