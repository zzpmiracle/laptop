# a ={
#  "_str": "b",
#  "_bin": "0a0a0a0a",
#  "_bool": True,
#  "_float": 1,
#  "sub": {
#  "field": "subfield",
#  "echo": [
#  {"text": "first"},
#  {"text": "second"}
#  ]
#  },
#  "_int": [10, 20, 30, 40],
#  "_enum": ["VALUE1", 20],
#  "str_list":["v0", "v1"],
#  "test.e_bool":False
#  }
# import json
# with open('a.json','w') as f:
#     json.dump(a,f)
from json2proto.person_3_pb2 import ComplexMessage
a = {"_str": "test string", "_bin": "0a0a0a0a","any_type":[{"this": 1},[1],'test',5], "_bool": True, "_float": 1, "_enum": ["VALUE1", 20],"sub": [{"field": "first_sub", "echo": [{"text": "first"}, {"text": "second"}],"sub_message":[1,3]},{"field": "second_sub", "echo": [{"text": "second"}, {"text": "third"}]}], "_int": [10, 20, 30, 40], "str_list": ["v0", "v1"], }

from json2proto.json2pb import json2pb
target = json2pb(ComplexMessage,a)
print(target)
