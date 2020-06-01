
from json2proto.person_3_pb2 import ComplexMessage
import json
a = {"_str": None,"duration":"103.0015001s","time":"2022-01-01T10:00:20.021-05:00","maps":{"key_1":"value_1","key_2":"value_2"} ,"_bin": "0a0a0a0a","any_type":[{"this": 1},[1],'test',5], "_bool": True, "_float": 1, "_enum": ["VALUE1", 20],"sub": [{"field": "first_sub", "echo": [{"text": "first"}, {"text": "second"}],"sub_message":[1,3]},{"field": "second_sub", "echo": [{"text": "second"}, {"text": "third"}]}], "_int": [10, 20, 30, 40], "str_list": ["v0", "v1"], }
with open('a.json','w') as f:
    json.dump(a,f)
from json2proto.json2pb import dict2pb
target = dict2pb(ComplexMessage,a)
print(target)
