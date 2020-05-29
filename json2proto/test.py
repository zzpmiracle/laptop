a ={
 "_str": "b",
 "_bin": "0a0a0a0a",
 "_bool": True,
 "_float": 1,
 "sub": {
 "field": "subfield",
 "echo": [
 {"text": "first"},
 {"text": "second"}
 ]
 },
 "_int": [10, 20, 30, 40],
 "_enum": ["VALUE1", 20],
 "str_list":["v0", "v1"],
 "test.e_bool":False
 }
import json
with open('a.json','w') as f:
    json.dump(a,f)