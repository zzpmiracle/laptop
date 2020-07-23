# with open('train.txt') as f,open('train_utf.txt','w',encoding='utf-8') as tu:
#     for line in f.readlines():
#         tu.write(line.encode('utf-8').decode('unicode_escape').replace('\r\n', ''))
#
import json

types  = {}
with open('entity_kb.txt','r') as tu:
    for line in tu.readlines():
        try:
            line_dic = json.loads(line)
            if line_dic['type'] not in types.keys():
                types[line_dic['type']]  = 1
            else:
                types[line_dic['type']] += 1
        except:
            print(line)
            pass
print(types)