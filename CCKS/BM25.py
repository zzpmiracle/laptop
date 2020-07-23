from elasticsearch import Elasticsearch
import json
es= Elasticsearch()
from elasticsearch.helpers import bulk

index_name = 'ccks_medical'
setting = {
  "settings": {
    "index" : {
            "similarity" : {
              "my_similarity" : {
                "type" : "BM25",
                "b" : "0.75",
                "k1" : "1"
              }
            }
        },
        "number_of_replicas": 1,
        "number_of_shards": 1
  },
  "mappings": {
      "_doc":{
      "properties": {
          "_sourse":{
"type": "text","fields": {

        "message": {
"analyzer": "ik_max_word",
"search_analyzer": "ik_max_word",
          "type": "text",
          "similarity": "my_similarity"
        }
      }}}
    }}
}
if not es.indices.exists(index=index_name):
    # res = es.indices.create(index=index_name)
    res = es.indices.create(index=index_name, body=setting)
    print(res)
bulk_size = 1024
def index():
  bulk_list= []
  with open('./data/entity_kb.txt') as f:
    try:
      for line in f.readlines():
          e = json.loads(line.replace('\n', ''))
          if e['subject'] is None:
            continue
          if e['type'] == 'Publication':
              continue
          entity_dic = {}
          entity_dic['_index'] = index_name
          entity_dic['_id'] = e["subject_id"]
          entity_dic['_type'] = "_doc"
          entity_dic['message'] = e['subject']
          for d in e['data']:
            if d['object'] is not None:
              entity_dic['message'] += str(d['object'])
          # entity_dic['_sourse'] = {}
          # entity_dic['_sourse']['message'] = entity_dic.pop('message')
          bulk_list.append(entity_dic)
          # es.index(index=index_name,id=e["subject_id"],body=entity_dic)
          if len(bulk_list) == bulk_size:
            bulk(es,bulk_list,index=index_name)
            bulk_list = []
            # break
      bulk(es, bulk_list, index=index_name)
    except IOError as e:
      print(e)
# index()
def search():
    dsl = {
        'query': {
            'match': {
                'message': None
            }
        },
        'size': 10
    }
    with open('./data/dev.txt',encoding='UTF-8') as dev,open('result_medical.txt','w') as result:
        for line in dev.readlines():
            dsl['query']['match']['message'] = line.replace('\n','')
            key_word_search = es.search(index=index_name, body=dsl, _source=True)
            hits = key_word_search['hits']['hits']
            if not hits:
                print(line.replace('\n',''))
                result.write('0\n')
            else:
                result.write(hits[0]['_id'])
                result.write('\n')
search()