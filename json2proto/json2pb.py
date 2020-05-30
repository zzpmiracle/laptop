import json
from google.protobuf.descriptor import FieldDescriptor as FD

class ConvertException(Exception):
    pass

def dict2pb(cls, json_dic, strict=False):
    """
    Takes a class representing the ProtoBuf Message and fills it with data from
    the dict.
    """
    target = cls()
    for field in target.DESCRIPTOR.fields:
        # not required
        if not field.label == field.LABEL_REQUIRED:
            continue
        # has default value
        if field.has_default_value:
            continue
        # required but missing,raise except
        if not field.name in json_dic:
            raise ConvertException('Field "%s" missing from descriptor dictionary.'
                                   % field.name)

    field_names = set([field.name for field in target.DESCRIPTOR.fields])
    # fields = set([field.label for field in target.DESCRIPTOR.fields])
    # print(fields)
    if strict:
        for key in json_dic.keys():
            if key not in field_names:
                raise ConvertException(
                    'Key "%s" can not be mapped to field in %s class.'
                    % (key, type(target)))

    for field in target.DESCRIPTOR.fields:
        if field.name not in json_dic:
            if not field.has_default_value:
                continue
        cur_value = json_dic[field.name] if field.name in json_dic else field.default_value
        msg_type = field.message_type
        if field.label == FD.LABEL_REPEATED:
            if field.type == FD.TYPE_MESSAGE:
                getattr(target, field.name).extend([dict2pb(msg_type._concrete_class,sub_dict) for sub_dict in cur_value])
            elif field.type == FD.TYPE_ENUM:
                getattr(target, field.name).extend([field.enum_type.values_by_name[sub_val].number if isinstance(sub_val,str) else sub_val for sub_val in cur_value])
            else:
                getattr(target,field.name).extend([str.encode(sub_val) if field.type == FD.TYPE_BYTES else sub_val for sub_val in cur_value])

        else:
            if field.type == FD.TYPE_MESSAGE:
                value = dict2pb(msg_type._concrete_class, cur_value)
                getattr(target, field.name).CopyFrom(value)
            elif field.type == FD.TYPE_BYTES:
                setattr(target, field.name, str.encode(cur_value))
            elif field.type == FD.TYPE_ENUM:
                setattr(target, field.name, field.enum_type.values_by_name[cur_value].number if isinstance(cur_value, str) else cur_value)
            else:
                setattr(target, field.name, cur_value)
    for extension_name in target._extensions_by_name:
        extension = target._extensions_by_name[extension_name]
        if extension.name not in json_dic:
            if not extension.has_default_value:
                continue
        cur_value = json_dic[extension.name] if extension.name in json_dic else extension.default_value
        msg_type = extension.message_type
        if extension.label == FD.LABEL_REPEATED:
            if extension.type == FD.TYPE_MESSAGE:
                target.Extensions[extension].extend([dict2pb(msg_type._concrete_class, sub_dict) for sub_dict in cur_value])
            elif extension.type == FD.TYPE_ENUM:
                target.Extensions[extension].extend([extension.enum_type.values_by_name[sub_val].number
                                                    if isinstance(sub_val, str) else sub_val for sub_val in cur_value])
            else:
                target.Extensions[extension].extend([str.encode(sub_val) if extension.type == FD.TYPE_BYTES else sub_val for sub_val in cur_value])
        else:
            if extension.type == FD.TYPE_MESSAGE:
                value = dict2pb(msg_type._concrete_class, cur_value)
                target.Extensions[extension].CopyFrom(value)
            elif extension.type == FD.TYPE_BYTES:
                target.Extensions[extension] =  str.encode(cur_value)
            elif extension.type == FD.TYPE_ENUM:
                target.Extensions[extension]= extension.enum_type.values_by_name[cur_value].number if isinstance(cur_value, str) else cur_value
            else:
                target.Extensions[extension] =  cur_value
    return target

def create_py(des_file_name,proto_file_name,class_name,json_file):
    with open(des_file_name,'w') as f:
        f.write('from {}_pb2 import {}\n'.format(proto_file_name.replace('.proto',''),class_name))
        f.write('from json2pb import dict2pb\n')
        f.write('import json\n')
        f.write('with open(\''+json_file+'\',\'r\') as f:\n\tdes_dic = json.load(f)\n')
        f.write('target = dict2pb({},des_dic)\n'.format(class_name))
        f.write('print(target)\n')
        f.write('Serialized_data = target.SerializeToString()\n')
        f.write('print(Serialized_data)\n')
        f.write('with open(\''+class_name+'.txt\',\'wb+\') as f:\n\tf.write(Serialized_data)')

if __name__ == '__main__':
    import sys
    des_file_name = sys.argv[1]
    proto_file_name = sys.argv[2]
    class_name = sys.argv[3]
    json_file = sys.argv[4]
    create_py(des_file_name, proto_file_name, class_name, json_file)
    print('.py file crated')