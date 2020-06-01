from google.protobuf.descriptor import FieldDescriptor as FD
import warnings

class ConvertException(Exception):
    pass

from google.protobuf.any_pb2 import Any,_ANY
from google.protobuf.timestamp_pb2 import Timestamp,_TIMESTAMP
from google.protobuf.duration_pb2 import Duration,_DURATION

def get_json_attr(field,attr):
    msg_type = field.message_type
    if field.type == FD.TYPE_MESSAGE:
        if field.message_type == _TIMESTAMP:
            time = Timestamp()
            time.FromJsonString(attr)
            return time
        if field.message_type == _DURATION:
            duration = Duration()
            duration.FromJsonString(attr)
            return duration

        if msg_type.name == 'Any':
            any = Any()
            any.type_url = 'json2pb/' + '_'.join([any.type_url, field.name])
            any.value = str.encode(str(attr))
            return any
        return dict2pb(msg_type._concrete_class, attr)
    elif field.type == FD.TYPE_ENUM:
        return field.enum_type.values_by_name[attr].number if isinstance(attr, str) else attr
    elif field.type == FD.TYPE_BYTES:
        return str.encode(attr)
    else:
        return attr

def dict2pb(cls,json_dic):
    """
    Takes a class representing the ProtoBuf Message and fills it with data from
    the dict.
    """
    target = cls()
    if target.DESCRIPTOR.syntax == 'proto2':
        for field in target.DESCRIPTOR.fields:
            # not required
            if not field.label == field.LABEL_REQUIRED:
                continue
            # has default value
            if field.has_default_value:
                continue
            # required but missing,raise except
            if field.name not in json_dic or not json_dic[field.name]:
                raise ConvertException('Field "%s" missing from descriptor dictionary.'
                                       % field.name)
            # for extension in target.Extensions:
        for extension_name in target._extensions_by_name:
            extension = target._extensions_by_name[extension_name]
            if extension.name not in json_dic:
                if not extension.has_default_value:
                    continue
            cur_value = json_dic[extension.name] if extension.name in json_dic else extension.default_value
            if extension.label == FD.LABEL_REPEATED:
                x = [get_json_attr(extension,sub_val) for sub_val in cur_value]
                target.Extensions[extension].extend(x)
            else:
                target.Extensions[extension] = get_json_attr(extension,cur_value)


    for field in target.DESCRIPTOR.fields:
        if field.name not in json_dic:
            if not field.has_default_value:
                continue
        cur_value = json_dic[field.name] if field.name in json_dic else field.default_value
        if cur_value is None:
            continue
        if field.name == 'any_type':
            print(1)
        for oneof in target.DESCRIPTOR.oneofs:
            if target.WhichOneof(oneof.name) is None:
                pass
            else:
                if field in oneof.fields:
                    warn_msg = 'Oneof field {} alrady has value in {} field,now replaced by {} field!'.format(oneof.name,target.WhichOneof(oneof.name),field.name)
                    warnings.warn(warn_msg,SyntaxWarning)

        if field.label == FD.LABEL_REPEATED:
            if field.message_type:
                if field.message_type.name == field.name.capitalize()+'Entry':
                    for k, v in cur_value.items():
                        getattr(target, field.name)[k] = v
                else:
                    getattr(target, field.name).extend([get_json_attr(field,sub_val) for sub_val in cur_value])
        else:
            if field.type == FD.TYPE_MESSAGE:
                getattr(target,field.name).CopyFrom(get_json_attr(field,cur_value))
            else:
                setattr(target, field.name, get_json_attr(field,cur_value))
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