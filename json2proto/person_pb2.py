# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: person.proto

from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor.FileDescriptor(
  name='person.proto',
  package='',
  syntax='proto2',
  serialized_options=None,
  create_key=_descriptor._internal_create_key,
  serialized_pb=b'\n\x0cperson.proto\"\x1b\n\x0b\x45\x63hoRequest\x12\x0c\n\x04text\x18\x01 \x02(\t\"\x85\x03\n\x0e\x43omplexMessage\x12\x0c\n\x04_str\x18\x01 \x02(\t\x12\x0e\n\x06_float\x18\x02 \x01(\x02\x12\x18\n\x0c_int_default\x18\r \x02(\x12:\x02\x39\x39\x12\x0c\n\x04_int\x18\x03 \x03(\x12\x12\x0c\n\x04_bin\x18\x04 \x02(\x0c\x12\r\n\x05_bool\x18\x05 \x02(\x08\x12\'\n\x03sub\x18\n \x03(\x0b\x32\x1a.ComplexMessage.SubMessage\x12&\n\x05_enum\x18\x0b \x03(\x0e\x32\x17.ComplexMessage.SubEnum\x12\x10\n\x08str_list\x18\x0c \x03(\t\x1a>\n\nSubMessage\x12\r\n\x05\x66ield\x18\x01 \x02(\t\x12\x1a\n\x04\x65\x63ho\x18\x02 \x03(\x0b\x32\x0c.EchoRequest*\x05\x08\x64\x10\xc8\x01\"!\n\x07SubEnum\x12\n\n\x06VALUE1\x10\n\x12\n\n\x06VALUE2\x10\x14*\x05\x08\x64\x10\xc8\x01\x32\x1e\n\x05\x65_int\x12\x0f.ComplexMessage\x18g \x01(\x11\x32#\n\ne_int_list\x12\x0f.ComplexMessage\x18h \x03(\x11:\x1f\n\x06\x65_bool\x12\x0f.ComplexMessage\x18\x65 \x01(\x08:$\n\x0b\x65_bool_list\x12\x0f.ComplexMessage\x18\x66 \x03(\x08'
)


E_BOOL_FIELD_NUMBER = 101
e_bool = _descriptor.FieldDescriptor(
  name='e_bool', full_name='e_bool', index=0,
  number=101, type=8, cpp_type=7, label=1,
  has_default_value=False, default_value=False,
  message_type=None, enum_type=None, containing_type=None,
  is_extension=True, extension_scope=None,
  serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key)
E_BOOL_LIST_FIELD_NUMBER = 102
e_bool_list = _descriptor.FieldDescriptor(
  name='e_bool_list', full_name='e_bool_list', index=1,
  number=102, type=8, cpp_type=7, label=3,
  has_default_value=False, default_value=[],
  message_type=None, enum_type=None, containing_type=None,
  is_extension=True, extension_scope=None,
  serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key)

_COMPLEXMESSAGE_SUBENUM = _descriptor.EnumDescriptor(
  name='SubEnum',
  full_name='ComplexMessage.SubEnum',
  filename=None,
  file=DESCRIPTOR,
  create_key=_descriptor._internal_create_key,
  values=[
    _descriptor.EnumValueDescriptor(
      name='VALUE1', index=0, number=10,
      serialized_options=None,
      type=None,
      create_key=_descriptor._internal_create_key),
    _descriptor.EnumValueDescriptor(
      name='VALUE2', index=1, number=20,
      serialized_options=None,
      type=None,
      create_key=_descriptor._internal_create_key),
  ],
  containing_type=None,
  serialized_options=None,
  serialized_start=326,
  serialized_end=359,
)
_sym_db.RegisterEnumDescriptor(_COMPLEXMESSAGE_SUBENUM)


_ECHOREQUEST = _descriptor.Descriptor(
  name='EchoRequest',
  full_name='EchoRequest',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='text', full_name='EchoRequest.text', index=0,
      number=1, type=9, cpp_type=9, label=2,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto2',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=16,
  serialized_end=43,
)


_COMPLEXMESSAGE_SUBMESSAGE = _descriptor.Descriptor(
  name='SubMessage',
  full_name='ComplexMessage.SubMessage',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='field', full_name='ComplexMessage.SubMessage.field', index=0,
      number=1, type=9, cpp_type=9, label=2,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='echo', full_name='ComplexMessage.SubMessage.echo', index=1,
      number=2, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=True,
  syntax='proto2',
  extension_ranges=[(100, 200), ],
  oneofs=[
  ],
  serialized_start=262,
  serialized_end=324,
)

_COMPLEXMESSAGE = _descriptor.Descriptor(
  name='ComplexMessage',
  full_name='ComplexMessage',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='_str', full_name='ComplexMessage._str', index=0,
      number=1, type=9, cpp_type=9, label=2,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='_float', full_name='ComplexMessage._float', index=1,
      number=2, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='_int_default', full_name='ComplexMessage._int_default', index=2,
      number=13, type=18, cpp_type=2, label=2,
      has_default_value=True, default_value=99,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='_int', full_name='ComplexMessage._int', index=3,
      number=3, type=18, cpp_type=2, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='_bin', full_name='ComplexMessage._bin', index=4,
      number=4, type=12, cpp_type=9, label=2,
      has_default_value=False, default_value=b"",
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='_bool', full_name='ComplexMessage._bool', index=5,
      number=5, type=8, cpp_type=7, label=2,
      has_default_value=False, default_value=False,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='sub', full_name='ComplexMessage.sub', index=6,
      number=10, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='_enum', full_name='ComplexMessage._enum', index=7,
      number=11, type=14, cpp_type=8, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='str_list', full_name='ComplexMessage.str_list', index=8,
      number=12, type=9, cpp_type=9, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
  ],
  extensions=[
    _descriptor.FieldDescriptor(
      name='e_int', full_name='ComplexMessage.e_int', index=0,
      number=103, type=17, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=True, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='e_int_list', full_name='ComplexMessage.e_int_list', index=1,
      number=104, type=17, cpp_type=1, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=True, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
  ],
  nested_types=[_COMPLEXMESSAGE_SUBMESSAGE, ],
  enum_types=[
    _COMPLEXMESSAGE_SUBENUM,
  ],
  serialized_options=None,
  is_extendable=True,
  syntax='proto2',
  extension_ranges=[(100, 200), ],
  oneofs=[
  ],
  serialized_start=46,
  serialized_end=435,
)

_COMPLEXMESSAGE_SUBMESSAGE.fields_by_name['echo'].message_type = _ECHOREQUEST
_COMPLEXMESSAGE_SUBMESSAGE.containing_type = _COMPLEXMESSAGE
_COMPLEXMESSAGE.fields_by_name['sub'].message_type = _COMPLEXMESSAGE_SUBMESSAGE
_COMPLEXMESSAGE.fields_by_name['_enum'].enum_type = _COMPLEXMESSAGE_SUBENUM
_COMPLEXMESSAGE_SUBENUM.containing_type = _COMPLEXMESSAGE
DESCRIPTOR.message_types_by_name['EchoRequest'] = _ECHOREQUEST
DESCRIPTOR.message_types_by_name['ComplexMessage'] = _COMPLEXMESSAGE
DESCRIPTOR.extensions_by_name['e_bool'] = e_bool
DESCRIPTOR.extensions_by_name['e_bool_list'] = e_bool_list
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

EchoRequest = _reflection.GeneratedProtocolMessageType('EchoRequest', (_message.Message,), {
  'DESCRIPTOR' : _ECHOREQUEST,
  '__module__' : 'person_pb2'
  # @@protoc_insertion_point(class_scope:EchoRequest)
  })
_sym_db.RegisterMessage(EchoRequest)

ComplexMessage = _reflection.GeneratedProtocolMessageType('ComplexMessage', (_message.Message,), {

  'SubMessage' : _reflection.GeneratedProtocolMessageType('SubMessage', (_message.Message,), {
    'DESCRIPTOR' : _COMPLEXMESSAGE_SUBMESSAGE,
    '__module__' : 'person_pb2'
    # @@protoc_insertion_point(class_scope:ComplexMessage.SubMessage)
    })
  ,
  'DESCRIPTOR' : _COMPLEXMESSAGE,
  '__module__' : 'person_pb2'
  # @@protoc_insertion_point(class_scope:ComplexMessage)
  })
_sym_db.RegisterMessage(ComplexMessage)
_sym_db.RegisterMessage(ComplexMessage.SubMessage)

ComplexMessage.RegisterExtension(e_bool)
ComplexMessage.RegisterExtension(e_bool_list)
ComplexMessage.RegisterExtension(_COMPLEXMESSAGE.extensions_by_name['e_int'])
ComplexMessage.RegisterExtension(_COMPLEXMESSAGE.extensions_by_name['e_int_list'])

# @@protoc_insertion_point(module_scope)
