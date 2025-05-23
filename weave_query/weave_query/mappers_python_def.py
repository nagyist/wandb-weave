import dataclasses
import datetime
import inspect
import math
import typing

from weave_query import storage
from weave_query import weave_types as types
from weave_query import (
    artifact_fs,
    box,
    mappers,
    mappers_python,
    mappers_weave,
    ref_base,
    val_const,
    errors,
)
from weave_query import timestamp as weave_timestamp
from weave_query.language_features.tagging import tagged_value_type
from weave_query.partial_object import PartialObject, PartialObjectType


class TypedDictToPyDict(mappers_weave.TypedDictMapper):
    def apply(self, obj):
        result = {}
        for k, prop_serializer in self._property_serializers.items():
            result[k] = prop_serializer.apply(obj.get(k, None))
        return result


class DictToPyDict(mappers_weave.DictMapper):
    def apply(self, obj):
        result = {}
        for k, v in obj.items():
            k = self.key_serializer.apply(k)
            v = self.value_serializer.apply(v)
            result[k] = v
        return result


class ObjectToPyDict(mappers_weave.ObjectMapper):
    def apply(self, obj):
        # Store the type name in the saved object. W&B for example stores
        # {"_type": "table-ref", "artifact_path": "..."} objects in run config and summary
        # fields.
        result = {"_type": self.type.name}
        for prop_name, prop_serializer in self._property_serializers.items():
            if prop_serializer is not None:
                obj_val = getattr(obj, prop_name, None)
                if obj_val is None:
                    # Shortcut if there is a None here. In boards there are some cases where
                    # we have incorrect types that are missing optional designation. Fixes
                    # plotboard.cy.ts
                    result[prop_name] = None
                else:
                    result[prop_name] = prop_serializer.apply(obj_val)
        return result


class ObjectDictToObject(mappers_weave.ObjectMapper):
    def apply(self, obj):
        from weave_query.op_def_type import OpDefType

        # Only add keys that are accepted by the constructor.
        # This is used for Panels where we have an Class-level id attribute
        # that we want to include in the serialized representation.
        result = {}
        result_type = self._obj_type

        # TODO: I think these are hacks in my branch. What do they do?
        instance_class = result_type._instance_classes()[0]
        constructor_sig = inspect.signature(instance_class)
        for k, serializer in self._property_serializers.items():
            if serializer.type != OpDefType() and k in constructor_sig.parameters:
                obj_val = obj.get(k)
                if obj_val is None:
                    # Shortcut if there is a None here. In boards there are some cases where
                    # we have incorrect types that are missing optional designation. Fixes
                    # plotboard.cy.ts
                    result[k] = None
                else:
                    result[k] = serializer.apply(obj_val)

        for prop_name, prop_type in result_type.type_vars.items():
            if isinstance(prop_type, types.Const):
                result[prop_name] = prop_type.val

        # deserialize op methods separately
        op_methods = {}
        for k, serializer in self._property_serializers.items():
            if (
                obj.get(k) is not None
                and isinstance(serializer, DefaultFromPy)
                and serializer.type == OpDefType()
            ):
                op_methods[k] = serializer.apply(obj.get(k))

        if "artifact" in constructor_sig.parameters and "artifact" not in result:
            result["artifact"] = self._artifact
        try:
            # Construct a new class, inheriting from the original instance_class
            # with overridden op methods. The op_methods are unbound on the class,
            # and will bind self upon construction as usual.
            if self.type._relocatable:
                # Attach fields to the relocated object, so we can
                # detect and reconstruct later.
                new_class = type(
                    instance_class.__name__,
                    (instance_class,),
                    op_methods,
                )

                return new_class(**result)
            else:
                return instance_class(**result)
        except:
            err = errors.WeaveSerializeError(
                "Failed to construct %s with %s" % (instance_class, result)
            )
            err.fingerprint = ["failed-to-construct", instance_class, result]
            raise err


class GQLClassWithKeysToPyDict(mappers_weave.GQLMapper):
    def apply(self, obj: PartialObject):
        result = {}
        for k, prop_serializer in self._property_serializers.items():
            result[k] = prop_serializer.apply(obj.get(k, None))
        return result


class PyDictToGQLClassWithKeys(mappers_weave.GQLMapper):
    def apply(self, obj: dict) -> PartialObject:
        deserialized_obj = {}
        for k, prop_serializer in self._property_serializers.items():
            deserialized_obj[k] = prop_serializer.apply(obj.get(k, None))
        return self.type.keyless_weave_type_class.instance_class(deserialized_obj)


class ListToPyList(mappers_weave.ListMapper):
    def apply(self, obj):
        return [self._object_type.apply(item) for item in obj]


class UnionToPyUnion(mappers_weave.UnionMapper):
    def apply(self, obj):
        obj_type = types.type_of_with_refs(obj)
        for i, (member_type, member_mapper) in enumerate(
            zip(self.type.members, self._member_mappers)
        ):
            # TODO: Should types.TypeRegistry.type_of always return a const type??
            if isinstance(member_type, types.Const) and not isinstance(
                obj_type, types.Const
            ):
                obj_type = types.Const(obj_type, obj)

            # TODO: assignment isn't right here (a dict with 'a', 'b' int keys is
            # assignable to a dict with an 'a' int key). We want type equality.
            # But that breaks some stuff
            #
            # Later (3/8/23): This is even more ridiculous now. First we check
            # assignablility, then mergability. The ecosystem notebook now fails
            # without the merge check.
            if member_type.assign_type(obj_type) or not isinstance(
                types.merge_types(obj_type, member_type), types.UnionType
            ):
                result = member_mapper.apply(obj)
                if isinstance(result, dict):
                    result["_union_id"] = i
                else:
                    result = {"_union_id": i, "_val": result}
                return result
        raise Exception("invalid %s" % obj)


class PyUnionToUnion(mappers_weave.UnionMapper):
    def apply(self, obj):
        # Another hack for dealing with lack of union support in weavejs.
        if self.is_single_object_nullable and obj is None:
            return None

        try:
            has_union_id = "_union_id" in obj
        except TypeError:
            has_union_id = False

        # hack for deserializing instances of optional types from JS
        # todo: update JS to handle union_id so this is not needed
        if self.is_single_object_nullable and not has_union_id:
            if obj is None:
                return None
            else:
                non_null_mapper = next(
                    filter(
                        lambda m: not types.NoneType().assign_type(m.type),
                        self._member_mappers,
                    )
                )
                return non_null_mapper.apply(obj)
        member_index = obj["_union_id"]
        if "_val" in obj:
            obj = obj["_val"]
        else:
            obj.pop("_union_id")
        return self._member_mappers[member_index].apply(obj)


class IntToPyInt(mappers.Mapper):
    def apply(self, obj):
        return obj


class BoolToPyBool(mappers.Mapper):
    def apply(self, obj):
        if isinstance(obj, box.BoxedBool):
            return obj.val
        return obj


class FloatToPyFloat(mappers.Mapper):
    def apply(self, obj):
        if math.isnan(obj):
            return "nan"
        return obj


class PyFloatToFloat(mappers.Mapper):
    def apply(self, obj):
        if isinstance(obj, str):
            if obj == "nan":
                return float("nan")
        return obj


class StringToPyString(mappers.Mapper):
    def apply(self, obj):
        return obj


class TimestampToPyTimestamp(mappers.Mapper):
    def apply(self, obj: datetime.datetime):
        return weave_timestamp.python_datetime_to_ms(obj)


class PyTimestampToTimestamp(mappers.Mapper):
    def apply(self, obj):
        # This is here to support the legacy "date" type, the frontend passes
        # RFC 3339 formatted strings
        if isinstance(obj, dict):
            if obj.get("type") != "date":
                raise errors.WeaveInternalError(
                    f'expected object with type date but got "{obj}"'
                )
            return datetime.datetime.strptime(obj["val"], "%Y-%m-%dT%H:%M:%S.%fZ")
        else:
            return weave_timestamp.ms_to_python_datetime(obj)


class NoneToPyNone(mappers.Mapper):
    def apply(self, obj):
        return None


class UnknownToPyUnknown(mappers.Mapper):
    def apply(self, obj):
        # This should never be called. Unknown for the object type
        # of empty lists
        # PR: return None instead of crash
        return None
        raise Exception("invalid %s" % obj)


class RefToPyRef(mappers_weave.RefMapper):
    def __init__(
        self, type_: types.Type, mapper, artifact, path=[], use_stable_refs=True
    ):
        super().__init__(type_, mapper, artifact, path)
        self._use_stable_refs = use_stable_refs

    def apply(self, obj: typing.Any):
        if not isinstance(obj, ref_base.Ref):
            # type_of_with_refs(obj) returns a Ref Type, so that we'll
            # use this Ref mapper. We'll save the ref that points to the
            # object instead of a copy of the object.
            obj = ref_base.get_ref(obj)
            if obj is None:
                raise errors.WeaveSerializeError(
                    "Ref mapper cannot serialize non-ref object %s" % obj
                )

        try:
            if self._use_stable_refs:
                return obj.uri
            else:
                return obj.initial_uri
        except NotImplementedError:
            raise errors.WeaveSerializeError('Cannot serialize ref "%s"' % obj)


class PyRefToRef(mappers_weave.RefMapper):
    def apply(self, obj):
        return ref_base.Ref.from_str(obj)


class TypeToPyType(mappers.Mapper):
    def apply(self, obj):
        return obj.to_dict()


class PyTypeToType(mappers.Mapper):
    def apply(self, obj):
        return types.TypeRegistry.type_from_dict(obj)


class ConstToPyConst(mappers_weave.ConstMapper):
    def apply(self, obj):
        val = obj
        if isinstance(obj, val_const.Const):
            return val.val
        return self._val_mapper.apply(obj)


class DefaultToPy(mappers.Mapper):
    def __init__(
        self, type_: types.Type, mapper, artifact, path=[], use_stable_refs=True
    ):
        self.type = type_
        self._artifact = artifact
        self._path = path
        self._row_id = 0
        self._use_stable_refs = use_stable_refs

    def apply(self, obj):
        from weave_query import op_def

        try:
            return self.type.instance_to_dict(obj)
        except NotImplementedError:
            pass
        # If the ref exists elsewhere, just return its uri.
        # TODO: This doesn't deal with MemArtifactRef!
        # gc = weave_client_context.get_weave_client()
        gc = None  # Dropped as part of query service refactor

        existing_ref = storage._get_ref(obj)
        if isinstance(existing_ref, artifact_fs.FilesystemArtifactRef):
            if (
                # If we have a graph_client (weaveflow), only save
                # a nested ref here if it is a ref to the same storage
                # engine.
                not gc
                or (gc and gc._ref_is_own(existing_ref))
            ) and existing_ref.is_saved:
                if self._use_stable_refs:
                    uri = existing_ref.uri
                else:
                    uri = existing_ref.initial_uri
                return str(uri)

        ref = None

        if gc and isinstance(obj, op_def.OpDef):
            # This is a hack to ensure op_defs are always published as
            # top-level objects. This should be achieved by a policy
            # instead. There is a parallel policy in to_weavejs_with_refs
            # at the moment.
            ref = gc._save_object(obj, obj.name, "latest")
        elif isinstance(obj, ref_base.Ref):
            ref = obj
        elif isinstance(obj, str):
            try:
                ref = ref_base.Ref.from_str(obj)
            except (errors.WeaveInternalError, NotImplementedError):
                pass
        if ref is None:
            # This defines the artifact layout!
            name = "/".join(self._path + [str(self._row_id)])
            self._row_id += 1

            ref = self._artifact.set(name, self.type, obj)
        if ref.artifact == self._artifact:
            return ref.local_ref_str()
        else:
            return str(ref)


class DefaultFromPy(mappers.Mapper):
    def __init__(self, type_: types.Type, mapper, artifact, path=[]):
        self.type = type_
        self._artifact = artifact
        self._path = path

    def apply(self, obj):
        if isinstance(obj, dict):
            return self.type.instance_from_dict(obj)
        # else its a ref string
        # TODO: this does not use self.artifact, can we just drop it?
        # Do we need the type so we can load here? No...
        if ":" in obj:
            ref = ref_base.Ref.from_str(obj)
            # Note: we are forcing type here, because we already know it
            # We don't save the types for every file in a remote artifact!
            # But you can still reference them, because you have to get that
            # file through an op, and therefore we know the type?
            ref._type = self.type
            return ref.get()
        return self._artifact.get(obj, self.type)


py_type = type


@dataclasses.dataclass
class RegisteredMapper:
    type_class: type[types.Type]
    to_mapper: type[mappers.Mapper]
    from_mapper: type[mappers.Mapper]


_additional_mappers: list[RegisteredMapper] = []


def add_mapper(
    type_class: type[types.Type],
    to_mapper: type[mappers.Mapper],
    from_mapper: type[mappers.Mapper],
):
    _additional_mappers.append(RegisteredMapper(type_class, to_mapper, from_mapper))


def map_to_python_(type, mapper, artifact, path=[], mapper_options=None):
    mapper_options = mapper_options or {}
    use_stable_refs = mapper_options.get("use_stable_refs", True)
    if isinstance(type, types.TypeType):
        # If we're actually serializing a type itself
        return TypeToPyType(type, mapper, artifact, path)
    elif isinstance(type, PartialObjectType):
        return GQLClassWithKeysToPyDict(type, mapper, artifact, path)
    elif isinstance(type, types.TypedDict):
        return TypedDictToPyDict(type, mapper, artifact, path)
    elif isinstance(type, types.Dict):
        return DictToPyDict(type, mapper, artifact, path)
    elif isinstance(type, types.List):
        return ListToPyList(type, mapper, artifact, path)
    elif isinstance(type, types.UnionType):
        return UnionToPyUnion(type, mapper, artifact, path)
    elif isinstance(type, types.ObjectType):
        return ObjectToPyDict(type, mapper, artifact, path)
    elif isinstance(type, tagged_value_type.TaggedValueType):
        return tagged_value_type.TaggedValueToPy(type, mapper, artifact, path)
    elif isinstance(type, types.Boolean):
        return BoolToPyBool(type, mapper, artifact, path)
    elif isinstance(type, types.Int):
        return IntToPyInt(type, mapper, artifact, path)
    elif isinstance(type, types.Float):
        return FloatToPyFloat(type, mapper, artifact, path)
    elif isinstance(type, types.Number):
        return FloatToPyFloat(type, mapper, artifact, path)
    elif isinstance(type, types.String):
        return StringToPyString(type, mapper, artifact, path)
    elif isinstance(type, types.Timestamp):
        return TimestampToPyTimestamp(type, mapper, artifact, path)
    elif isinstance(type, types.Const):
        return ConstToPyConst(type, mapper, artifact, path)
    elif isinstance(type, types.NoneType):
        return NoneToPyNone(type, mapper, artifact, path)
    elif isinstance(type, types.UnknownType):
        return UnknownToPyUnknown(type, mapper, artifact, path)
    elif isinstance(type, types.RefType):
        return RefToPyRef(type, mapper, artifact, path, use_stable_refs=use_stable_refs)
    else:
        for m in _additional_mappers:
            if isinstance(type, m.type_class):
                return m.to_mapper(type, mapper, artifact, path)
        return DefaultToPy(
            type, mapper, artifact, path, use_stable_refs=use_stable_refs
        )


def map_from_python_(type: types.Type, mapper, artifact, path=[], mapper_options=None):
    if isinstance(type, types.TypeType):
        # If we're actually serializing a type itself
        return PyTypeToType(type, mapper, artifact, path)
    elif isinstance(type, PartialObjectType):
        return PyDictToGQLClassWithKeys(type, mapper, artifact, path)
    elif isinstance(type, types.ObjectType):
        return ObjectDictToObject(type, mapper, artifact, path)
    elif isinstance(type, types.TypedDict):
        return TypedDictToPyDict(type, mapper, artifact, path)
    elif isinstance(type, types.Dict):
        return DictToPyDict(type, mapper, artifact, path)
    elif isinstance(type, types.List):
        return ListToPyList(type, mapper, artifact, path)
    elif isinstance(type, types.UnionType):
        return PyUnionToUnion(type, mapper, artifact, path)
    elif isinstance(type, tagged_value_type.TaggedValueType):
        return tagged_value_type.TaggedValueFromPy(type, mapper, artifact, path)
    elif isinstance(type, types.Boolean):
        return BoolToPyBool(type, mapper, artifact, path)
    elif isinstance(type, types.Int):
        return IntToPyInt(type, mapper, artifact, path)
    elif isinstance(type, types.Float):
        return PyFloatToFloat(type, mapper, artifact, path)
    elif isinstance(type, types.Number):
        return PyFloatToFloat(type, mapper, artifact, path)
    elif isinstance(type, types.String):
        return StringToPyString(type, mapper, artifact, path)
    elif isinstance(type, types.Timestamp):
        return PyTimestampToTimestamp(type, mapper, artifact, path)
    elif isinstance(type, types.LegacyDate):
        return PyTimestampToTimestamp(type, mapper, artifact, path)
    elif isinstance(type, types.Const):
        return ConstToPyConst(type, mapper, artifact, path)
    elif isinstance(type, types.NoneType):
        return NoneToPyNone(type, mapper, artifact, path)
    elif isinstance(type, types.UnknownType):
        return UnknownToPyUnknown(type, mapper, artifact, path)
    elif isinstance(type, types.RefType):
        return PyRefToRef(type, mapper, artifact, path)
    else:
        for m in _additional_mappers:
            if isinstance(type, m.type_class):
                return m.from_mapper(type, mapper, artifact, path)
        return DefaultFromPy(type, mapper, artifact, path)


mappers_python.map_to_python = mappers.make_mapper(map_to_python_)
mappers_python.map_from_python = mappers.make_mapper(map_from_python_)
