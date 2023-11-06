"""Unit tests for the collection utils."""
import logging

import pytest

from queens.utils.collection_utils import CollectionObject

_logger = logging.getLogger(__name__)


def test_initialization():
    """Test the different initialization."""
    field_names = ["a", "b", "c"]
    collobj = CollectionObject(*field_names)
    assert set(field_names) == set(collobj.keys())


def test_create_collection_object_from_dict():
    """Test create_collection_from_dict."""
    field_names = ["a", "b", "c"]
    data = [1, 2, 3]
    data_dict = dict(zip(field_names, data))
    collobj = CollectionObject.create_collection_object_from_dict(data_dict)
    assert set(field_names) == set(collobj.keys())
    assert set(data) == set(collobj.values())


def test_add():
    """Test adding values."""
    field_names = ["a", "b", "c"]
    collobj = CollectionObject(*field_names)
    data = [1, 2, 3]
    data_dict = dict(zip(field_names, data))
    collobj.add(**data_dict)
    collobj.add(**data_dict)
    exact_response = dict(zip(field_names, [[d, d] for d in data]))
    assert exact_response == collobj.to_dict()


def test_add_failure():
    """Test if exception is raised."""
    field_names = ["a", "b", "c"]
    collobj = CollectionObject(*field_names)
    collobj.add(a=1)

    with pytest.raises(ValueError, match="Can not add value"):
        collobj.add(a=1)


def test_len():
    """Test len function of the object."""
    field_names = ["a", "b", "c"]
    collobj = CollectionObject(*field_names)
    data = [1, 2, 3]
    data_dict = dict(zip(field_names, data))

    assert len(collobj) == 0

    collobj.add(**data_dict)
    assert len(collobj) == 1

    collobj.add(a=1)
    assert len(collobj) == 1

    collobj.add(b=1, c=1)
    assert len(collobj) == 2


def test_indexing():
    """Test indexing."""
    field_names = ["a", "b", "c"]
    data = [[1, 1.1, 1.2], [2, 2.1, 2.2], [3, 3.1, 3.2]]
    data_dict = dict(zip(field_names, data))
    collobj = CollectionObject.create_collection_object_from_dict(data_dict)

    assert collobj[1].to_dict() == {"a": 1.1, 'b': 2.1, "c": 3.1}
    assert collobj[-1].to_dict() == {"a": 1.2, 'b': 2.2, "c": 3.2}
    assert collobj[1:].to_dict() == {"a": [1.1, 1.2], 'b': [2.1, 2.2], "c": [3.1, 3.2]}


def test_indexing_failure():
    """Test indexing failure."""
    field_names = ["a", "b", "c"]
    data = [[1, 1.1, 1.2], [2, 2.1, 2.2], [3, 3.1, 3.2]]
    data_dict = dict(zip(field_names, data))
    collobj = CollectionObject.create_collection_object_from_dict(data_dict)

    with pytest.raises(IndexError, match="index 6 out of range for size 3"):
        _logger.info(collobj[6])


def test_bool():
    """Test bool function."""
    field_names = ["a", "b", "c"]
    collobj = CollectionObject(*field_names)

    assert not bool(collobj)

    collobj.add(a=1)
    assert bool(collobj)

    collobj.add(b=1, c=1)
    assert bool(collobj)
