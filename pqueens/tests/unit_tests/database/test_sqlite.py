"""Test sqlite."""
import numpy as np
import pandas as pd
import pytest
import xarray as xr

from pqueens.database.sqlite import SQLite

pytestmark = pytest.mark.unit_tests


def compare_documents(reference_document, obtained_document):
    """Compare documents.

    Args:
        reference_document (dict): Reference value
        obtained_document (dict): Document to compare

    Returns:
        bool: True if the documents are equal
    """
    document_equal = []
    for key, value in reference_document.items():
        obtained_value = obtained_document[key]
        if isinstance(value, np.ndarray):
            document_equal.append(np.array_equal(value, obtained_value))
        elif isinstance(value, (xr.DataArray, pd.DataFrame)):
            document_equal.append(value.equals(obtained_value))
        else:
            document_equal.append(value == obtained_value)
    return all(document_equal)


@pytest.fixture(name="db_sqlite")
def fixture_db_sqlite(tmp_path):
    """Sqlite fixture."""
    db_dict = {
        "database": {
            "name": "test_db",
            "file": tmp_path.joinpath("../test_sqlite.db"),
            "reset_existing_db": False,
        }
    }
    db = SQLite.from_config_create_database(db_dict)
    return db


@pytest.fixture(name="document_1")
def fixture_document_1():
    """First document."""
    document_to_save_1 = {
        "id": 12,
        "str": "test_string",
        "double": 12.5,
        "bool": True,
        "list": ["1", 1, 2.5],
    }
    return document_to_save_1


@pytest.fixture(name="document_2")
def fixture_document_2():
    """Second document."""
    document_to_save_2 = {
        "nparray": np.ones((2, 1, 3)),
        "xr": xr.DataArray(data=np.array([1, 2, 3])),
        "pd": pd.DataFrame(data={'col1': [1, 2], 'col2': [3, 4]}),
    }
    return document_to_save_2


@pytest.fixture(name="document_3")
def fixture_document_3():
    """Third document."""
    document_to_save_3 = {
        "str": "test_string",
    }
    return document_to_save_3


def test_from_config_create(db_sqlite):
    """Test if fcc works."""
    assert isinstance(db_sqlite, SQLite)


def test_save_document(db_sqlite, document_1, document_2, document_3):
    """Test if documents can be saved without errors."""
    with db_sqlite:
        db_sqlite.save(document_1, "experiment_name", "table_1", 2)
        db_sqlite.save(document_2, "experiment_name", "table_1", 2)
        db_sqlite.save(document_3, "experiment_name", "table_1", 3)
        db_sqlite.save(document_3, "experiment_name", "table_2", 1)


def test_load_document(db_sqlite, document_1, document_2):
    """Test if loaded documents are correct."""
    documents = [document_1, document_2]
    with db_sqlite:
        for i, document in enumerate(documents):
            loaded_doc = db_sqlite.load("experiment_name", 2, "table_1", {"id": i + 12})
            assert compare_documents(document, loaded_doc)


def test_overwrie_document(db_sqlite):
    """Overwrite existing document."""
    with db_sqlite:
        db_sqlite.save(
            {"id": 14, "str": "second_string"},
            "experiment_name",
            "table_1",
            3,
        )
        loaded_doc = db_sqlite.load("experiment_name", 3, "table_1", {"id": 14})
        assert loaded_doc["str"] == "second_string"


def test_count_documents(db_sqlite):
    """Test count documents."""
    with db_sqlite:
        counted_documents = db_sqlite.count_documents("", 2, "table_1")
        assert counted_documents == 2
        counted_documents = db_sqlite.count_documents("", 3, "table_1")
        assert counted_documents == 1


def test_remove_document(db_sqlite):
    """Test if document is deleted properly."""
    with db_sqlite:
        assert db_sqlite.load("experiment_name", 2, "table_1", {"id": 12})
        db_sqlite.remove("", "table_1", 2, {"id": 12})
        assert not db_sqlite.load("experiment_name", 2, "table_1", {"id": 12})


def test_get_table_names(db_sqlite):
    """Test table names."""
    with db_sqlite:
        assert (
            db_sqlite._get_all_table_names()
            == list(db_sqlite.existing_tables.keys())
            == ['table_1', 'table_2']
        )


def test_delete_table(db_sqlite):
    """Test delete table."""
    with db_sqlite:
        db_sqlite._delete_table('table_2')
        assert (
            db_sqlite._get_all_table_names()
            == list(db_sqlite.existing_tables.keys())
            == ['table_1']
        )
