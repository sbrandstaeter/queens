"""Test SQLite."""

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
        bool: *True* if the documents are equal
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


data_to_store = [
    {"str": "test_string"},
    {"float": 12.5},
    {"bool": True},
    {"list": ["1", 1, 2.5]},
    {"nparray": np.ones((2, 1, 3))},
    {"xarray": xr.DataArray(data=np.array([1, 2, 3]))},
    {"pandas": pd.DataFrame(data={'col1': [1, 2], 'col2': [3, 4]})},
]


@pytest.fixture(params=data_to_store, name="data")
def fixture_data(request):
    """Fixture to request data sequentially."""
    return request.param


def create_db(tmp_path):
    """Sqlite fixture."""
    db_dict = {
        "database": {
            "name": "test_db",
            "file": tmp_path / "test_sqlite.db",
            "reset_existing_db": True,
        },
    }
    db = SQLite.from_config_create_database(db_dict)
    return db


def test_save_load_seperatly(data, tmp_path):
    """Test if documents are saved loaded correctly (sequential)."""
    db_sqlite = create_db(tmp_path)
    with db_sqlite:
        db_sqlite.save(data, "experiment_name", "table_1", 1)
        loaded_doc = db_sqlite.load("experiment_name", 1, "table_1")
        assert compare_documents(data, loaded_doc)


def test_save_load_common(tmp_path):
    """Test if documents are saved loaded correctly."""
    db_sqlite = create_db(tmp_path)
    document = {"id": 10}
    for field in data_to_store:
        document.update(field)

    with db_sqlite:
        db_sqlite.save(document, "experiment_name", "table_1", 1)
        loaded_document = db_sqlite.load("experiment_name", 1, "table_1", field_filters={"id": 10})
        assert compare_documents(document, loaded_document)


def test_from_config_create(tmp_path):
    """Test if fcc works."""
    db_sqlite = create_db(tmp_path)
    assert isinstance(db_sqlite, SQLite)


def test_overwrie_document(tmp_path):
    """Overwrite existing document."""
    db_sqlite = create_db(tmp_path)
    with db_sqlite:
        db_sqlite.save(
            {"id": 14, "str": "second_string"},
            "experiment_name",
            "table_1",
            3,
        )
        loaded_doc = db_sqlite.load("experiment_name", 3, "table_1", {"id": 14})
        assert loaded_doc["str"] == "second_string"


def test_count_documents(tmp_path):
    """Test count documents."""
    db_sqlite = create_db(tmp_path)
    with db_sqlite:
        for document in data_to_store:
            db_sqlite.save(document, "experiment_name", "table_1", 1)

        counted_documents = db_sqlite.count_documents("", 1, "table_1")
        assert counted_documents == len(data_to_store)


def test_remove_document(tmp_path):
    """Test if document is deleted properly."""
    db_sqlite = create_db(tmp_path)
    with db_sqlite:
        db_sqlite.save({"name": "queens", "id": 2}, "experiment_name", "table_1", 1)
        assert db_sqlite.load("experiment_name", 1, "table_1", {"id": 2})
        db_sqlite.remove("", "table_1", 1, {"id": 2})
        assert not db_sqlite.load("experiment_name", 1, "table_1", {"id": 2})


def test_get_table_names(tmp_path):
    """Test table names."""
    db_sqlite = create_db(tmp_path)
    with db_sqlite:
        db_sqlite.save({"name": "queens"}, "experiment_name", "table_1", 1)
        db_sqlite.save({"name": "queens"}, "experiment_name", "table_2", 1)
        assert (
            db_sqlite._load_table_names() == list(db_sqlite.tables.keys()) == ['table_1', 'table_2']
        )


def test_delete_table(tmp_path):
    """Test delete table."""
    db_sqlite = create_db(tmp_path)
    with db_sqlite:
        db_sqlite.save({"name": "queens"}, "experiment_name", "table_1", 1)
        db_sqlite.save({"name": "queens"}, "experiment_name", "table_2", 1)
        db_sqlite._delete_table('table_2')
        assert db_sqlite._load_table_names() == list(db_sqlite.tables.keys()) == ['table_1']


def test_remote_option_with_sqlite(tmp_path):
    """Test if error is raised for remote runs."""
    db_dict = {
        "database": {
            "name": "test_db",
            "reset_existing_db": False,
            "file": tmp_path / "../test_sqlite.db",
        },
        "interface": {"scheduler": {"remote": "True"}},
    }
    with pytest.raises(NotImplementedError):
        SQLite.from_config_create_database(db_dict)
