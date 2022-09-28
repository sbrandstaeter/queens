"""Utils to collect data during iterative processes."""
from pqueens.utils.print_utils import get_str_table


class CollectionObject:
    """Collection object which stores data.

    This object can be index by iteration i: `collection_object[i]` but also using the collected
    fields `collection_object.field1`.

    Attributes:
        lists of the data that is stored in the object
    """

    def __init__(self, *field_names):
        """Initialze the collection item.

        Args:
            field_names (tuple): Name of fields to be stored
        """
        for key in field_names:
            self.__dict__.update({key: []})

    @classmethod
    def create_collection_object_from_dict(cls, data_dict):
        """Create collection item from dict.

        Args:
            data_dict (dict): Dictionary with values to be stored in this object

        Returns:
            CollectionObject: collection object created from dict
        """
        collection_object = cls()
        collection_object.__dict__.update(data_dict)
        return collection_object

    def add(self, **field_names_and_values):
        """Add data to the object.

        This function can be called with one or multiple fields, i.e.:
        `collection_object.add(field1=value1)` or
        `collection_object.add(field1=value1, field2=value2)`. An error
        is raised if one tries to add data to a field for a new
        iteration before all fields are filled for the current iteraton.
        """
        # Select only the fields to be stored the others are ignored
        field_names_to_be_stored = {
            key: field_names_and_values[key] for key in self.keys() & field_names_and_values.keys()
        }
        for key, value in field_names_to_be_stored.items():
            # Check if current iteration is completed
            if len(self.__dict__[key]) > min(self._get_lens()):
                fields_with_lens = ', '.join(
                    [f'{key}: {length}' for key, length in zip(self.keys(), self._get_lens())]
                )
                raise ValueError(
                    f"Can not add value to {key} list as it has length {len(self.__dict__[key])} "
                    f"but the other entries are at {len(self)}: {fields_with_lens}"
                )

            self.__dict__[key].append(value)

    def _get_lens(self):
        """Get number of elements per field.

        Returns:
            list: List of number of elements per field
        """
        return [len(value) for value in self.values()]

    def __str__(self):
        """Print table of current collection.

        Returns:
            str: Print table
        """
        return get_str_table(f"Collection object with {self.__len__()} iterations", self.__dict__)

    def values(self):
        """Values of the current object.

        This allows to use the object like a dict.

        Returns:
            dict_values: values of the collection object
        """
        return self.__dict__.values()

    def items(self):
        """Items of the current object.

        This allows to use the object like a dict.

        Returns:
            dict_items: items of the collection object
        """
        return self.__dict__.items()

    def keys(self):
        """Keys, i.e. field names, of the current object.

        This allows to use the object like a dict.

        Returns:
            dict_keys: keys of the collection object
        """
        return self.__dict__.keys()

    def __len__(self):
        """Len function for the object.

        Returns:
            int: number of complete iterations
        """
        lens = self._get_lens()

        if lens:
            return min(self._get_lens())

        # No data
        return 0

    def __getitem__(self, i):
        """Python intern method to index the collection object.

        Args:
            i (int, slice): int or slice

        Returns:
            CollectionObject: collection object with values and field names for provided indexes
        """
        if isinstance(i, int):
            if i > len(self):
                raise IndexError(f"index {i} out of range for size {self.__len__()}")

        new_dict = {}
        for key, value in self.items():
            new_dict.update({key: value[i]})
        return self.create_collection_object_from_dict(new_dict)

    def to_dict(self):
        """Create a dictionary from the collection object.

        Returns:
            dict: dictionary with all data
        """
        return self.__dict__

    def __bool__(self):
        """Bool value of the object.

        Returns:
            bool: returns True if data is stored in the object
        """
        lens = self._get_lens()
        return any(lens)