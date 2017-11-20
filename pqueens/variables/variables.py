import numpy as np
class Variables(object):
    """ Class for storing variables

    For now basically only a wrapper around a dictionary
    At this point variables are simply a dictionary with three fields. The key
    is the name of the variable and then we have a 'type' field, a 'value'
    field and a 'active' field

    Attributes:
        variables (dict):  dictionary containing the data
    """
    @classmethod
    def from_uncertain_parameters_create(cls, uncertain_parameters):
        """ Create variables from uncertain parameter

        Args:
            uncertain_parameters (dict): Dictionary with uncertain parameters

        Returns:
            variables: Instance of variables object
        """
        values = []
        active = []
        for key, data in uncertain_parameters.items():
            values.append(None)
            active.append(True)
        return cls(uncertain_parameters, values, active)

    @classmethod
    def from_data_vector_create(cls, uncertain_parameters, data_vector):
        """ Create variables from uncertain parameter

        Args:
            uncertain_parameters (dict): Dictionary with uncertain parameters
            data_vector (np.array):      Vector with variable values

        Returns:
            variables: Instance of variables object
        """

        values = data_vector.tolist()
        active = [True]*len(values)

        return cls(uncertain_parameters, values, active)

    def __init__(self, uncertain_parameters, values, active):
        """ Initialize variable object

        Args:
            uncertain_parameters (dict): description of all uncertain params
            values (list):               list with variable values
            active (list):               list with flag whether or not variable
                                         is active
        """
        self.variables = {}
        i = 0
        for key, data in uncertain_parameters.items():
            self.variables[key] = {}
            self.variables[key]['value'] = values[i]
            self.variables[key]['type'] = data['type']
            self.variables[key]['active'] = active[i]
            i += 1

    def get_active_variables(self):
        """ Get dictinary of all active variables

        Returns:
            dict: dictionary with active variables, name and value only
        """
        active_vars = {}
        for key, data in self.variables.items():
            if data['active'] is not True:
                continue
            active_vars[key] = data['value']
        return active_vars

    def get_active_variables_vector(self):
        """ Get vector with values of all active variables

        Returns:
            np.array: vector with values of all active variables
        """
        active_var_vals = []
        for _, data in self.variables.items():
            if data['active'] is not True:
                continue
            active_var_vals.append(data['value'])
        return np.array(active_var_vals).reshape((-1, 1))

    def get_number_of_active_variables(self):
        """ Get number of currently active variables

        Returns:
            int: number of active variables
        """
        num_active_vars = 0
        for _, data in self.variables.items():
            if data['active'] is not True:
                continue
            num_active_vars += 1
        return num_active_vars


    def update_variables(self, new_variable_data):
        """ Update variable data

        Args:
            new_variable_data (dict): data to update the variables with
        """
        for key, _ in self.variables.items():
            self.variables[key]['value'] = new_variable_data[key]['value']
            self.variables[key]['active'] = new_variable_data[key]['active']
            self.variables[key]['type'] = new_variable_data[key]['type']

    def update_variables_from_vector(self, data_vector):
        """ Update variable values from vector

        Args:
            data_vector (np.array): Vector with new data for active variables
        """
        i = 0
        for key, _ in self.variables.items():
            self.variables[key]['value'] = data_vector[i]
            i += 1
        if i != len(data_vector):
            raise IndexError('The passed vector is to long!')