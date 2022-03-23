import numpy as np

import pqueens.visualization.grid_iterator_visualization as qvis
from pqueens.models import from_config_create_model
from pqueens.utils.process_outputs import process_ouputs, write_results

from .iterator import Iterator


class GridIterator(Iterator):
    """Grid Iterator to enable meshgrid evaluations with different axis scaling
    such as linear, log10 or ln.

    Attributes:
        model (model): Model to be evaluated by iterator
        grid_dict (dict): Dictionary containing grid information
        result_description (dict):  Description of desired results
        parameters (dict) :    dictionary containing parameter information
        num_parameters (int)          :   number of parameters to be varied
        samples (np.array):   Array with all samples
        output (np.array):   Array with all model outputs
        num_grid_points_per_axis (list):  list with number of grid points for each grid axis
        scale_type (list): list with string entries denoting scaling type for each grid axis
    """

    def __init__(
        self,
        model,
        result_description,
        global_settings,
        grid_dict,
        parameters,
        num_parameters,
    ):
        super(GridIterator, self).__init__(model, global_settings)
        self.grid_dict = grid_dict
        self.parameters = parameters
        self.result_description = result_description
        self.samples = None
        self.output = None
        self.num_grid_points_per_axis = []
        self.num_parameters = num_parameters
        self.scale_type = []

    @classmethod
    def from_config_create_iterator(cls, config, iterator_name=None, model=None):
        """Create grid iterator from problem description.

        Args:
            config (dict):       Dictionary with QUEENS problem description
            iterator_name (str): Name of iterator to identify right section
                                 in options dict (optional)
            model (model):       Model to use (optional)


        Returns:
            iterator (obj): GridIterator object
        """
        if iterator_name is None:
            method_options = config["method"]["method_options"]
        else:
            method_options = config[iterator_name]["method_options"]
        if model is None:
            model_name = method_options["model"]
            model = from_config_create_model(model_name, config)

        result_description = method_options.get("result_description", None)
        global_settings = config.get("global_settings", None)
        grid_dict = method_options.get("grid_design", None)
        parameters = config["parameters"]["random_variables"]
        num_parameters = len(grid_dict)

        # take care of wrong user input
        if num_parameters is None:
            raise RuntimeError("Number of parameters (num_parameters) not given by user!")

        # ---------------------- CREATE VISUALIZATION BORG ----------------------------
        qvis.from_config_create(config, iterator_name=iterator_name)

        return cls(
            model, result_description, global_settings, grid_dict, parameters, num_parameters
        )

    def eval_model(self):
        """Evaluate the model."""
        return self.model.evaluate()

    def pre_run(self):
        """Generate samples based on description in grid_dict."""
        # get variables from problem description (needed to design the grid)
        parameters = self.model.get_parameter()

        # Sanity check for random fields
        random_fields = parameters.get("random_fields", None)
        if random_fields is not None:
            raise RuntimeError(
                "The grid iterator is currently not implemented in conjunction with random fields."
            )

        # pre-allocate empty list for filling up with vectors of grid points as elements
        grid_point_list = []

        #  set up 1D arrays for each parameter (needs bounds and type of axis)
        for index, (parameter_name, parameter) in enumerate(self.parameters.items()):
            start_value = parameter["min"]
            stop_value = parameter["max"]
            data_type = parameter["type"]
            axis_type = self.grid_dict[parameter_name].get("axis_type", None)
            num_grid_points = self.grid_dict[parameter_name].get("num_grid_points", None)
            self.num_grid_points_per_axis.append(num_grid_points)
            self.scale_type.append(axis_type)

            # check user input
            if axis_type is None:
                raise RuntimeError(
                    "Scaling of axis not given properly by user (possible: 'lin', "
                    "'log10' and 'ln')"
                )

            if num_grid_points is None:
                raise RuntimeError(
                    " Number of grid points ('num_grid_points') not given properly by user "
                )

            if axis_type == 'lin':
                grid_point_list.append(
                    np.linspace(
                        start_value,
                        stop_value,
                        num=num_grid_points,
                        endpoint=True,
                        retstep=False,
                    )
                )

            if axis_type == 'log10':
                grid_point_list.append(
                    np.logspace(
                        np.log10(start_value),
                        np.log10(stop_value),
                        num=num_grid_points,
                        endpoint=True,
                        base=10,
                    )
                )

            if axis_type == "ln":
                grid_point_list.append(
                    np.logspace(
                        np.log(start_value),
                        np.log(stop_value),
                        num=num_grid_points,
                        endpoint=True,
                        base=np.e,
                    )
                )

            # handle data types different from float (default)
            if data_type == 'INT':
                grid_point_list[index] = grid_point_list[index].astype(int)

            elif data_type == 'FLOAT':
                pass

            else:
                raise RuntimeError(
                    " Datatype of parameter / random variable given by user not supported by "
                    " grid iterator (possible: 'FLOAT' or 'INT') "
                )

        if self.num_parameters == 1:
            # change to correct order of samples array
            self.samples = np.atleast_2d(grid_point_list[0]).T

        elif self.num_parameters == 2:
            # get mesh_grid coordinates
            grid_coord0, grid_coord1 = np.meshgrid(grid_point_list[0], grid_point_list[1])
            # flatten to 2D array
            self.samples = np.empty([np.prod(self.num_grid_points_per_axis), self.num_parameters])
            self.samples[:, 0] = grid_coord0.flatten()
            self.samples[:, 1] = grid_coord1.flatten()

        elif self.num_parameters == 3:
            # get mesh_grid coordinates
            grid_coord0, grid_coord1, grid_coord2 = np.meshgrid(
                grid_point_list[0], grid_point_list[1], grid_point_list[2]
            )
            # flatten to 2D array
            self.samples = np.empty([np.prod(self.num_grid_points_per_axis), self.num_parameters])
            self.samples[:, 0] = grid_coord0.flatten()
            self.samples[:, 1] = grid_coord1.flatten()
            self.samples[:, 2] = grid_coord2.flatten()

        else:
            raise ValueError("More than 3 grid parameters are currently not supported! Abort...")

    def core_run(self):
        """Evaluate the meshgrid on model."""
        self.model.update_model_from_sample_batch(self.samples)
        self.output = self.eval_model()

    def post_run(self):
        """Analyze the results."""
        if self.result_description is not None:
            results = process_ouputs(self.output, self.result_description, self.samples)
            if self.result_description["write_results"] is True:
                write_results(
                    results,
                    self.global_settings["output_dir"],
                    self.global_settings["experiment_name"],
                )

        # plot QoI over grid
        qvis.grid_iterator_visualization_instance.plot_QoI_grid(
            self.output,
            self.samples,
            self.num_parameters,
            self.num_grid_points_per_axis,
        )
