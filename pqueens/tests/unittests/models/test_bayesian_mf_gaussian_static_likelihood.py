import pytest
import numpy as np
from mock import patch
from pqueens.models.likelihood_models.bayesian_mf_gaussian_static_likelihood import (
    BMFGaussianStaticModel,
)


# ------------ fixtures --------------------------


# ------------ unittests -------------------------
def test_init(mocker, default_interface):
    mp = mocker.patch('pqueens.models.model.Model.__init__')

    model = BMFGaussianStaticModel(
        model_name,
        model_parameters,
        nugget_noise_var,
        forward_model,
        coords_mat,
        time_vec,
        y_obs_vec,
        likelihood_noise_type,
        fixed_likelihood_noise_value,
        output_label,
        coord_labels,
        settings_probab_mapping,
        mf_interface,
        bmfia_subiterator,
        noise_upper_bound,
        x_train,
        y_hf_train,
        y_lfs_train,
        gammas_train,
        z_train,
        eigenfunc_random_fields,
        eigenvals,
        f_mean_train,
        noise_var,
        noise_var_lst,
    )

    # tests / asserts ----------------------------------
    mp.assert_called_once_with(name='bmfmc_model', uncertain_parameters=parameters, data_flag=True)
    assert model.model_parameters =
    assert model.nugget_noise_var =
    assert model.forward_model =
    assert model.coords_mat =
    assert model.time_vec =
    assert model.y_obs_vec =
    assert model.likelihood_noise_type =
    assert model.fixed_likelihood_noise_value =
    assert model.output_label =
    assert model.coord_labels =
    assert model.settings_probab_mapping =
    assert model.mf_interface =
    assert model.bmfia_subiterator =
    assert model.noise_upper_bound =
    assert model.x_train = None
    assert model.y_hf_train = None
    assert model.y_lfs_train = None
    assert model.gammas_train = None
    assert model.z_train = None
    assert model.eigenfunc_random_fields = None
    assert model.eigenvals = None
    assert model.f_mean_train = None
    assert model.noise_var = None
    assert model.noise_var_lst = []



def test_evaluate():
    pass


def test_evaluate_mf_likelihood():
    pass


def test_log_likelihood_fun():
    pass


def test_get_feature_mat():
    pass


def test_initialize():
    pass


def test_build_approximation():
    pass


def test_input_dim_red():
    pass


def test_get_random_fields_and_truncated_basis():
    pass


def test_update_and_evaluate_forward_model():
    pass


def test_project_samples_on_truncated_basis():
    pass
