"""Test valid options utils."""
import pytest

from pqueens.utils.valid_options_utils import InvalidOptionError, check_if_valid_options, get_option


def test_check_if_valid_options_valid(requested_options_valid, valid_options):
    """Test *check_if_valid_options* for valid options."""
    assert check_if_valid_options(valid_options, requested_options_valid) is None


def test_check_if_valid_options_invalid(requested_options_invalid, valid_options):
    """Test *check_if_valid_options* for invalid options."""
    with pytest.raises(InvalidOptionError):
        check_if_valid_options(valid_options, requested_options_invalid)


def test_check_if_valid_options_error_message(valid_options):
    """Test error message raised by *check_if_valid_options*."""
    requested_option = "not_a_valid_option"
    try:
        check_if_valid_options(valid_options, requested_option, "Error estimating valid options")
    except InvalidOptionError as invalid_option_exception:
        error_message = str(invalid_option_exception)
        assert "Error estimating valid options" in error_message
        assert "Invalid option(s) 'not_a_valid_option'. Valid options are:" in error_message
        assert "a_valid_option, another_valid_option" in error_message


def test_get_option_valid(valid_options):
    """Test *get_option* for a valid option."""
    requested_option = "a_valid_option"
    assert get_option(valid_options, requested_option) == valid_options[requested_option]


def test_get_option_invalid(valid_options):
    """Test *get_option* for an invalid option."""
    requested_option = "not_a_valid_option"
    with pytest.raises(InvalidOptionError):
        get_option(valid_options, requested_option)


@pytest.fixture()
def valid_options():
    """Valid options fixture."""
    options = {
        "a_valid_option": "its response",
        "another_valid_option": "a different response",
    }
    return options


@pytest.fixture(
    params=[
        "not_a_valid_option",
        ["another_invalid_option", "a_valid_option"],
        ["another_invalid_option", "not_a_valid_option"],
        dict(zip(["not_a_valid_option", "another_valid_option"], [1, 2])),
        dict(zip(["another_ivalid_option", "another_valid_option"], [1, 2])),
    ]
)
def requested_options_invalid(request):
    """Invalid requested options."""
    return request.param


@pytest.fixture(
    params=[
        "a_valid_option",
        ["a_valid_option", "another_valid_option"],
        dict(zip(["a_valid_option", "another_valid_option"], [1, 2])),
    ]
)
def requested_options_valid(request):
    """Valid requested options."""
    return request.param
