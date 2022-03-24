"""Test valid options utils."""
import pytest

from pqueens.utils.valid_options_utils import InvalidOptionError, check_if_valid_option, get_option


@pytest.mark.unit_tests
def test_check_if_valid_option_valid(valid_options):
    """Test check_if_valid_option for a valid option."""
    requested_option = "a_valid_option"
    assert check_if_valid_option(valid_options, requested_option) == True


@pytest.mark.unit_tests
def test_check_if_valid_option_invalid(valid_options):
    """Test check_if_valid_option for an invalid option."""
    requested_option = "not_a_valid_option"
    with pytest.raises(InvalidOptionError):
        check_if_valid_option(valid_options, requested_option)


@pytest.mark.unit_tests
def test_check_if_valid_option_error_message(valid_options):
    """Test error message raised by check_if_valid_option."""
    requested_option = "not_a_valid_option"
    try:
        check_if_valid_option(valid_options, requested_option, "Error estimating valid options")
    except Exception as invalid_option_exception:
        error_message = str(invalid_option_exception)
        assert "Error estimating valid options" in error_message
        assert "Invalid option 'not_a_valid_option'. Valid options are:" in error_message
        assert "a_valid_option, another_valid_option" in error_message


@pytest.mark.unit_tests
def test_get_option_valid(valid_options):
    """Test get_option for a valid option."""
    requested_option = "a_valid_option"
    assert get_option(valid_options, requested_option) == valid_options[requested_option]


@pytest.mark.unit_tests
def test_get_option_invalid(valid_options):
    """Test get_option for an invalid option."""
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
