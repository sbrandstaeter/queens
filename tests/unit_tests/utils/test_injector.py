"""Test injector util."""
import pytest

from queens.utils.exceptions import InjectionError
from queens.utils.injector import render_template


@pytest.mark.parametrize(
    "injection_parameters",
    [
        {"parameter_1": 1},
        {"parameter_1": 1, "wrong_parameter": 3},
        {"parameter_1": 1, "parameter_2": 2, "wrong_parameter": 3},
    ],
)
def test_failure_injection(injection_parameters):
    """Test if injection raises an error."""
    template = "{{ parameter_1 }} {{ parameter_2 }}"

    with pytest.raises(InjectionError):
        render_template(injection_parameters, template)


@pytest.mark.parametrize(
    "injection_parameters,expected_result",
    [
        ({"parameter_1": 1}, "1 "),
        ({"parameter_1": 1, "wrong_parameter": 3}, "1 "),
        ({"parameter_1": 1, "parameter_2": 2, "wrong_parameter": 3}, "1 2"),
    ],
)
def test_injection_strict_is_false(injection_parameters, expected_result):
    """Test injection with strict is False."""
    template = "{{ parameter_1 }} {{ parameter_2 }}"

    obtained_output = render_template(injection_parameters, template, strict=False)
    assert obtained_output == expected_result
