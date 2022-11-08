# TODO: Document and comment ;)

import os
from pathlib import Path

from collections.abc import Callable

import pytest
from numpy.testing import assert_array_equal

import numpy as np

from aoc21.day25 import (
    ascii_map_to_ndarray,
    check_for_extraneous_values,
    Step,
)


_module_path = os.path.dirname(os.path.realpath(__file__))


@pytest.fixture
def fixture_full_example_array_map() -> Callable:
    def _fixture_full_example_array_map(step) -> np.ndarray:
        return ascii_map_to_ndarray(
            Path(
                os.path.join(_module_path, f"example_maps/example_step{step}_map.txt")
            ).read_text()
        )

    return _fixture_full_example_array_map


def test_ascii_map_to_ndarray() -> None:
    ascii_map = "v>.v\n" ">.>.\n" ".vv>\n" ".>.."

    actual = ascii_map_to_ndarray(ascii_map)

    desired = np.asarray(
        [
            [118, 62, 46, 118],
            [62, 46, 62, 46],
            [46, 118, 118, 62],
            [46, 62, 46, 46],
        ]
    )

    assert_array_equal(actual, desired)


def test_check_for_extraneous_values_passes() -> None:
    good_array = np.array([46, 62, 118], dtype=np.uint8)

    try:
        check_for_extraneous_values(good_array)
    except ValueError:
        assert False


def test_check_for_extraneous_values_fails() -> None:
    bad_array = np.array([46, 62, 119], dtype=np.uint8)

    with pytest.raises(ValueError):
        check_for_extraneous_values(bad_array)


@pytest.mark.parametrize("steps", [1, 2, 3, 4, 5, 10, 20, 30, 40, 50, 55, 56, 57])
def test_step_generator_intermediate_maps(
    steps, fixture_full_example_array_map
) -> None:
    expected_after_n_steps = fixture_full_example_array_map(steps)

    step = Step(fixture_full_example_array_map(0))

    for n in range(steps):
        output_after_n_steps = next(step)

    assert_array_equal(output_after_n_steps, expected_after_n_steps)


def test_step_generator_full_example_answer(fixture_full_example_array_map) -> None:
    initial_array_map = fixture_full_example_array_map(0)

    step = Step(initial_array_map)

    with pytest.raises(StopIteration) as err:
        while True:
            next(step)

        assert err.value == 58
