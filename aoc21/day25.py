"""Solver for Advent of Code 2021, Day 25

https://adventofcode.com/2021/day/25

"""
from collections.abc import Sequence, Callable, Generator
from typing import Literal, get_args

import numpy as np
from scipy.ndimage import generic_filter  # type: ignore


# These are the characters used in the ascii maps. In theory, we could handle maps
# that use other symbols by changing the ordinals to match.
# To satisfy mypy we create the return type for the `_move` function
# (`_move_direction` factory method, `Step` generator class) from the ordinals...
MapOrdinalsLiteral = Literal[46, 62, 118]  # '.', '>', 'v'
# ...then create the dictionary that matches them to their symbols from the type
# (https://stackoverflow.com/a/64522240)
SYMBOLS_TO_ORDINALS = dict(
    zip(("empty", "east", "south"), get_args(MapOrdinalsLiteral))
)

# TODO: Figure out a way to handle cases where |difference(south-east)| and
# |difference(east-empty)| or |difference(south-empty)| are equal, because in those
# cases the method used in this solution will not be able to tell if there is an
# empty space or just the other type of cucumber ahead of a position. This is
# necessary to fully support general combinations of characters for each type.


def ascii_map_to_ndarray(ascii_map: str) -> np.ndarray:
    """Load 1-D or 2-D ascii 'map' into ndarray of character ordinals

    Parameters
    ----------
    ascii_map : str
        String containing a rectangular ascii 'map' of cucumbers and empty spaces.

    Returns
    -------
    array_map : numpy.ndarray
        Each element is the ordinal that corresponds to the character in the
        same position (row, column) of the ascii map.

    """
    array_map = np.asarray(
        # Split the ascii into lines (rows), then split each line into individual
        # characters and get the ordinal of each character to form an array of
        # unsigned 8-bit integers
        [[ord(char) for char in line] for line in ascii_map.split("\n") if line],
        dtype=np.uint8,
    )

    return array_map


def check_for_extraneous_values(
    array: np.ndarray,
    allowed_values: Sequence[int] = list(SYMBOLS_TO_ORDINALS.values()),
) -> None:
    if np.isin(array, np.array([*allowed_values], dtype=np.uint8), invert=True).any():
        raise ValueError("Not all values are allowed")


# TODO: Merge `check_for_extraneous_values` into `ascii_map_to_ndarray`


class Step(Generator):
    """Subclass of the built-in Generator abstract base class that yields a
    seafloor map after each cucumber move cycle and keeps track of the number of
    steps taken

    Parameters
    ----------
    initial_map : numpy.ndarray

    Attributes
    ----------
    move_east : Callable
        This exists to prevent mypy from complaining that it can't find this
        attribute, since it is set during class instantiation from the
        `_move_direction` factory method.
    move_south : Callable
        This exists to prevent mypy from complaining that it can't find this
        attribute, since it is set during class instantiation from the
        `_move_direction` factory method.

    Methods
    -------
    _move_direction(direction: Literal["east", "south"]) -> Callable, staticmethod
        Factory method used to create direction-specific move methods for use
        in calls to scipy.ndimage.generic_filter.
    send(ignored_value=None) -> numpy.ndarray
        Calculates the map for the next step and either returns that map if it
        is different from the previous step's map, or calls the `throw` method
        to trigger generator exit.
    throw(typ=None, val=none, tb=None) -> None
        Raises a StopIteration that exits the generator and includes the number of
        steps taken for thecucumbers to stop moving.

    """

    # This prevents mypy from complaining about these two functions which are set
    # during class instantiation
    move_east: Callable
    move_south: Callable

    def __init__(self, initial_map: np.ndarray) -> None:
        self.step = 0
        self.starting_map = initial_map

        # Set direction-specific move functions for use with
        # scipy.ndimage.generic_filter
        setattr(self.__class__, "move_east", staticmethod(self._move_direction("east")))
        setattr(
            self.__class__, "move_south", staticmethod(self._move_direction("south"))
        )

    @staticmethod
    def _move_direction(
        direction: Literal["east", "south"]
    ) -> Callable[[np.ndarray], MapOrdinalsLiteral]:
        """Factory method that returns direction-specific move functions

        Instead of writing a separate function for each direction, we can curry the
        same general function to get direction-specific functions.

        Parameters
        ----------
        direction : Literal["east", "south"]

        Returns
        -------
        Callable[[numpy.ndarray], int]
            ...where the return value of the function is specifically an int in the
            MapOrdinalsLiteral defined at the beginning of the module.

        """
        cucumber = SYMBOLS_TO_ORDINALS[direction]
        empty_space = SYMBOLS_TO_ORDINALS["empty"]
        difference = cucumber - empty_space

        def _move(footprint: np.ndarray) -> MapOrdinalsLiteral:
            """Function sent to scipy.ndimage.generic_filter to be applied to each
            element of an array (seafloor map)

            The (absolute value of the) difference between two elements encodes the
            types of the two elements (empty, east-facing cucumber, south-facing
            cucumber). The sign of the difference reveals which of the two elements
            is which.

            For example, the difference between an east-facing cucumber (62) and an
            empty space (46) is 16. We always start with the current location
            (footprint[1]), so if we find a difference of 16 between the current
            location and the location ahead (footprint[2]), we know that the
            current location has an east-facing cucumber and the location ahead is
            empty. If we instead find a difference of -16, we know that it is the
            current location that is empty and the location ahead has the
            east-facing cucumber.

            Since scipy.ndimage.generic_filter uses a separate array for output--as
            opposed to making changes to the input array in-place--we can
            immediately perform half of the necessary swap operation to 'move' a
            cucumber when we either find one that is able to move in the direction
            *or* find an empty space with a cucumber behind. To do so, we record the
            'opposite' element at the current position of the output array. The
            other half of the swap simply gets performed when the filter reaches
            that element (in the input array), where it finds the inverse of the
            difference.

            Parameters
            ----------
            footprint : numpy.ndarray
                footprint[1] is the map location of interest
                footprint[0] is 'behind'
                footprint[2] is 'ahead'

            Returns
            -------
            int
                ...where int is specifically in the MOVE_RETURN_TYPE literal defined
                at the beginning of the module.

            """
            # Look ahead of the current position
            if footprint[1] - footprint[2] == difference:
                # Reaching here means we found a cucumber with an empty space in
                # front of it. We perform half the swap by changing the cucumber to
                # an empty space.
                return empty_space

            # Look behind the current position
            if footprint[1] - footprint[0] == -difference:
                # Reaching here means we found an empty space with a cucumber ready
                # to move into it. We perform half the swap by changing the empty
                # space to the cucumber.
                return cucumber

            # If we didn't find either spaces involved in a move, the current space
            # doesn't change
            return footprint[1]

        return _move

    def send(self, ignored_value=None) -> np.ndarray:
        # Increment the step counter
        self.step += 1

        # As per the problem description, the east-facing cucumbers move first
        after_move_east_map = generic_filter(
            self.starting_map,
            self.move_east,
            footprint=np.array([[0, 0, 0], [1, 1, 1], [0, 0, 0]]),
            output=np.uint8,
            mode="wrap",  # Cukes moving off one edge appear at the opposite edge
        )

        # Then the south-facing cucumbers
        ending_map = generic_filter(
            after_move_east_map,
            self.move_south,
            footprint=np.array([[0, 1, 0], [0, 1, 0], [0, 1, 0]]),
            output=np.uint8,
            mode="wrap",  # Cukes moving off one edge appear at the opposite edge
        )

        # Check if the resulting map is the same as the previous step;
        # if yes, exit the generator (and provide the number of steps it took for
        # the cucumbers to stop moving).
        if np.array_equal(ending_map, self.starting_map):
            self.throw()

        # In preparation for the next step, store the current map
        self.starting_map = ending_map

        # yield current map
        return ending_map

    def throw(self, typ=None, val=None, tb=None) -> None:
        raise StopIteration(self.step)


if __name__ == "__main__":
    import os
    from pathlib import Path

    # Absolute path to this file, used to find absolute path to input map
    module_path = os.path.dirname(os.path.realpath(__file__))

    initial_array_map = ascii_map_to_ndarray(
        # Read ascii map from file and convert to numpy array
        Path(os.path.join(module_path, "../input", "starting_map.txt")).read_text(
            encoding="ascii"
        )
    )

    check_for_extraneous_values(initial_array_map)

    # Initialize generator with starting map
    step = Step(initial_array_map)

    while True:
        try:
            next(step)
        # When the generator finds a map hasn't changed from the previous step
        # it will exit and return the number of steps via the exception
        except StopIteration as stopped_moving:
            print(f"The sea cucumbers stopped moving after {stopped_moving} steps")
            break  # We're done, don't want to iterate unchanging maps forever
