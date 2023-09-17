# -*- coding: utf-8 -*-
# Copyright 2007-2023 The HyperSpy developers
#
# This file is part of HyperSpy.
#
# HyperSpy is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# HyperSpy is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with HyperSpy. If not, see <https://www.gnu.org/licenses/#GPL>.

from hyperspy.drawing.markers import Markers
from matplotlib.collections import LineCollection
import numpy as np


class VerticalLines(Markers):
    """A set of Vertical Line Markers"""

    marker_type = "VerticalLines"

    def __init__(
        self, offsets, offsets_transform="display", **kwargs
    ):
        """
        Initialize the set of Vertical Line Markers.

        Parameters
        ----------
        x: [n]
            Positions of the markers
        kwargs: dict
            Keyword arguments passed to the underlying marker collection. Any argument
            that is array-like and has `dtype=object` is assumed to be an iterating
            argument and is treated as such.
        """
        if "transform" in kwargs and kwargs["transform"] != "xaxis":
            raise ValueError(
                "VerticalLines markers must have transform='xaxis'."
            )
        transform = "xaxis"

        # Data attributes
        Markers.__init__(
            self,
            offsets=offsets,
            offsets_transform=offsets_transform,
            transform=transform,  # so that the markers span the whole y-axis
            collection=LineCollection,
            **kwargs
        )

    def get_data_position(self, get_static_kwargs=True):
        kwargs = super().get_data_position(get_static_kwargs=get_static_kwargs)
        x_pos = kwargs.pop("offsets")
        new_segments = np.array(
            [
                [
                    [x, 0],
                    [x, 1],
                ]
                for x in x_pos
            ]
        )
        kwargs["segments"] = new_segments
        return kwargs
