# Copyright (C) 2026 Andrea Bistacchi
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

# FracLine.py - Analysis of fractures along 1D scanlines

from qgis.core import (
    QgsProcessingException,
)

def validate_layer(layer):
    """
    Validates the input layer to ensure it contains only single-part segments
    with exactly two nodes.

    :param layer: The QgsVectorLayer to validate.
    :raises QgsProcessingException: If validation fails.
    """
    for feature in layer.getFeatures():
        geom = feature.geometry()

        # Check for multipart features
        if geom.isMultipart():
            raise QgsProcessingException(
                f'Feature {feature.id()} is a multipart geometry. Please use single part geometries.'
            )

        # Check for segments with exactly two nodes
        points = geom.asPolyline()
        if len(points) != 2:
            raise QgsProcessingException(
                f'Feature {feature.id()} is not a segment with exactly two nodes. It has {len(points)} nodes.'
            )
