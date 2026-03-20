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

from qgis.core import (
    QgsProcessing,
    QgsProcessingAlgorithm,
    QgsProcessingParameterVectorLayer,
    QgsProcessingException
)

# Import and reload FracLine
from . import FracLine
import importlib
importlib.reload(FracLine)

class FracLinePlugin(QgsProcessingAlgorithm):
    INPUT_LAYER = 'INPUT_LAYER'

    def createInstance(self):
        return FracLinePlugin()

    def name(self):
        return 'fracline'

    def displayName(self):
        return 'FracLine: Analyze fractures along 1D scanlines'

    def group(self):
        return 'FracLine'

    def groupId(self):
        return 'fracline'

    def shortHelpString(self):
        return 'This script analyzes fractures (polylines) along 1D scanlines.'

    def initAlgorithm(self, config=None):
        self.addParameter(
            QgsProcessingParameterVectorLayer(
                self.INPUT_LAYER,
                'Input Polyline Layer',
                [QgsProcessing.TypeVectorLine]
            )
        )

    def processAlgorithm(self, parameters, context, feedback):
        input_layer = self.parameterAsVectorLayer(parameters, self.INPUT_LAYER, context)

        if not input_layer:
            raise QgsProcessingException('No input layer selected.')

        FracLine.validate_layer(input_layer)

        feedback.pushInfo('All validation checks passed.')

        return {}
