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

import os
from qgis.PyQt.QtCore import Qt
from qgis.PyQt.QtWidgets import QAction, QWidget, QVBoxLayout, QTextBrowser, QLabel
from qgis.PyQt.QtGui import QIcon
from qgis.core import QgsMapLayerProxyModel, QgsProject, QgsProcessingException, QgsWkbTypes
from qgis.gui import QgsMapLayerComboBox, QgsDockWidget

def check_layer(layer, name, is_reference_line=False, check_unique_id=False, is_polygon=False):
    """
    Validates the input layer.
    For lines: checks that they are not "disconnected" multipart geometries.
    For polygons: no geometry validation is performed.
    Also checks for ID field and other constraints.
    """
    if not layer:
        raise QgsProcessingException(f'Layer not found: {name}')

    # Check for a text field called ID or id
    id_field_name = None
    if 'ID' in layer.fields().names():
        id_field_name = 'ID'
    elif 'id' in layer.fields().names():
        id_field_name = 'id'
    else:
        raise QgsProcessingException(
            f'Layer {name} must have a text field called ID or id.'
        )

    ids = set()
    for feature in layer.getFeatures():
        geom = feature.geometry()

        if not is_polygon: # It's a line layer
            # For lines, check for "disconnected" parts. A connected MultiLineString
            # is okay, but a MultiLineString with separate, non-touching parts is not.
            # The buffer(0) trick merges touching parts. If the result is still
            # multipart, the parts were disconnected.
            merged_geom = geom.buffer(0, 5)
            if merged_geom.isMultipart():
                raise QgsProcessingException(
                    f'Feature {feature.id()} in {name} has disconnected parts. Please use single, connected polylines.'
                )

        if is_reference_line:
            # This check is only for the reference line.
            # We use vertices() as it works for both LineString and MultiLineString.
            vertices = list(geom.vertices())
            if len(vertices) != 2:
                raise QgsProcessingException(
                    f'Feature {feature.id()} in {name} is not a segment with exactly two nodes. It has {len(vertices)} nodes.'
                )
        
        if check_unique_id:
            feature_id = feature[id_field_name]
            if feature_id in ids:
                raise QgsProcessingException(
                    f'Duplicate ID "{feature_id}" found in layer {name}. IDs must be unique.'
                )
            ids.add(feature_id)

    # Check for reference line with a single feature
    if is_reference_line:
        if layer.featureCount() != 1:
            raise QgsProcessingException(
                f'Layer {name} must have a single feature.'
            )

class FracLinePlugin:
    def __init__(self, iface):
        self.iface = iface
        self.dockwidget = None
        self.action = None

    def initGui(self):
        icon_path = os.path.join(os.path.dirname(__file__), 'icon.png')
        self.action = QAction(QIcon(icon_path), 'FracLine', self.iface.mainWindow())
        self.action.triggered.connect(self.run)
        self.iface.addToolBarIcon(self.action)
        self.iface.addPluginToMenu('&FracLine', self.action)

    def unload(self):
        self.iface.removePluginMenu('&FracLine', self.action)
        self.iface.removeToolBarIcon(self.action)
        if self.dockwidget:
            self.iface.removeDockWidget(self.dockwidget)
            self.dockwidget.deleteLater()
            self.dockwidget = None

    def run(self):
        if not self.dockwidget:
            self.dockwidget = FracLineDockWidget(self.iface)
            self.iface.addDockWidget(Qt.RightDockWidgetArea, self.dockwidget)
        self.dockwidget.show()

class FracLineDockWidget(QgsDockWidget):
    def __init__(self, iface):
        super().__init__('FracLine')
        self.iface = iface
        self.setAllowedAreas(Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea)

        # Create widgets
        self.fractures_combo = QgsMapLayerComboBox(self)
        self.fractures_combo.setAllowEmptyLayer(True)
        self.fractures_combo.setCurrentIndex(0)

        self.scanlines_combo = QgsMapLayerComboBox(self)
        self.scanlines_combo.setAllowEmptyLayer(True)
        self.scanlines_combo.setCurrentIndex(0)

        self.reference_line_combo = QgsMapLayerComboBox(self)
        self.reference_line_combo.setAllowEmptyLayer(True)
        self.reference_line_combo.setCurrentIndex(0)

        self.interpretation_boundary_combo = QgsMapLayerComboBox(self)
        self.interpretation_boundary_combo.setAllowEmptyLayer(True)
        self.interpretation_boundary_combo.setCurrentIndex(0)

        self.log_browser = QTextBrowser(self)

        # Set layer filters
        self.fractures_combo.setFilters(QgsMapLayerProxyModel.LineLayer)
        self.scanlines_combo.setFilters(QgsMapLayerProxyModel.LineLayer)
        self.reference_line_combo.setFilters(QgsMapLayerProxyModel.LineLayer)
        self.interpretation_boundary_combo.setFilters(QgsMapLayerProxyModel.PolygonLayer)

        # Layout
        layout = QVBoxLayout()
        layout.addWidget(QLabel('Fractures:'))
        layout.addWidget(self.fractures_combo)
        layout.addWidget(QLabel('Scanlines:'))
        layout.addWidget(self.scanlines_combo)
        layout.addWidget(QLabel('Reference line:'))
        layout.addWidget(self.reference_line_combo)
        layout.addWidget(QLabel('Interpretation boundary:'))
        layout.addWidget(self.interpretation_boundary_combo)
        layout.addWidget(QLabel('Log:'))
        layout.addWidget(self.log_browser)

        widget = QWidget()
        widget.setLayout(layout)
        self.setWidget(widget)

        # Connect signals
        self.fractures_combo.layerChanged.connect(self.validate_layers)
        self.scanlines_combo.layerChanged.connect(self.validate_layers)
        self.reference_line_combo.layerChanged.connect(self.validate_layers)
        self.interpretation_boundary_combo.layerChanged.connect(self.validate_layers)

        self.find_and_set_layers()
        self.validate_layers()

    def find_and_set_layers(self):
        """Finds layers with specific names and sets them in the combo boxes."""
        layer_map = {
            'fractures': self.fractures_combo,
            'scanlines': self.scanlines_combo,
            'reference_line': self.reference_line_combo,
            'interpretation_boundary': self.interpretation_boundary_combo
        }

        for layer in QgsProject.instance().mapLayers().values():
            if layer.name() in layer_map:
                layer_map[layer.name()].setLayer(layer)

    def validate_layers(self):
        self.log_browser.clear()
        try:
            fractures_layer = self.fractures_combo.currentLayer()
            if fractures_layer:
                self.log_browser.append('Validating Fractures layer...')
                check_layer(fractures_layer, 'Fractures')
                self.log_browser.append('Fractures layer validated successfully.')

            scanlines_layer = self.scanlines_combo.currentLayer()
            if scanlines_layer:
                self.log_browser.append('Validating Scanlines layer...')
                check_layer(scanlines_layer, 'Scanlines', check_unique_id=True)
                self.log_browser.append('Scanlines layer validated successfully.')

            reference_line_layer = self.reference_line_combo.currentLayer()
            if reference_line_layer:
                self.log_browser.append('Validating Reference line layer...')
                check_layer(reference_line_layer, 'Reference line', is_reference_line=True)
                self.log_browser.append('Reference line layer validated successfully.')

            interpretation_boundary_layer = self.interpretation_boundary_combo.currentLayer()
            if interpretation_boundary_layer:
                self.log_browser.append('Validating Interpretation boundary layer...')
                check_layer(interpretation_boundary_layer, 'Interpretation boundary', is_polygon=True)
                self.log_browser.append('Interpretation boundary layer validated successfully.')

        except Exception as e:
            self.log_browser.append(f'ERROR: {e}')
