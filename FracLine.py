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
import geopandas
import qgis.processing
from shapely.wkt import loads
from qgis.PyQt.QtCore import Qt
from qgis.PyQt.QtWidgets import QAction, QWidget, QVBoxLayout, QTextBrowser, QLabel, QPushButton, QMessageBox
from qgis.PyQt.QtGui import QIcon
from qgis.core import (
    QgsMapLayerProxyModel, 
    QgsProject, 
    QgsProcessingException, 
    QgsWkbTypes,
    QgsLineSymbol,
    QgsSymbol,
    QgsSingleSymbolRenderer # Import QgsSingleSymbolRenderer for type checking
)
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
        
        # Initialize layer variables
        self.scanlines_clip = None
        self.scanlines_layer = None
        self.boundary_layer = None
        self.reference_line_layer = None
        self.fractures_layer = None

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
        self.run_button = QPushButton("Run Analysis")

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
        layout.addWidget(self.run_button)
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
        self.run_button.clicked.connect(self.run_analysis)

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
            self.fractures_layer = self.fractures_combo.currentLayer()
            if self.fractures_layer:
                self.log_browser.append('Validating Fractures layer...')
                check_layer(self.fractures_layer, 'Fractures')
                self.log_browser.append('Fractures layer validated successfully.')

            self.scanlines_layer = self.scanlines_combo.currentLayer()
            if self.scanlines_layer:
                self.log_browser.append('Validating Scanlines layer...')
                check_layer(self.scanlines_layer, 'Scanlines', check_unique_id=True)
                self.log_browser.append('Scanlines layer validated successfully.')

            self.reference_line_layer = self.reference_line_combo.currentLayer()
            if self.reference_line_layer:
                self.log_browser.append('Validating Reference line layer...')
                check_layer(self.reference_line_layer, 'Reference line', is_reference_line=True)
                self.log_browser.append('Reference line layer validated successfully.')

            self.boundary_layer = self.interpretation_boundary_combo.currentLayer()
            if self.boundary_layer:
                self.log_browser.append('Validating Interpretation boundary layer...')
                check_layer(self.boundary_layer, 'Interpretation boundary', is_polygon=True)
                self.log_browser.append('Interpretation boundary layer validated successfully.')

        except Exception as e:
            self.log_browser.append(f'ERROR: {e}')

    def run_analysis(self):
        self.log_browser.clear()
        self.log_browser.append("Running analysis...")

        if not self.scanlines_layer or not self.boundary_layer:
            self.log_browser.append("ERROR: Both scanlines and interpretation boundary layers must be selected.")
            return

        # Check for existing 'scanlines_clip' layer
        existing_clip_layer = QgsProject.instance().mapLayersByName('scanlines_clip')
        if existing_clip_layer:
            existing_clip_layer = existing_clip_layer[0]
            # Check if it's a file-based layer
            if existing_clip_layer.source().startswith('/') or existing_clip_layer.source().startswith('file://'):
                self.log_browser.append("ERROR: A file-based layer named 'scanlines_clip' already exists. Aborting to prevent overwrite.")
                return
            else:
                # It's a memory layer, ask to overwrite
                reply = QMessageBox.question(self, 'Overwrite Layer?',
                                             "A temporary layer named 'scanlines_clip' already exists. Do you want to overwrite it?",
                                             QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
                if reply == QMessageBox.No:
                    self.log_browser.append("Analysis aborted by user.")
                    return
                else:
                    QgsProject.instance().removeMapLayer(existing_clip_layer.id())
                    self.log_browser.append("Existing 'scanlines_clip' layer removed.")


        self.log_browser.append("Clipping scanlines to boundary...")
        try:
            result = qgis.processing.run("native:clip", {
                'INPUT': self.scanlines_layer,
                'OVERLAY': self.boundary_layer,
                'OUTPUT': 'memory:'
            })
            
            self.scanlines_clip = result['OUTPUT']
            self.scanlines_clip.setName('scanlines_clip')
            QgsProject.instance().addMapLayer(self.scanlines_clip)
            self.log_browser.append("Temporary layer 'scanlines_clip' created and added to canvas.")

            # Apply style to scanlines_clip layer
            renderer = self.scanlines_layer.renderer()
            if renderer:
                symbol = renderer.symbol()
                if symbol:
                    new_symbol = symbol.clone()
                    new_symbol.setWidth(symbol.width() * 2)
                    
                    new_renderer = QgsSingleSymbolRenderer(new_symbol)
                    self.scanlines_clip.setRenderer(new_renderer)
                    self.scanlines_clip.triggerRepaint()
                    self.log_browser.append("Style applied to 'scanlines_clip' with doubled line thickness.")
                else:
                    self.log_browser.append("Warning: Could not get symbol from renderer. Cannot apply style.")
            else:
                self.log_browser.append("Warning: Could not get renderer from scanlines layer. Cannot apply style.")

            # Convert to GeoDataFrame
            features = list(self.scanlines_clip.getFeatures())
            if not features:
                self.log_browser.append("Warning: Clip operation resulted in an empty layer.")
                self.scanlines_clip = geopandas.GeoDataFrame([], columns=['ID', 'geometry'])
                return

            ids = [f['ID'] for f in features]
            geoms_wkt = [f.geometry().asWkt() for f in features]
            
            shapely_geoms = [loads(wkt) for wkt in geoms_wkt]

            self.scanlines_clip = geopandas.GeoDataFrame(
                {'ID': ids},
                geometry=shapely_geoms,
                crs=self.scanlines_clip.crs().toWkt()
            )
            
            self.log_browser.append("Analysis complete. 'scanlines_clip' GeoDataFrame created.")
            self.log_browser.append("GeoDataFrame head:\n" + str(self.scanlines_clip.head()))

        except Exception as e:
            self.log_browser.append(f"ERROR during analysis: {e}")
