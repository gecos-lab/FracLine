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
import numpy as np
import qgis.processing
from shapely.wkt import loads
from qgis.PyQt.QtCore import Qt, QVariant
from qgis.PyQt.QtWidgets import QAction, QWidget, QVBoxLayout, QTextBrowser, QLabel, QPushButton, QMessageBox, QHBoxLayout, QSpinBox, QComboBox
from qgis.PyQt.QtGui import QIcon
from qgis.core import (
    QgsMapLayerProxyModel, 
    QgsProject, 
    QgsProcessingException, 
    QgsWkbTypes,
    QgsLineSymbol,
    QgsSymbol,
    QgsSingleSymbolRenderer,
    QgsMarkerSymbol,
    QgsUnitTypes,
    QgsField,
    QgsVectorLayer, # Added
    QgsFeature,     # Added
    QgsGeometry,    # Added
    QgsFeatureRequest, # Added
    QgsFieldProxyModel,
    QgsLayerTreeGroup # Added
)
from qgis.gui import QgsMapLayerComboBox, QgsDockWidget, QgsFieldComboBox
from collections import defaultdict, Counter

import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

def check_layer(layer, name, is_reference_line=False, check_unique_id=False, is_polygon=False, skip_id_check=False):
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
    if not skip_id_check:
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
        
        if check_unique_id and id_field_name:
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

class FracLinePlotWidget(QgsDockWidget):
    def __init__(self, iface):
        super().__init__('FracLine Plots')
        self.iface = iface
        self.setAllowedAreas(Qt.BottomDockWidgetArea)

        # Main widget and layout
        main_widget = QWidget()
        main_layout = QHBoxLayout()
        main_widget.setLayout(main_layout)
        self.setWidget(main_widget)

        # Create two figures and canvases
        self.figure1, self.ax1 = plt.subplots()
        self.canvas1 = FigureCanvas(self.figure1)
        
        self.figure2, self.ax2 = plt.subplots()
        self.canvas2 = FigureCanvas(self.figure2)

        # Add canvases to the layout
        main_layout.addWidget(self.canvas1)
        main_layout.addWidget(self.canvas2)

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
        self.intersections_layer = None
        self.scanlines_clip_split = None

        # Create widgets
        self.fractures_combo = QgsMapLayerComboBox(self)
        self.fractures_combo.setAllowEmptyLayer(True)
        self.fractures_combo.setCurrentIndex(0)

        self.scanlines_combo = QgsMapLayerComboBox(self)
        self.scanlines_combo.setAllowEmptyLayer(True)
        self.scanlines_combo.setCurrentIndex(0)

        self.scanline_id_field_combo = QgsFieldComboBox(self)
        self.scanline_id_field_combo.setFilters(QgsFieldProxyModel.String)

        self.reference_line_combo = QgsMapLayerComboBox(self)
        self.reference_line_combo.setAllowEmptyLayer(True)
        self.reference_line_combo.setCurrentIndex(0)

        self.interpretation_boundary_combo = QgsMapLayerComboBox(self)
        self.interpretation_boundary_combo.setAllowEmptyLayer(True)
        self.interpretation_boundary_combo.setCurrentIndex(0)

        self.log_browser = QTextBrowser(self)
        self.measure_button = QPushButton("Measure spacing and distance")
        
        self.barcode_height_spinbox = QSpinBox()
        self.barcode_height_spinbox.setRange(1, 100)
        self.barcode_height_spinbox.setValue(10)

        self.barcode_color_combo = QComboBox()
        self.barcode_color_combo.addItems(['Black', 'Red', 'Green', 'Blue', 'Cyan', 'Magenta', 'Yellow', 'Orange', 'Purple', 'Grey'])

        self.run_scanline_analysis_button = QPushButton("Run analysis on scanlines")
        self.run_scanline_analysis_button.setEnabled(False)

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
        layout.addWidget(QLabel('Scanline ID Field:'))
        layout.addWidget(self.scanline_id_field_combo)
        layout.addWidget(QLabel('Reference line:'))
        layout.addWidget(self.reference_line_combo)
        layout.addWidget(QLabel('Interpretation boundary:'))
        layout.addWidget(self.interpretation_boundary_combo)
        layout.addWidget(self.measure_button)
        
        barcode_layout = QHBoxLayout()
        barcode_layout.addWidget(QLabel("Barcode Height:"))
        barcode_layout.addWidget(self.barcode_height_spinbox)
        barcode_layout.addWidget(QLabel("Barcode Color:"))
        barcode_layout.addWidget(self.barcode_color_combo)
        layout.addLayout(barcode_layout)

        layout.addWidget(self.run_scanline_analysis_button)
        layout.addWidget(QLabel('Log:'))
        layout.addWidget(self.log_browser)

        widget = QWidget()
        widget.setLayout(layout)
        self.setWidget(widget)

        # Connect signals
        self.fractures_combo.layerChanged.connect(self.validate_layers)
        self.scanlines_combo.layerChanged.connect(self.validate_layers)
        self.scanlines_combo.layerChanged.connect(self.update_scanline_id_field_combo)
        self.reference_line_combo.layerChanged.connect(self.validate_layers)
        self.interpretation_boundary_combo.layerChanged.connect(self.validate_layers)
        self.measure_button.clicked.connect(self.run_analysis)

        self.find_and_set_layers()
        self.validate_layers()

    def update_scanline_id_field_combo(self, layer):
        self.scanline_id_field_combo.setLayer(layer)
        if layer:
            # Try to pre-select 'scanline_id' or 'ID' or 'id'
            field_name_to_find = 'scanline_id'
            idx = layer.fields().indexOf(field_name_to_find)
            if idx == -1:
                field_name_to_find = 'ID'
                idx = layer.fields().indexOf(field_name_to_find)
            if idx == -1:
                field_name_to_find = 'id'
                idx = layer.fields().indexOf(field_name_to_find)
            
            if idx != -1:
                self.scanline_id_field_combo.setCurrentIndex(idx)

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
                combo = layer_map[layer.name()]
                combo.setLayer(layer)
                if layer.name() == 'scanlines':
                    self.update_scanline_id_field_combo(layer)

    def validate_layers(self):
        self.log_browser.clear()
        try:
            project_crs = QgsProject.instance().crs()
            if not project_crs.isValid():
                raise QgsProcessingException("Project CRS is not set or is invalid.")

            self.log_browser.append(f"Project CRS: {project_crs.description()}")

            layers_to_check = [
                (self.fractures_combo.currentLayer(), 'Fractures'),
                (self.scanlines_combo.currentLayer(), 'Scanlines'),
                (self.reference_line_combo.currentLayer(), 'Reference line'),
                (self.interpretation_boundary_combo.currentLayer(), 'Interpretation boundary')
            ]

            for layer, name in layers_to_check:
                if layer:
                    layer_crs = layer.crs()
                    if not layer_crs.isValid():
                        raise QgsProcessingException(f"Layer '{name}' has an invalid CRS.")
                    if layer_crs != project_crs:
                        raise QgsProcessingException(
                            f"CRS mismatch: Layer '{name}' ({layer_crs.description()}) "
                            f"does not match Project CRS ({project_crs.description()})."
                        )
                    self.log_browser.append(f"Layer '{name}' CRS validated successfully.")

            self.fractures_layer = self.fractures_combo.currentLayer()
            if self.fractures_layer:
                self.log_browser.append('Validating Fractures layer...')
                check_layer(self.fractures_layer, 'Fractures')
                self.log_browser.append('Fractures layer validated successfully.')

            self.scanlines_layer = self.scanlines_combo.currentLayer()
            if self.scanlines_layer:
                self.log_browser.append('Validating Scanlines layer...')
                check_layer(self.scanlines_layer, 'Scanlines', skip_id_check=True)
                self.log_browser.append('Scanlines layer validated successfully.')

            self.reference_line_layer = self.reference_line_combo.currentLayer()
            if self.reference_line_layer:
                self.log_browser.append('Validating Reference line layer...')
                check_layer(self.reference_line_layer, 'Reference line', is_reference_line=True)
                self.log_browser.append('Reference line layer validated successfully.')

            self.boundary_layer = self.interpretation_boundary_combo.currentLayer()
            if self.boundary_layer:
                self.log_browser.append('Validating Interpretation boundary layer...')
                check_layer(self.boundary_layer, 'Interpretation boundary', is_polygon=True, skip_id_check=True)
                self.log_browser.append('Interpretation boundary layer validated successfully.')

        except Exception as e:
            self.log_browser.append(f'ERROR: {e}')

    def _get_or_create_output_group(self):
        """Finds or creates a static group for output layers."""
        group_name = "FracLine Results"
        root = QgsProject.instance().layerTreeRoot()
        output_group = root.findGroup(group_name)
        if not output_group:
            output_group = root.addGroup(group_name)
            self.log_browser.append(f"Created output layer group: '{group_name}'")
        else:
            self.log_browser.append(f"Using existing output layer group: '{group_name}'")
        return output_group

    def _check_and_remove_existing_temp_layer(self, layer_name):
        """Checks for and offers to remove an existing temporary layer."""
        existing_layer = QgsProject.instance().mapLayersByName(layer_name)
        if existing_layer:
            existing_layer = existing_layer[0]
            if existing_layer.source().startswith('/') or existing_layer.source().startswith('file://'):
                self.log_browser.append(f"ERROR: A file-based layer named '{layer_name}' already exists. Aborting to prevent overwrite.")
                return False
            else:
                reply = QMessageBox.question(self, 'Overwrite Layer?',
                                             f"A temporary layer named '{layer_name}' already exists. Do you want to overwrite it?",
                                             QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
                if reply == QMessageBox.No:
                    self.log_browser.append(f"Analysis aborted by user for layer '{layer_name}'.")
                    return False
                else:
                    QgsProject.instance().removeMapLayer(existing_layer.id())
                    self.log_browser.append(f"Existing '{layer_name}' layer removed.")
        return True

    def _apply_layer_style(self, target_layer, source_layer, is_point_layer=False):
        """
        Applies a consistent style to a target layer based on the source layer's renderer.
        If is_point_layer is True, it creates a point symbol based on the source line style.
        """
        renderer = source_layer.renderer()
        if not renderer:
            self.log_browser.append(f"Warning: Could not get renderer from source layer '{source_layer.name()}'. Cannot apply style to '{target_layer.name()}'.")
            return

        symbol = renderer.symbol()
        if not symbol:
            self.log_browser.append(f"Warning: Could not get symbol from renderer of source layer '{source_layer.name()}'. Cannot apply style to '{target_layer.name()}'.")
            return

        if is_point_layer:
            symbol_layer = symbol.symbolLayer(0)
            if not symbol_layer:
                self.log_browser.append(f"Warning: Could not get symbol layer from source layer '{source_layer.name()}'. Cannot apply point style to '{target_layer.name()}'.")
                return

            unit_string = QgsUnitTypes.encodeUnit(symbol_layer.widthUnit())
            point_symbol = QgsMarkerSymbol.createSimple({
                'name': 'circle',
                'color': 'white',
                'outline_color': symbol_layer.color().name(),
                'outline_width': str(symbol_layer.width()),
                'outline_width_unit': unit_string,
                'size': str(symbol_layer.width() * 6),
                'size_unit': unit_string
            })
            point_renderer = QgsSingleSymbolRenderer(point_symbol)
            target_layer.setRenderer(point_renderer)
            self.log_browser.append(f"Style applied to '{target_layer.name()}' (point layer).")
        else: # Line layer
            new_symbol = symbol.clone()
            new_symbol.setWidth(symbol.width() * 2) # Doubled thickness for emphasis
            new_renderer = QgsSingleSymbolRenderer(new_symbol)
            target_layer.setRenderer(new_renderer)
            self.log_browser.append(f"Style applied to '{target_layer.name()}' (line layer) with doubled thickness.")
        
        target_layer.triggerRepaint()

    def _prepare_scanlines_clip(self, project_crs, scanline_id_field_name, ref_line_geom, output_group):
        """
        Handles clipping, single parts conversion, field renaming,
        scanline_part_id assignment, and initial styling for scanlines_clip.
        """
        if not self._check_and_remove_existing_temp_layer('scanlines_clip'):
            return None

        self.log_browser.append("Clipping scanlines to boundary...")
        clip_result = qgis.processing.run("native:clip", {
            'INPUT': self.scanlines_layer,
            'OVERLAY': self.boundary_layer,
            'OUTPUT': 'memory:',
            'CRS': project_crs
        })
        clipped_layer = clip_result['OUTPUT']

        self.log_browser.append("Converting clipped scanlines to single parts...")
        singleparts_result = qgis.processing.run("native:multiparttosingleparts", {
            'INPUT': clipped_layer,
            'OUTPUT': 'memory:',
            'CRS': project_crs
        })
        single_parts_layer = singleparts_result['OUTPUT']

        self.log_browser.append("Processing clipped scanlines and calculating distances...")
        self.scanlines_clip = QgsVectorLayer(f"LineString?crs={project_crs.toWkt()}", "scanlines_clip", "memory")
        provider = self.scanlines_clip.dataProvider()
        provider.addAttributes([
            QgsField("scanline_id", QVariant.String),
            QgsField("scanline_part_id", QVariant.String),
            QgsField("distance", QVariant.Double),
            QgsField("normalized_length", QVariant.Double)
        ])
        self.scanlines_clip.updateFields()

        features_by_scanline_id = defaultdict(list)
        for feature in single_parts_layer.getFeatures():
            scanline_id = feature[scanline_id_field_name]
            features_by_scanline_id[scanline_id].append(feature)

        new_features = []
        for scanline_id, features in features_by_scanline_id.items():
            features_with_dist = []
            for feature in features:
                first_node_geom = QgsGeometry.fromPoint(feature.geometry().vertexAt(0))
                sorting_dist = first_node_geom.distance(ref_line_geom)
                features_with_dist.append((sorting_dist, feature))
            
            features_with_dist.sort(key=lambda x: x[0])
            
            for i, (dist, feature) in enumerate(features_with_dist):
                part_number = i + 1
                new_part_id = f"{scanline_id}-{part_number}"
                
                geom = feature.geometry()
                midpoint_geom = geom.interpolate(geom.length() / 2.0)
                distance_to_ref = midpoint_geom.distance(ref_line_geom)

                vertices = list(geom.vertices())
                first_point_geom = QgsGeometry.fromPoint(vertices[0])
                last_point_geom = QgsGeometry.fromPoint(vertices[-1])

                dist_first = first_point_geom.distance(ref_line_geom)
                dist_last = last_point_geom.distance(ref_line_geom)
                normalized_length = abs(dist_last - dist_first)
                
                new_feat = QgsFeature(self.scanlines_clip.fields())
                new_feat.setGeometry(geom)
                new_feat.setAttributes([scanline_id, new_part_id, distance_to_ref, normalized_length])
                new_features.append(new_feat)

        provider.addFeatures(new_features)
        self.log_browser.append("'scanlines_clip' layer created with simplified fields.")
        
        QgsProject.instance().addMapLayer(self.scanlines_clip, False)
        output_group.addLayer(self.scanlines_clip)
        self.log_browser.append("Temporary layer 'scanlines_clip' created and added to canvas.")

        self._apply_layer_style(self.scanlines_clip, self.scanlines_layer, is_point_layer=False)

        if self.scanlines_clip.featureCount() == 0:
            self.log_browser.append("Warning: Clip operation resulted in an empty layer.")
            return None
        
        return self.scanlines_clip

    def _process_intersections(self, project_crs, ref_line_geom, output_group):
        """
        Handles intersection, creating the intersections layer, calculating distances,
        filtering unique scanline_part_ids, and styling.
        Returns a set of scanline_part_ids to remove from split scanlines.
        """
        if not self._check_and_remove_existing_temp_layer('intersections'):
            return None, set()

        self.log_browser.append("Intersecting scanlines_clip with fractures...")
        intersection_result = qgis.processing.run("native:lineintersections", {
            'INPUT': self.scanlines_clip,
            'INTERSECT': self.fractures_layer,
            'INPUT_FIELDS': ['scanline_id', 'scanline_part_id'],
            'INTERSECT_FIELDS': [],
            'OUTPUT': 'memory:',
            'CRS': project_crs
        })
        intersections_temp_layer = intersection_result['OUTPUT']

        self.intersections_layer = QgsVectorLayer(f"Point?crs={project_crs.toWkt()}", "intersections", "memory")
        provider_int = self.intersections_layer.dataProvider()
        provider_int.addAttributes([
            QgsField("scanline_id", QVariant.String),
            QgsField("scanline_part_id", QVariant.String),
            QgsField("distance", QVariant.Double)
        ])
        self.intersections_layer.updateFields()

        intersection_features = []
        for feature in intersections_temp_layer.getFeatures():
            scanline_id = feature['scanline_id']
            scanline_part_id = feature['scanline_part_id']
            
            point_geom = feature.geometry()
            distance_to_ref = point_geom.distance(ref_line_geom)
            
            new_feat = QgsFeature(self.intersections_layer.fields())
            new_feat.setGeometry(point_geom)
            new_feat.setAttributes([scanline_id, scanline_part_id, distance_to_ref])
            intersection_features.append(new_feat)
        
        provider_int.addFeatures(intersection_features)
        self.log_browser.append("'intersections' layer created with simplified fields.")

        self.log_browser.append("Filtering intersections: removing points from scanline parts with only one intersection...")
        all_part_ids = [f['scanline_part_id'] for f in self.intersections_layer.getFeatures()]
        id_counts = Counter(all_part_ids)
        
        ids_to_remove = {part_id for part_id, count in id_counts.items() if count == 1}
        
        if ids_to_remove:
            fids_to_delete = []
            for feature in self.intersections_layer.getFeatures():
                if feature['scanline_part_id'] in ids_to_remove:
                    fids_to_delete.append(feature.id())
            
            self.intersections_layer.startEditing()
            self.intersections_layer.deleteFeatures(fids_to_delete)
            if not self.intersections_layer.commitChanges():
                self.log_browser.append("ERROR: Could not commit deletion of unique intersection points.")
            else:
                self.log_browser.append(f"Removed {len(fids_to_delete)} points from {len(ids_to_remove)} scanline parts.")
        else:
            self.log_browser.append("No unique intersection points found to remove.")

        QgsProject.instance().addMapLayer(self.intersections_layer, False)
        output_group.addLayer(self.intersections_layer)
        self.log_browser.append("Temporary layer 'intersections' created and added to canvas.")

        self._apply_layer_style(self.intersections_layer, self.scanlines_layer, is_point_layer=True)

        return self.intersections_layer, ids_to_remove

    def _process_scanlines_clip_split(self, project_crs, ref_line_geom, output_group, ids_to_remove):
        """
        Handles splitting scanlines_clip by fractures, filtering segments based on
        ids_to_remove from intersections, calculating distances, and styling.
        """
        if not self._check_and_remove_existing_temp_layer('scanlines_clip_split'):
            return None

        self.log_browser.append("Splitting scanlines and filtering segments...")

        split_result = qgis.processing.run("native:splitwithlines", {
            'INPUT': self.scanlines_clip,
            'LINES': self.fractures_layer,
            'OUTPUT': 'memory:',
            'CRS': project_crs
        })
        split_layer = split_result['OUTPUT']

        self.scanlines_clip_split = QgsVectorLayer(f"LineString?crs={project_crs.toWkt()}", "scanlines_clip_split", "memory")
        provider_split = self.scanlines_clip_split.dataProvider()
        provider_split.addAttributes([
            QgsField("scanline_id", QVariant.String),
            QgsField("scanline_part_id", QVariant.String),
            QgsField("distance", QVariant.Double),
            QgsField("spacing", QVariant.Double),
            QgsField("distance_order", QVariant.Int),
            QgsField("spacing_order", QVariant.Int)
        ])
        self.scanlines_clip_split.updateFields()
        
        output_features = []
        segments_by_scanline_part = defaultdict(list)

        for feature in split_layer.getFeatures():
            scanline_part_id = feature['scanline_part_id']
            segments_by_scanline_part[scanline_part_id].append(feature)

        original_geoms = {f['scanline_part_id']: f.geometry() for f in self.scanlines_clip.getFeatures()}

        self.log_browser.append("Filtering split scanlines based on intersection results...")
        for scanline_part_id, segments in segments_by_scanline_part.items():
            if scanline_part_id in ids_to_remove:
                continue

            original_geom = original_geoms.get(scanline_part_id)
            if not original_geom:
                self.log_browser.append(f"Warning: Original scanline part with ID '{scanline_part_id}' not found for ordering.")
                continue

            sorted_segments_with_dist = []
            for segment_feature in segments:
                first_point_of_segment = segment_feature.geometry().asPolyline()[0]
                distance_along = original_geom.lineLocatePoint(QgsGeometry.fromPointXY(first_point_of_segment))
                sorted_segments_with_dist.append((distance_along, segment_feature))

            sorted_segments_with_dist.sort(key=lambda x: x[0])

            # Keep only the middle segments (between the first and last intersection)
            segments_to_keep = [item[1] for item in sorted_segments_with_dist[1:-1]]

            for segment_feature in segments_to_keep:
                scanline_id = segment_feature['scanline_id']
                
                geom = segment_feature.geometry()
                midpoint_geom = geom.interpolate(geom.length() / 2.0)
                distance_to_ref = midpoint_geom.distance(ref_line_geom)

                vertices = list(geom.vertices())
                first_point_geom = QgsGeometry.fromPoint(vertices[0])
                last_point_geom = QgsGeometry.fromPoint(vertices[-1])

                dist_first = first_point_geom.distance(ref_line_geom)
                dist_last = last_point_geom.distance(ref_line_geom)
                spacing = abs(dist_last - dist_first)
                
                new_feature = QgsFeature(self.scanlines_clip_split.fields())
                new_feature.setGeometry(geom)
                new_feature.setAttributes([scanline_id, scanline_part_id, distance_to_ref, spacing, None, None])
                output_features.append(new_feature)

        # Group features by scanline_id and rank them
        features_by_scanline = defaultdict(list)
        for feature in output_features:
            features_by_scanline[feature['scanline_id']].append(feature)

        final_features_with_ranks = []
        for scanline_id, features_in_group in features_by_scanline.items():
            # Rank by distance
            features_in_group.sort(key=lambda f: f['distance'])
            for i, feature in enumerate(features_in_group):
                feature['distance_order'] = i + 1

            # Rank by spacing
            features_in_group.sort(key=lambda f: f['spacing'])
            for i, feature in enumerate(features_in_group):
                feature['spacing_order'] = i + 1
            
            final_features_with_ranks.extend(features_in_group)

        provider_split.addFeatures(final_features_with_ranks)
        QgsProject.instance().addMapLayer(self.scanlines_clip_split, False)
        output_group.addLayer(self.scanlines_clip_split)
        self.log_browser.append("Temporary layer 'scanlines_clip_split' created with filtered segments and simplified fields.")

        self._apply_layer_style(self.scanlines_clip_split, self.scanlines_layer, is_point_layer=False)

        return self.scanlines_clip_split

    def run_analysis(self):
        self.log_browser.append("==================================================")
        self.log_browser.append("Running analysis...")
        
        try:
            project_crs = QgsProject.instance().crs()
            if not project_crs.isValid():
                raise QgsProcessingException("Project CRS is not set or is invalid.")

            scanline_id_field_name = self.scanline_id_field_combo.currentField()
            if not scanline_id_field_name:
                self.log_browser.append("ERROR: Scanline ID Field must be selected.")
                return

            if not self.fractures_layer or not self.scanlines_layer or \
               not self.reference_line_layer or not self.boundary_layer:
                self.log_browser.append("ERROR: All input layers must be selected.")
                return

            self.log_browser.append(f"Validating uniqueness of '{scanline_id_field_name}' in '{self.scanlines_layer.name()}'...")
            ids = set()
            for feature in self.scanlines_layer.getFeatures():
                try:
                    feature_id = feature[scanline_id_field_name]
                    if feature_id in ids:
                        self.log_browser.append(f"ERROR: Duplicate ID '{feature_id}' found in layer '{self.scanlines_layer.name()}'. IDs must be unique.")
                        return
                    ids.add(feature_id)
                except KeyError:
                    self.log_browser.append(f"ERROR: Field '{scanline_id_field_name}' not found in '{self.scanlines_layer.name()}'.")
                    return
            self.log_browser.append("Scanline ID field validated for uniqueness.")

            output_group = self._get_or_create_output_group()
            
            ref_line_feature = next(self.reference_line_layer.getFeatures())
            ref_line_geom = ref_line_feature.geometry()

            self.scanlines_clip = self._prepare_scanlines_clip(
                project_crs, scanline_id_field_name, ref_line_geom, output_group
            )
            if self.scanlines_clip is None: # Indicates an error or user cancellation
                return

            if self.fractures_layer:
                self.intersections_layer, ids_to_remove = self._process_intersections(
                    project_crs, ref_line_geom, output_group
                )
                if self.intersections_layer is None: # Indicates an error or user cancellation
                    return

                self.scanlines_clip_split = self._process_scanlines_clip_split(
                    project_crs, ref_line_geom, output_group, ids_to_remove
                )
                if self.scanlines_clip_split is None: # Indicates an error or user cancellation
                    return
                
                if self.intersections_layer and self.scanlines_clip_split:
                    self.run_scanline_analysis_button.setEnabled(True)
            else:
                self.log_browser.append("No fractures layer selected. Skipping intersection and splitting.")

        except Exception as e:
            self.log_browser.append(f"ERROR during analysis: {e}")
