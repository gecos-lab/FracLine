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
from qgis.PyQt.QtWidgets import QAction, QWidget, QVBoxLayout, QTextBrowser, QLabel, QPushButton, QMessageBox
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
    QgsFieldProxyModel
)
from qgis.gui import QgsMapLayerComboBox, QgsDockWidget, QgsFieldComboBox
from collections import defaultdict # Added

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

def calculate_perpendicular_distance(point, line_start, line_end):
    """
    Calculates the perpendicular distance from a point to a line segment defined by two points.
    Uses numpy for vector calculations.
    """
    p0 = np.array([point.x(), point.y()])
    p1 = np.array([line_start.x(), line_start.y()])
    p2 = np.array([line_end.x(), line_end.y()])

    # Vector from p1 to p2
    v = p2 - p1
    # Vector from p1 to p0
    w = p0 - p1

    # Squared length of the line segment
    l2 = np.dot(v, v)
    if l2 == 0.0: # p1 and p2 are the same point
        return np.linalg.norm(p0 - p1)

    # Parameter t of the closest point on the line (p1 + t*v) to p0
    # t = dot(w, v) / l2
    t = np.dot(w, v) / l2

    if t < 0.0:
        # Closest point is p1
        return np.linalg.norm(p0 - p1)
    elif t > 1.0:
        # Closest point is p2
        return np.linalg.norm(p0 - p2)
    else:
        # Closest point is on the segment
        projection = p1 + t * v
        return np.linalg.norm(p0 - projection)

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
        self.intersections_layer = None
        self.scanlines_clip_split = None # Added

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
        layout.addWidget(QLabel('Scanline ID Field:'))
        layout.addWidget(self.scanline_id_field_combo)
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
        self.scanlines_combo.layerChanged.connect(self.update_scanline_id_field_combo)
        self.reference_line_combo.layerChanged.connect(self.validate_layers)
        self.interpretation_boundary_combo.layerChanged.connect(self.validate_layers)
        self.run_button.clicked.connect(self.run_analysis)

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

    def run_analysis(self):
        self.log_browser.append("==================================================")
        self.log_browser.append("Running analysis...")
        
        # Get project CRS
        project_crs = QgsProject.instance().crs()

        # Get selected scanline ID field
        scanline_id_field_name = self.scanline_id_field_combo.currentField()
        if not scanline_id_field_name:
            self.log_browser.append("ERROR: Scanline ID Field must be selected.")
            return

        # Check layers availability
        if not self.fractures_layer:
            self.log_browser.append("ERROR: fractures layer must be selected.")
            return
        if not self.scanlines_layer:
            self.log_browser.append("ERROR: scanlines layer must be selected.")
            return
        if not self.reference_line_layer:
            self.log_browser.append("ERROR: reference line must be selected.")
            return
        if not self.boundary_layer:
            self.log_browser.append("ERROR: interpretation boundary layer must be selected.")
            return

        # Uniqueness check for the selected scanline ID field
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

        # Check for existing 'scanlines_clip' layer
        existing_clip_layer = QgsProject.instance().mapLayersByName('scanlines_clip')
        if existing_clip_layer:
            existing_clip_layer = existing_clip_layer[0]
            if existing_clip_layer.source().startswith('/') or existing_clip_layer.source().startswith('file://'):
                self.log_browser.append("ERROR: A file-based layer named 'scanlines_clip' already exists. Aborting to prevent overwrite.")
                return
            else:
                reply = QMessageBox.question(self, 'Overwrite Layer?',
                                             "A temporary layer named 'scanlines_clip' already exists. Do you want to overwrite it?",
                                             QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
                if reply == QMessageBox.No:
                    self.log_browser.append("Analysis aborted by user.")
                    return
                else:
                    QgsProject.instance().removeMapLayer(existing_clip_layer.id())
                    self.log_browser.append("Existing 'scanlines_clip' layer removed.")

        # Check for existing 'scanlines_clip_split' layer
        existing_split_layer = QgsProject.instance().mapLayersByName('scanlines_clip_split')
        if existing_split_layer:
            existing_split_layer = existing_split_layer[0]
            if existing_split_layer.source().startswith('/') or existing_split_layer.source().startswith('file://'):
                self.log_browser.append("ERROR: A file-based layer named 'scanlines_clip_split' already exists. Aborting to prevent overwrite.")
                return
            else:
                reply = QMessageBox.question(self, 'Overwrite Layer?',
                                             "A temporary layer named 'scanlines_clip_split' already exists. Do you want to overwrite it?",
                                             QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
                if reply == QMessageBox.No:
                    self.log_browser.append("Split scanlines analysis aborted by user.")
                    return
                else:
                    QgsProject.instance().removeMapLayer(existing_split_layer.id())
                    self.log_browser.append("Existing 'scanlines_clip_split' layer removed.")

        self.log_browser.append("Clipping scanlines to boundary...")
        try:
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

            self.scanlines_clip = singleparts_result['OUTPUT']
            self.scanlines_clip.setName('scanlines_clip')
            
            # --- Rename field and add scanline_part_id ---
            self.log_browser.append("Renaming ID field and adding 'scanline_part_id'...")
            
            provider = self.scanlines_clip.dataProvider()
            
            # Rename the original ID field to 'scanline_id'
            original_field_index = self.scanlines_clip.fields().indexOf(scanline_id_field_name)
            if original_field_index != -1:
                provider.renameAttributes({original_field_index: 'scanline_id'})
                self.scanlines_clip.updateFields()
                self.log_browser.append(f"Field '{scanline_id_field_name}' renamed to 'scanline_id'.")
            else:
                self.log_browser.append(f"Warning: Could not find field '{scanline_id_field_name}' to rename.")

            # Add the new 'scanline_part_id' field
            if self.scanlines_clip.fields().indexOf('scanline_part_id') == -1:
                provider.addAttributes([QgsField("scanline_part_id", QVariant.String)])
                self.scanlines_clip.updateFields()
                self.log_browser.append("Field 'scanline_part_id' added.")

            # Get reference line geometry for distance calculation
            ref_line_feature = next(self.reference_line_layer.getFeatures())
            vertices = list(ref_line_feature.geometry().vertices())
            line_start, line_end = vertices[0], vertices[1]

            # Group features by the new 'scanline_id'
            features_by_scanline_id = defaultdict(list)
            for feature in self.scanlines_clip.getFeatures():
                scanline_id = feature['scanline_id']
                features_by_scanline_id[scanline_id].append(feature)

            self.scanlines_clip.startEditing()
            scanline_part_id_field_index = self.scanlines_clip.fields().indexOf('scanline_part_id')

            for scanline_id, features in features_by_scanline_id.items():
                # Calculate distance for each feature and store it
                features_with_dist = []
                for feature in features:
                    first_node = feature.geometry().vertexAt(0)
                    distance = calculate_perpendicular_distance(first_node, line_start, line_end)
                    features_with_dist.append((distance, feature))
                
                # Sort features by distance
                features_with_dist.sort(key=lambda x: x[0])
                
                # Assign the new scanline_part_id
                for i, (dist, feature) in enumerate(features_with_dist):
                    part_number = i + 1
                    new_part_id = f"{scanline_id}-{part_number}"
                    self.scanlines_clip.changeAttributeValue(feature.id(), scanline_part_id_field_index, new_part_id)

            if not self.scanlines_clip.commitChanges():
                self.log_browser.append("ERROR: Could not commit changes for 'scanline_part_id'.")
            else:
                self.log_browser.append("'scanline_part_id' field populated successfully.")
            
            QgsProject.instance().addMapLayer(self.scanlines_clip)
            self.log_browser.append("Temporary layer 'scanlines_clip' (single parts) created and added to canvas.")

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

            # Check if the clip operation resulted in an empty layer
            if self.scanlines_clip.featureCount() == 0:
                self.log_browser.append("Warning: Clip operation resulted in an empty layer.")
                return

            # Intersect scanlines_clip with fractures
            if self.fractures_layer:
                self.log_browser.append("Intersecting scanlines_clip with fractures...")
                
                existing_intersections_layer = QgsProject.instance().mapLayersByName('intersections')
                if existing_intersections_layer:
                    existing_intersections_layer = existing_intersections_layer[0]
                    if existing_intersections_layer.source().startswith('/') or existing_intersections_layer.source().startswith('file://'):
                        self.log_browser.append("ERROR: A file-based layer named 'intersections' already exists. Aborting to prevent overwrite.")
                        return
                    else:
                        reply = QMessageBox.question(self, 'Overwrite Layer?',
                                                     "A temporary layer named 'intersections' already exists. Do you want to overwrite it?",
                                                     QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
                        if reply == QMessageBox.No:
                            self.log_browser.append("Intersection analysis aborted by user.")
                            return
                        else:
                            QgsProject.instance().removeMapLayer(existing_intersections_layer.id())
                            self.log_browser.append("Existing 'intersections' layer removed.")

                intersection_result = qgis.processing.run("native:lineintersections", {
                    'INPUT': self.scanlines_clip,
                    'INTERSECT': self.fractures_layer,
                    'INPUT_FIELDS': [],
                    'INTERSECT_FIELDS': [],
                    'OUTPUT': 'memory:',
                    'CRS': project_crs
                })
                self.intersections_layer = intersection_result['OUTPUT']

                self.intersections_layer.setName('intersections')
                QgsProject.instance().addMapLayer(self.intersections_layer)
                self.log_browser.append("Temporary layer 'intersections' created and added to canvas.")

                # --- Split scanlines_clip and filter segments ---
                self.log_browser.append("Splitting scanlines and filtering segments...")

                # Perform the split operation
                split_result = qgis.processing.run("native:splitwithlines", {
                    'INPUT': self.scanlines_clip,
                    'LINES': self.fractures_layer,
                    'OUTPUT': 'memory:',
                    'CRS': project_crs
                })
                split_layer = split_result['OUTPUT']
                split_layer.setName('temp_split_scanlines') # Give it a temporary name

                if split_layer.featureCount() == 0:
                    self.log_browser.append("Warning: Split operation resulted in an empty layer. No segments to process.")
                    # Create an empty output layer if no splits occurred
                    output_layer = QgsVectorLayer("LineString", "scanlines_clip_split", "memory")
                    output_layer.setCrs(project_crs)
                    # Copy fields from the original scanlines_clip layer
                    output_layer.dataProvider().addAttributes(self.scanlines_clip.fields().toList()) 
                    output_layer.updateFields()
                    QgsProject.instance().addMapLayer(output_layer)
                    self.scanlines_clip_split = output_layer
                    self.log_browser.append("Empty 'scanlines_clip_split' layer created.")
                    QgsProject.instance().removeMapLayer(split_layer.id()) # Clean up temp layer
                    return


                # Create output layer for filtered segments
                output_layer = QgsVectorLayer("LineString", "scanlines_clip_split", "memory")
                output_layer.setCrs(project_crs)
                # Copy fields from the split_layer (which should have fields from scanlines_clip)
                output_layer.dataProvider().addAttributes(split_layer.fields().toList())
                output_layer.updateFields()
                
                output_features = []
                segments_by_scanline_part = defaultdict(list)

                # Group segments by their original scanline part ID
                for feature in split_layer.getFeatures():
                    # Use 'scanline_part_id' for grouping
                    scanline_part_id = feature['scanline_part_id']
                    segments_by_scanline_part[scanline_part_id].append(feature)

                # Process each original scanline part's segments
                for scanline_part_id, segments in segments_by_scanline_part.items():
                    if len(segments) <= 2: # If 0, 1, or 2 segments, remove all (first and last)
                        continue 

                    # Get the original unsplit scanline geometry to determine order
                    request = QgsFeatureRequest().setFilterExpression(f"\"scanline_part_id\" = '{scanline_part_id}'")
                    original_scanline_feature = next(self.scanlines_clip.getFeatures(request), None)
                    
                    if not original_scanline_feature:
                        self.log_browser.append(f"Warning: Original scanline part with ID '{scanline_part_id}' not found in scanlines_clip for ordering.")
                        continue

                    original_geom = original_scanline_feature.geometry()

                    # Sort segments by their position along the original line
                    sorted_segments_with_dist = []
                    for segment_feature in segments:
                        segment_geom = segment_feature.geometry()
                        # Get the first point of the segment
                        first_point_of_segment = segment_geom.asPolyline()[0]
                        
                        # Calculate distance along the original line
                        distance = original_geom.lineLocatePoint(QgsGeometry.fromPointXY(first_point_of_segment))
                        
                        if distance >= 0: # lineLocatePoint returns -1 if point is not on line
                            sorted_segments_with_dist.append((distance, segment_feature))
                        else:
                            self.log_browser.append(f"Warning: Segment for scanline part ID '{scanline_part_id}' could not be located on original line for sorting.")

                    sorted_segments_with_dist.sort(key=lambda x: x[0])

                    # Filter segments: Exclude first and last
                    segments_to_keep = [item[1] for item in sorted_segments_with_dist[1:-1]] # Exclude first and last

                    for segment_feature in segments_to_keep:
                        new_feature = QgsFeature(output_layer.fields())
                        new_feature.setGeometry(segment_feature.geometry())
                        new_feature.setAttributes(segment_feature.attributes())
                        output_features.append(new_feature)
                
                output_layer.dataProvider().addFeatures(output_features)
                QgsProject.instance().addMapLayer(output_layer)
                self.scanlines_clip_split = output_layer
                self.log_browser.append("Temporary layer 'scanlines_clip_split' created and added to canvas with filtered segments.")

                # Apply style to scanlines_clip_split layer
                if renderer and symbol:
                    new_symbol_split = symbol.clone()
                    new_symbol_split.setWidth(symbol.width() * 2) # Same doubled thickness
                    new_renderer_split = QgsSingleSymbolRenderer(new_symbol_split)
                    self.scanlines_clip_split.setRenderer(new_renderer_split)
                    self.scanlines_clip_split.triggerRepaint()
                    self.log_browser.append("Style applied to 'scanlines_clip_split' with doubled line thickness.")
                else:
                    self.log_browser.append("Warning: Could not get renderer/symbol from scanlines layer. Cannot apply style to scanlines_clip_split.")


                # Remove the temporary split_layer from the project if it was added
                QgsProject.instance().removeMapLayer(split_layer.id())


                # Style the intersection layer (existing code)
                if renderer and symbol:
                    symbol_layer = symbol.symbolLayer(0)
                    if symbol_layer:
                        # Convert QgsUnitTypes.RenderUnit enum to string for createSimple
                        unit_string = QgsUnitTypes.encodeUnit(symbol_layer.widthUnit())

                        point_symbol = QgsMarkerSymbol.createSimple({
                            'name': 'circle',
                            'color': 'white',  # Fill color is white
                            'outline_color': symbol_layer.color().name(), # Outline color from scanlines
                            'outline_width': str(symbol_layer.width()), # Outline width same as scanlines line width
                            'outline_width_unit': unit_string, # Use converted unit string
                            'size': str(symbol_layer.width() * 6),
                            'size_unit': unit_string # Use converted unit string
                        })
                        point_renderer = QgsSingleSymbolRenderer(point_symbol)
                        self.intersections_layer.setRenderer(point_renderer)
                        self.intersections_layer.triggerRepaint()
                        self.log_browser.append("Style applied to 'intersections' layer.")
                    else:
                        self.log_browser.append("Warning: Could not get symbol layer from scanlines layer. Cannot apply style to intersections.")
                else:
                    self.log_browser.append("Warning: Could not get style from scanlines layer. Cannot apply style to intersections.")

            else:
                self.log_browser.append("No fractures layer selected. Skipping intersection and splitting.")

        except Exception as e:
            self.log_browser.append(f"ERROR during analysis: {e}")
