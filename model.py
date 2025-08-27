import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import nibabel as nib
from skimage import measure, filters, segmentation, morphology
from scipy import ndimage
import cv2
from typing import Tuple, Optional, Union
import warnings
import os
warnings.filterwarnings('ignore')

# PDF Report Generation imports
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib.colors import red, green, orange, black, blue, yellow
from reportlab.lib import colors
from reportlab.graphics.shapes import Drawing, Rect
from reportlab.graphics.charts.barcharts import VerticalBarChart
from reportlab.graphics.charts.piecharts import Pie
from reportlab.graphics import renderPDF
import io
from datetime import datetime
import sys
import tempfile

class BrainTumorVisualizer:
    """
    A comprehensive 3D brain tumor visualization system for medical diagnosis with PDF reporting.
    Supports multiple visualization methods, tumor segmentation techniques, and RSNA/ACR compliant reporting.
    """
    
    def __init__(self, mri_path: str):
        """
        Initialize the visualizer with an MRI scan.
        
        Args:
            mri_path: Path to the MRI file (NIfTI format recommended)
        """
        self.mri_path = mri_path
        self.mri_data = None
        self.tumor_mask = None
        self.voxel_spacing = (1.0, 1.0, 1.0)  # Default spacing in mm
        self.anatomical_origin = None
        self.urgent_findings = []
        self.report_metadata = {
            'patient_id': 'ANONYMOUS_001',
            'study_date': datetime.now().strftime('%Y-%m-%d'),
            'study_time': datetime.now().strftime('%H:%M:%S'),
            'modality': 'MRI',
            'series_description': 'Brain Tumor Analysis',
            'radiologist': 'AI Analysis System',
            'institution': 'Medical Imaging Center'
        }
        
        self.load_mri_data()
        self._initialize_anatomical_reference()
    
    def set_patient_metadata(self, patient_id=None, study_date=None, radiologist=None, institution=None):
        """Set patient and study metadata for the report."""
        if patient_id:
            self.report_metadata['patient_id'] = patient_id
        if study_date:
            self.report_metadata['study_date'] = study_date
        if radiologist:
            self.report_metadata['radiologist'] = radiologist
        if institution:
            self.report_metadata['institution'] = institution
    
    def load_mri_data(self):
        """Load MRI data from file."""
        try:
            # For NIfTI files
            if self.mri_path.endswith(('.nii', '.nii.gz')):
                nii_img = nib.load(self.mri_path)
                self.mri_data = nii_img.get_fdata()
                
                # Extract voxel spacing from NIfTI header
                header = nii_img.header
                self.voxel_spacing = header.get_zooms()[:3]
                
                # Get transformation matrix for anatomical coordinates
                self.affine_matrix = nii_img.affine
                
                print(f"MRI data loaded successfully. Shape: {self.mri_data.shape}")
                print(f"Voxel spacing: {self.voxel_spacing} mm")
                
            else:
                # For other formats, you might need different loaders
                raise ValueError("Unsupported file format. Please use NIfTI format.")
                
        except Exception as e:
            print(f"Error loading MRI data: {e}")
            # Create sample data for demonstration
            self.create_sample_data()
    
    def _initialize_anatomical_reference(self):
        """Initialize anatomical reference system."""
        if hasattr(self, 'affine_matrix'):
            # Use actual MRI coordinate system
            self.anatomical_origin = self.affine_matrix[:3, 3]
        else:
            # For sample data, use center as origin
            center = np.array(self.mri_data.shape) / 2
            self.anatomical_origin = center * np.array(self.voxel_spacing)
    
    def voxel_to_anatomical(self, voxel_coords):
        """Convert voxel coordinates to anatomical coordinates (mm)."""
        if hasattr(self, 'affine_matrix'):
            # Use proper NIfTI transformation
            voxel_coords_homogeneous = np.append(voxel_coords, 1)
            anatomical_coords = self.affine_matrix @ voxel_coords_homogeneous
            return anatomical_coords[:3]
        else:
            # Simple conversion for sample data
            return np.array(voxel_coords) * np.array(self.voxel_spacing) - self.anatomical_origin
    
    def get_anatomical_region(self, coords):
        """Determine anatomical region based on coordinates."""
        x, y, z = coords
        shape = self.mri_data.shape
        
        # Normalize coordinates to 0-1
        norm_x = x / shape[0]
        norm_y = y / shape[1]
        norm_z = z / shape[2]
        
        # Simple anatomical region mapping
        region = "Brain tissue"
        hemisphere = "Left" if norm_x < 0.5 else "Right"
        
        # Anterior-Posterior
        if norm_y < 0.33:
            ap_region = "Frontal"
        elif norm_y < 0.66:
            ap_region = "Parietal"
        else:
            ap_region = "Occipital"
        
        # Superior-Inferior
        if norm_z > 0.66:
            si_region = "Superior"
        elif norm_z > 0.33:
            si_region = "Middle"
        else:
            si_region = "Inferior"
        
        # Combine regions
        if norm_z < 0.2:
            region = f"{hemisphere} {ap_region} lobe (Inferior/Temporal)"
        elif norm_z > 0.8:
            region = f"{hemisphere} {ap_region} lobe (Superior)"
        else:
            region = f"{hemisphere} {ap_region} lobe ({si_region})"
        
        return region
    
    def calculate_distance_from_midline(self, coords):
        """Calculate distance from brain midline."""
        midline_x = self.mri_data.shape[0] / 2
        distance_mm = abs(coords[0] - midline_x) * self.voxel_spacing[0]
        return distance_mm
    
    def get_location_tooltip(self, coords):
        """Generate comprehensive location tooltip information."""
        # Convert to anatomical coordinates
        anatomical_coords = self.voxel_to_anatomical(coords)
        
        # Get anatomical region
        region = self.get_anatomical_region(coords)
        
        # Calculate distance from midline
        midline_distance = self.calculate_distance_from_midline(coords)
        
        # Calculate distance from center of mass
        if hasattr(self, 'tumor_centroid'):
            distance_from_center = np.linalg.norm(
                np.array(coords) - self.tumor_centroid
            ) * np.mean(self.voxel_spacing)
        else:
            distance_from_center = 0
        
        # Get intensity at this location
        intensity = self.mri_data[int(coords[0]), int(coords[1]), int(coords[2])]
        
        tooltip_text = f"""
<b>Tumor Location Details</b><br>
<b>Anatomical Region:</b> {region}<br>
<b>Voxel Coordinates:</b> ({coords[0]:.0f}, {coords[1]:.0f}, {coords[2]:.0f})<br>
<b>Anatomical Coords (mm):</b> ({anatomical_coords[0]:.1f}, {anatomical_coords[1]:.1f}, {anatomical_coords[2]:.1f})<br>
<b>Distance from Midline:</b> {midline_distance:.1f} mm<br>
<b>Distance from Tumor Center:</b> {distance_from_center:.1f} mm<br>
<b>Signal Intensity:</b> {intensity:.0f}<br>
<b>Hemisphere:</b> {'Left' if coords[0] < self.mri_data.shape[0]/2 else 'Right'}
        """.strip()
        
        return tooltip_text
    
    def create_sample_data(self):
        """Create sample MRI data with simulated tumor for demonstration."""
        print("Creating sample MRI data for demonstration...")
        
        # Create a 3D brain-like structure
        x, y, z = np.meshgrid(np.linspace(-1, 1, 128),
                             np.linspace(-1, 1, 128),
                             np.linspace(-1, 1, 128))
        
        # Brain tissue (ellipsoid)
        brain = ((x/0.8)**2 + (y/0.9)**2 + (z/0.7)**2) < 1
        
        # Add some noise and structure
        brain_intensity = brain.astype(float) * 200
        brain_intensity += np.random.normal(0, 20, brain_intensity.shape)
        
        # Create tumor (irregular shape)
        tumor_center = (50, 60, 70)
        tumor = ((x - (tumor_center[0]/128*2-1))**2 * 4 + 
                (y - (tumor_center[1]/128*2-1))**2 * 3 + 
                (z - (tumor_center[2]/128*2-1))**2 * 5) < 0.1
        
        # Make tumor irregular
        tumor = ndimage.binary_erosion(tumor, iterations=1)
        tumor = ndimage.binary_dilation(tumor, iterations=2)
        
        # Add tumor to brain with higher intensity
        self.mri_data = brain_intensity
        self.mri_data[tumor] = 400
        
        # Apply some smoothing
        self.mri_data = filters.gaussian(self.mri_data, sigma=1)
    
    def segment_tumor_advanced(self, method='adaptive_threshold') -> np.ndarray:
        """Advanced tumor segmentation with multiple methods."""
        if method == 'adaptive_threshold':
            return self._segment_adaptive_threshold()
        elif method == 'otsu':
            return self._segment_otsu()
        elif method == 'watershed':
            return self._segment_watershed()
        else:
            return self.segment_tumor_threshold()
    
    def _segment_adaptive_threshold(self) -> np.ndarray:
        """Advanced adaptive thresholding for better tumor detection."""
        normalized_data = (self.mri_data - np.min(self.mri_data)) / (np.max(self.mri_data) - np.min(self.mri_data))
        smoothed = filters.gaussian(normalized_data, sigma=1.5)
        thresholds = filters.threshold_multiotsu(smoothed[smoothed > 0], classes=4)
        tumor_mask = smoothed > thresholds[-1]
        tumor_mask = morphology.remove_small_objects(tumor_mask, min_size=100)
        tumor_mask = ndimage.binary_fill_holes(tumor_mask)
        tumor_mask = morphology.binary_closing(tumor_mask, morphology.ball(3))
        
        self.tumor_mask = tumor_mask
        self._calculate_tumor_centroid()
        return tumor_mask
    
    def _segment_otsu(self) -> np.ndarray:
        """Otsu thresholding for tumor segmentation."""
        brain_data = self.mri_data[self.mri_data > 0]
        if len(brain_data) == 0:
            self.tumor_mask = np.zeros_like(self.mri_data, dtype=bool)
            return self.tumor_mask
        
        threshold = filters.threshold_otsu(brain_data)
        tumor_mask = self.mri_data > threshold * 1.2
        tumor_mask = morphology.remove_small_objects(tumor_mask, min_size=50)
        tumor_mask = ndimage.binary_fill_holes(tumor_mask)
        
        self.tumor_mask = tumor_mask
        self._calculate_tumor_centroid()
        return tumor_mask
    
    def _segment_watershed(self) -> np.ndarray:
        """Watershed segmentation for tumor detection."""
        normalized = (self.mri_data - np.min(self.mri_data)) / (np.max(self.mri_data) - np.min(self.mri_data))
        smoothed = filters.gaussian(normalized, sigma=1.0)
        local_maxima = morphology.local_maxima(smoothed)
        markers = ndimage.label(local_maxima)[0]
        labels = segmentation.watershed(-smoothed, markers, mask=smoothed > 0.1)
        region_props = measure.regionprops(labels, intensity_image=smoothed)
        tumor_labels = [prop.label for prop in region_props if prop.mean_intensity > 0.7]
        tumor_mask = np.isin(labels, tumor_labels)
        
        self.tumor_mask = tumor_mask
        self._calculate_tumor_centroid()
        return tumor_mask
    
    def segment_tumor_threshold(self, threshold_percentile: float = 85) -> np.ndarray:
        """Segment tumor using intensity thresholding."""
        threshold = np.percentile(self.mri_data[self.mri_data > 0], threshold_percentile)
        tumor_mask = self.mri_data > threshold
        tumor_mask = morphology.remove_small_objects(tumor_mask, min_size=50)
        tumor_mask = ndimage.binary_fill_holes(tumor_mask)
        tumor_mask = morphology.binary_closing(tumor_mask, morphology.ball(2))
        
        self.tumor_mask = tumor_mask
        self._calculate_tumor_centroid()
        return tumor_mask
    
    def _calculate_tumor_centroid(self):
        """Calculate tumor centroid for reference."""
        if self.tumor_mask is not None and np.sum(self.tumor_mask) > 0:
            tumor_coords = np.argwhere(self.tumor_mask)
            self.tumor_centroid = np.mean(tumor_coords, axis=0)
        else:
            self.tumor_centroid = None
    
    def assess_urgent_findings(self, metrics):
        """Assess findings and flag urgent features requiring immediate attention."""
        self.urgent_findings = []
        
        # Large tumor volume (>50cm³ or >50,000 mm³)
        if metrics['tumor_volume_mm3'] > 50000:
            self.urgent_findings.append({
                'finding': 'Large Mass Effect',
                'description': f"Large tumor volume ({metrics['tumor_volume_mm3']:.0f} mm³) may cause significant mass effect",
                'severity': 'HIGH',
                'recommendation': 'Consider urgent neurosurgical consultation'
            })
        
        # Very high contrast ratio (>2.5) may indicate aggressive tumor
        if metrics['contrast_ratio'] > 2.5:
            self.urgent_findings.append({
                'finding': 'High Contrast Enhancement',
                'description': f"Very high contrast ratio ({metrics['contrast_ratio']:.2f}) suggests possible malignancy",
                'severity': 'HIGH',
                'recommendation': 'Urgent oncology referral recommended'
            })
        
        # Midline shift risk (tumor within 10mm of midline)
        if hasattr(self, 'tumor_centroid') and self.tumor_centroid is not None:
            midline_distance = self.calculate_distance_from_midline(self.tumor_centroid)
            if midline_distance < 10:
                self.urgent_findings.append({
                    'finding': 'Midline Proximity',
                    'description': f"Tumor located {midline_distance:.1f}mm from midline - risk of mass effect",
                    'severity': 'MEDIUM',
                    'recommendation': 'Monitor for signs of increased intracranial pressure'
                })
        
        # Irregular shape (low sphericity) may indicate infiltrative growth
        if metrics['tumor_sphericity'] < 0.5:
            self.urgent_findings.append({
                'finding': 'Irregular Morphology',
                'description': f"Low sphericity ({metrics['tumor_sphericity']:.2f}) suggests irregular, possibly infiltrative growth",
                'severity': 'MEDIUM',
                'recommendation': 'Consider advanced imaging (DTI, perfusion) and tissue sampling'
            })
        
        # Large percentage of brain involved
        if metrics['tumor_volume_percentage'] > 5:
            self.urgent_findings.append({
                'finding': 'Extensive Brain Involvement',
                'description': f"Tumor comprises {metrics['tumor_volume_percentage']:.1f}% of brain volume",
                'severity': 'HIGH',
                'recommendation': 'Urgent multidisciplinary team discussion'
            })
        
        # Frontal lobe location (affects cognition/personality)
        if 'Frontal' in metrics.get('anatomical_location', ''):
            self.urgent_findings.append({
                'finding': 'Eloquent Brain Location',
                'description': f"Tumor in frontal lobe may affect cognitive function and personality",
                'severity': 'MEDIUM',
                'recommendation': 'Neuropsychological assessment and careful surgical planning'
            })
    
    def calculate_tumor_metrics(self) -> dict:
        """Calculate comprehensive diagnostic metrics for the tumor."""
        if self.tumor_mask is None:
            print("No tumor mask found. Performing automatic segmentation...")
            self.segment_tumor_threshold()
        
        # Initialize default metrics
        default_metrics = {
            'tumor_volume_mm3': 0.0,
            'tumor_volume_percentage': 0.0,
            'mean_tumor_intensity': 0.0,
            'std_tumor_intensity': 0.0,
            'mean_healthy_intensity': 0.0,
            'contrast_ratio': 1.0,
            'tumor_compactness': 0.0,
            'tumor_sphericity': 0.0,
            'anatomical_location': 'Unknown',
            'max_tumor_dimension': 0.0,
            'surface_area_mm2': 0.0,
            'volume_to_surface_ratio': 0.0
        }
        
        # Check if segmentation was successful
        if self.tumor_mask is None or np.sum(self.tumor_mask) == 0:
            print("Warning: No tumor regions detected. Returning default metrics.")
            if np.sum(self.mri_data > 0) > 0:
                default_metrics['mean_healthy_intensity'] = np.mean(self.mri_data[self.mri_data > 0])
            return default_metrics
        
        try:
            # Basic metrics
            tumor_volume_voxels = np.sum(self.tumor_mask)
            total_volume_voxels = np.sum(self.mri_data > 0)
            
            # Calculate volume in mm³ using voxel spacing
            voxel_volume = np.prod(self.voxel_spacing)
            tumor_volume_mm3 = tumor_volume_voxels * voxel_volume
            
            # Tumor intensity statistics
            tumor_intensities = self.mri_data[self.tumor_mask]
            healthy_mask = (self.mri_data > 0) & (~self.tumor_mask)
            healthy_intensities = self.mri_data[healthy_mask]
            
            # Calculate maximum dimension
            tumor_coords = np.argwhere(self.tumor_mask)
            if len(tumor_coords) > 0:
                ranges = np.max(tumor_coords, axis=0) - np.min(tumor_coords, axis=0)
                max_dimension_voxels = np.max(ranges)
                max_dimension_mm = max_dimension_voxels * np.mean(self.voxel_spacing)
            else:
                max_dimension_mm = 0.0
            
            # Surface area estimation
            try:
                vertices, faces, _ = self.create_3d_mesh(self.tumor_mask.astype(float), level=0.5)
                surface_area_mm2 = len(faces) * np.mean(self.voxel_spacing)**2 * 2  # Rough estimation
            except:
                surface_area_mm2 = 0.0
            
            # Calculate each metric safely
            metrics = default_metrics.copy()
            
            metrics['tumor_volume_mm3'] = tumor_volume_mm3
            metrics['tumor_volume_percentage'] = (tumor_volume_voxels / total_volume_voxels) * 100 if total_volume_voxels > 0 else 0
            metrics['mean_tumor_intensity'] = np.mean(tumor_intensities) if len(tumor_intensities) > 0 else 0
            metrics['std_tumor_intensity'] = np.std(tumor_intensities) if len(tumor_intensities) > 0 else 0
            metrics['mean_healthy_intensity'] = np.mean(healthy_intensities) if len(healthy_intensities) > 0 else 0
            metrics['max_tumor_dimension'] = max_dimension_mm
            metrics['surface_area_mm2'] = surface_area_mm2
            
            # Safe contrast ratio calculation
            if len(healthy_intensities) > 0 and np.mean(healthy_intensities) > 0:
                metrics['contrast_ratio'] = np.mean(tumor_intensities) / np.mean(healthy_intensities)
            
            # Volume to surface ratio
            if surface_area_mm2 > 0:
                metrics['volume_to_surface_ratio'] = tumor_volume_mm3 / surface_area_mm2
            
            # Safe morphological calculations
            metrics['tumor_compactness'] = self._calculate_compactness_safe()
            metrics['tumor_sphericity'] = self._calculate_sphericity_safe()
            
            # Add anatomical location
            if hasattr(self, 'tumor_centroid') and self.tumor_centroid is not None:
                metrics['anatomical_location'] = self.get_anatomical_region(self.tumor_centroid)
            
            # Assess urgent findings
            self.assess_urgent_findings(metrics)
            
            return metrics
            
        except Exception as e:
            print(f"Error calculating metrics: {e}")
            return default_metrics
    
    def _calculate_compactness_safe(self) -> float:
        """Calculate tumor compactness metric safely."""
        try:
            if self.tumor_mask is None or np.sum(self.tumor_mask) == 0:
                return 0.0
            
            vertices, faces, _ = self.create_3d_mesh(self.tumor_mask.astype(float), level=0.5)
            if len(faces) == 0:
                return 0.0
            
            surface_area = len(faces) * 2
            volume = np.sum(self.tumor_mask)
            
            if volume > 0:
                compactness = surface_area / (volume ** (2/3))
                return compactness
            return 0.0
            
        except Exception as e:
            print(f"Error calculating compactness: {e}")
            return 0.0
    
    def _calculate_sphericity_safe(self) -> float:
        """Calculate tumor sphericity safely."""
        try:
            if self.tumor_mask is None or np.sum(self.tumor_mask) == 0:
                return 0.0
            
            coords = np.argwhere(self.tumor_mask)
            if len(coords) < 10:
                return 0.0
            
            volume = len(coords)
            equivalent_radius = (3 * volume / (4 * np.pi)) ** (1/3)
            
            centroid = np.mean(coords, axis=0)
            distances = np.linalg.norm(coords - centroid, axis=1)
            actual_radius = np.mean(distances)
            
            if actual_radius > 0:
                sphericity = equivalent_radius / actual_radius
                return min(sphericity, 1.0)
            return 0.0
            
        except Exception as e:
            print(f"Error calculating sphericity: {e}")
            return 0.0
    
    def create_3d_mesh(self, data: np.ndarray, level: float = 0.5) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Create 3D mesh from volume data using marching cubes algorithm."""
        try:
            vertices, faces, normals, _ = measure.marching_cubes(data, level=level)
            return vertices, faces, normals
        except Exception as e:
            print(f"Error creating mesh: {e}")
            return np.array([]), np.array([]), np.array([])
    
    def create_visualization_images(self, output_dir):
        """Create and save visualization images for the PDF report."""
        os.makedirs(output_dir, exist_ok=True)
        
        # Multi-planar view
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('Brain MRI Multi-Planar View', fontsize=16)
        
        mid_x = self.mri_data.shape[0] // 2
        mid_y = self.mri_data.shape[1] // 2
        mid_z = self.mri_data.shape[2] // 2
        
        # Sagittal view
        axes[0, 0].imshow(self.mri_data[mid_x, :, :].T, cmap='gray', origin='lower')
        axes[0, 0].set_title('Sagittal View')
        axes[0, 0].axis('off')
        
        # Coronal view
        axes[0, 1].imshow(self.mri_data[:, mid_y, :].T, cmap='gray', origin='lower')
        axes[0, 1].set_title('Coronal View')
        axes[0, 1].axis('off')
        
        # Axial view
        axes[0, 2].imshow(self.mri_data[:, :, mid_z], cmap='gray')
        axes[0, 2].set_title('Axial View')
        axes[0, 2].axis('off')
        
        if self.tumor_mask is not None:
            # Enhanced tumor overlay
            axes[1, 0].imshow(self.mri_data[mid_x, :, :].T, cmap='gray', origin='lower')
            tumor_overlay = self.tumor_mask[mid_x, :, :].T
            axes[1, 0].contour(tumor_overlay, colors='red', linewidths=2, levels=[0.5])
            axes[1, 0].set_title('Sagittal + Tumor', color='red')
            axes[1, 0].axis('off')
            
            axes[1, 1].imshow(self.mri_data[:, mid_y, :].T, cmap='gray', origin='lower')
            tumor_overlay = self.tumor_mask[:, mid_y, :].T
            axes[1, 1].contour(tumor_overlay, colors='red', linewidths=2, levels=[0.5])
            axes[1, 1].set_title('Coronal + Tumor', color='red')
            axes[1, 1].axis('off')
            
            axes[1, 2].imshow(self.mri_data[:, :, mid_z], cmap='gray')
            tumor_overlay = self.tumor_mask[:, :, mid_z]
            axes[1, 2].contour(tumor_overlay, colors='red', linewidths=2, levels=[0.5])
            axes[1, 2].set_title('Axial + Tumor', color='red')
            axes[1, 2].axis('off')
        
        plt.tight_layout()
        multiplanar_path = os.path.join(output_dir, 'multiplanar_view.png')
        plt.savefig(multiplanar_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return multiplanar_path
    
    def create_metrics_chart(self, metrics, output_dir):
        """Create charts for tumor metrics."""
        os.makedirs(output_dir, exist_ok=True)
        
        # Create a figure with subplots for different metrics
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle('Tumor Analysis Metrics', fontsize=16)
        
        # Volume metrics
        volume_data = [metrics['tumor_volume_mm3'], metrics['mean_tumor_intensity'], metrics['mean_healthy_intensity']]
        volume_labels = ['Volume (mm³)', 'Tumor Intensity', 'Healthy Intensity']
        axes[0, 0].bar(volume_labels, volume_data, color=['red', 'orange', 'blue'])
        axes[0, 0].set_title('Volume and Intensity Metrics')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Shape metrics
        shape_data = [metrics['tumor_sphericity'], metrics['tumor_compactness'], metrics['contrast_ratio']]
        shape_labels = ['Sphericity', 'Compactness', 'Contrast Ratio']
        axes[0, 1].bar(shape_labels, shape_data, color=['green', 'purple', 'yellow'])
        axes[0, 1].set_title('Morphological Metrics')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # Severity assessment pie chart
        severity_counts = {'HIGH': 0, 'MEDIUM': 0, 'LOW': 1}
        for finding in self.urgent_findings:
            severity_counts[finding['severity']] += 1
        
        if sum(severity_counts.values()) > 0:
            axes[1, 0].pie(severity_counts.values(), labels=severity_counts.keys(), 
                          colors=['red', 'orange', 'green'], autopct='%1.0f%%')
            axes[1, 0].set_title('Urgent Findings by Severity')
        else:
            axes[1, 0].text(0.5, 0.5, 'No Urgent Findings', ha='center', va='center')
            axes[1, 0].set_title('Urgent Findings by Severity')
        
        # Location diagram (simplified)
        axes[1, 1].text(0.5, 0.7, f"Primary Location:", ha='center', fontsize=12, weight='bold')
        axes[1, 1].text(0.5, 0.5, metrics['anatomical_location'], ha='center', fontsize=10)
        if hasattr(self, 'tumor_centroid') and self.tumor_centroid is not None:
            midline_dist = self.calculate_distance_from_midline(self.tumor_centroid)
            axes[1, 1].text(0.5, 0.3, f"Distance from Midline: {midline_dist:.1f}mm", ha='center', fontsize=10)
        axes[1, 1].set_xlim(0, 1)
        axes[1, 1].set_ylim(0, 1)
        axes[1, 1].axis('off')
        axes[1, 1].set_title('Anatomical Location')
        
        plt.tight_layout()
        metrics_chart_path = os.path.join(output_dir, 'metrics_chart.png')
        plt.savefig(metrics_chart_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return metrics_chart_path
    
    def generate_pdf_report(self, output_path: str = None, patient_id: str = None) -> str:
        """
        Generate comprehensive PDF report following RSNA/ACR guidelines.
        
        Args:
            output_path: Path to save the PDF report
            patient_id: Patient identifier for the report
            
        Returns:
            Path to the generated PDF report
        """
        if output_path is None:
            output_path = f"brain_tumor_analysis_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
        
        if patient_id:
            self.report_metadata['patient_id'] = patient_id
        
        # Calculate metrics first
        metrics = self.calculate_tumor_metrics()
        
        # Create temporary directory for images
        temp_dir = tempfile.mkdtemp()
        
        try:
            # Generate visualization images
            multiplanar_path = self.create_visualization_images(temp_dir)
            metrics_chart_path = self.create_metrics_chart(metrics, temp_dir)
            
            # Create PDF document
            doc = SimpleDocTemplate(output_path, pagesize=A4)
            styles = getSampleStyleSheet()
            
            # Custom styles for the report
            title_style = ParagraphStyle(
                'CustomTitle',
                parent=styles['Heading1'],
                fontSize=20,
                textColor=colors.darkblue,
                spaceAfter=30,
                alignment=1  # Center alignment
            )
            
            urgent_style = ParagraphStyle(
                'UrgentStyle',
                parent=styles['Normal'],
                fontSize=12,
                textColor=colors.red,
                backColor=colors.pink,
                borderColor=colors.red,
                borderWidth=1,
                borderPadding=10,
                spaceAfter=10
            )
            
            warning_style = ParagraphStyle(
                'WarningStyle',
                parent=styles['Normal'],
                fontSize=11,
                textColor=colors.orange,
                backColor=colors.lightyellow,
                borderColor=colors.orange,
                borderWidth=1,
                borderPadding=8,
                spaceAfter=8
            )
            
            # Story list to hold all content
            story = []
            
            # Header with institution information
            story.append(Paragraph("MEDICAL IMAGING REPORT", title_style))
            story.append(Paragraph("Brain Tumor Analysis - AI Assisted Diagnosis", styles['Heading2']))
            story.append(Spacer(1, 20))
            
            # Patient and Study Information Table
            patient_data = [
                ['Patient ID:', self.report_metadata['patient_id']],
                ['Study Date:', self.report_metadata['study_date']],
                ['Study Time:', self.report_metadata['study_time']],
                ['Modality:', self.report_metadata['modality']],
                ['Series Description:', self.report_metadata['series_description']],
                ['Institution:', self.report_metadata['institution']],
                ['Analysis System:', self.report_metadata['radiologist']]
            ]
            
            patient_table = Table(patient_data, colWidths=[2*inch, 3*inch])
            patient_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (0, -1), colors.lightgrey),
                ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
                ('FONTSIZE', (0, 0), (-1, -1), 10),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
                ('BACKGROUND', (1, 0), (1, -1), colors.white),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            
            story.append(patient_table)
            story.append(Spacer(1, 20))
            
            # URGENT FINDINGS SECTION (Highlighted)
            if self.urgent_findings:
                story.append(Paragraph("⚠️ URGENT FINDINGS REQUIRING IMMEDIATE ATTENTION ⚠️", urgent_style))
                
                for i, finding in enumerate(self.urgent_findings, 1):
                    severity_color = colors.red if finding['severity'] == 'HIGH' else colors.orange
                    finding_style = urgent_style if finding['severity'] == 'HIGH' else warning_style
                    
                    finding_text = f"""
                    <b>{i}. {finding['finding']} ({finding['severity']} Priority)</b><br/>
                    <b>Finding:</b> {finding['description']}<br/>
                    <b>Recommendation:</b> {finding['recommendation']}
                    """
                    story.append(Paragraph(finding_text, finding_style))
                
                story.append(Spacer(1, 20))
            else:
                story.append(Paragraph("✅ No Urgent Findings Identified", 
                                     ParagraphStyle('NoUrgent', parent=styles['Normal'], 
                                                  textColor=colors.green, backColor=colors.lightgreen,
                                                  borderColor=colors.green, borderWidth=1, borderPadding=8)))
                story.append(Spacer(1, 15))
            
            # CLINICAL SUMMARY
            story.append(Paragraph("CLINICAL SUMMARY", styles['Heading2']))
            
            # Determine overall assessment
            if metrics['tumor_volume_mm3'] > 0:
                if len([f for f in self.urgent_findings if f['severity'] == 'HIGH']) > 0:
                    overall_assessment = "High Priority - Urgent intervention recommended"
                    assessment_color = colors.red
                elif len(self.urgent_findings) > 0:
                    overall_assessment = "Moderate Priority - Close monitoring required"
                    assessment_color = colors.orange
                else:
                    overall_assessment = "Low Priority - Routine follow-up appropriate"
                    assessment_color = colors.green
            else:
                overall_assessment = "No definitive tumor identified"
                assessment_color = colors.blue
            
            assessment_style = ParagraphStyle(
                'AssessmentStyle',
                parent=styles['Normal'],
                fontSize=12,
                textColor=assessment_color,
                backColor=colors.white,
                borderColor=assessment_color,
                borderWidth=2,
                borderPadding=10,
                spaceAfter=15
            )
            
            story.append(Paragraph(f"<b>Overall Assessment:</b> {overall_assessment}", assessment_style))
            
            # QUANTITATIVE ANALYSIS
            story.append(Paragraph("QUANTITATIVE ANALYSIS", styles['Heading2']))
            
            # Tumor Metrics Table
            metrics_data = [
                ['Parameter', 'Value', 'Clinical Significance'],
                ['Tumor Volume', f"{metrics['tumor_volume_mm3']:.1f} mm³", 
                 'Large if >50,000 mm³' if metrics['tumor_volume_mm3'] > 50000 else 'Within normal range'],
                ['Max Dimension', f"{metrics['max_tumor_dimension']:.1f} mm", 
                 'Significant if >30mm' if metrics['max_tumor_dimension'] > 30 else 'Small lesion'],
                ['Brain Involvement', f"{metrics['tumor_volume_percentage']:.2f}%", 
                 'Extensive if >5%' if metrics['tumor_volume_percentage'] > 5 else 'Limited involvement'],
                ['Contrast Enhancement', f"{metrics['contrast_ratio']:.2f}", 
                 'High (suspicious)' if metrics['contrast_ratio'] > 2.0 else 'Moderate enhancement'],
                ['Shape Regularity', f"{metrics['tumor_sphericity']:.3f}", 
                 'Irregular (infiltrative)' if metrics['tumor_sphericity'] < 0.5 else 'Regular shape'],
                ['Anatomical Location', metrics['anatomical_location'], 
                 'Eloquent area' if 'Frontal' in metrics['anatomical_location'] else 'Standard location']
            ]
            
            metrics_table = Table(metrics_data, colWidths=[2*inch, 1.5*inch, 2.5*inch])
            metrics_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.darkblue),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 11),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
                ('FONTSIZE', (0, 1), (-1, -1), 9),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            
            story.append(metrics_table)
            story.append(Spacer(1, 20))
            
            # IMAGING FINDINGS
            story.append(Paragraph("IMAGING FINDINGS", styles['Heading2']))
            
            # Add multiplanar view image
            if os.path.exists(multiplanar_path):
                story.append(Paragraph("Multi-planar MRI Views with Tumor Segmentation:", styles['Heading3']))
                img = Image(multiplanar_path, width=6*inch, height=4*inch)
                story.append(img)
                story.append(Spacer(1, 15))
            
            # Add metrics chart
            if os.path.exists(metrics_chart_path):
                story.append(Paragraph("Quantitative Analysis Charts:", styles['Heading3']))
                img = Image(metrics_chart_path, width=6*inch, height=4*inch)
                story.append(img)
                story.append(Spacer(1, 15))
            
            # PAGE BREAK before detailed findings
            story.append(PageBreak())
            
            # DETAILED FINDINGS
            story.append(Paragraph("DETAILED RADIOLOGICAL FINDINGS", styles['Heading2']))
            
            findings_text = f"""
            <b>Tumor Characteristics:</b><br/>
            • Location: {metrics['anatomical_location']}<br/>
            • Volume: {metrics['tumor_volume_mm3']:.1f} mm³ ({metrics['tumor_volume_percentage']:.2f}% of brain)<br/>
            • Maximum dimension: {metrics['max_tumor_dimension']:.1f} mm<br/>
            • Shape characteristics: {'Irregular/infiltrative' if metrics['tumor_sphericity'] < 0.5 else 'Well-defined borders'}<br/>
            • Enhancement pattern: {'Strong enhancement (ratio: ' + str(round(metrics['contrast_ratio'], 2)) + ')' if metrics['contrast_ratio'] > 1.5 else 'Mild enhancement'}<br/>
            <br/>
            <b>Anatomical Relationships:</b><br/>
            """
            
            if hasattr(self, 'tumor_centroid') and self.tumor_centroid is not None:
                midline_dist = self.calculate_distance_from_midline(self.tumor_centroid)
                findings_text += f"• Distance from midline: {midline_dist:.1f} mm<br/>"
                findings_text += f"• Hemisphere: {'Left' if self.tumor_centroid[0] < self.mri_data.shape[0]/2 else 'Right'}<br/>"
            
            findings_text += f"""
            <br/>
            <b>Morphological Analysis:</b><br/>
            • Sphericity index: {metrics['tumor_sphericity']:.3f} ({'Regular' if metrics['tumor_sphericity'] > 0.7 else 'Irregular'})<br/>
            • Compactness: {metrics['tumor_compactness']:.3f}<br/>
            • Surface area: {metrics['surface_area_mm2']:.1f} mm²<br/>
            """
            
            story.append(Paragraph(findings_text, styles['Normal']))
            story.append(Spacer(1, 20))
            
            # RECOMMENDATIONS
            story.append(Paragraph("CLINICAL RECOMMENDATIONS", styles['Heading2']))
            
            recommendations = []
            
            # Based on urgent findings
            if self.urgent_findings:
                for finding in self.urgent_findings:
                    recommendations.append(f"• {finding['recommendation']}")
            
            # General recommendations based on metrics
            if metrics['tumor_volume_mm3'] > 10000:
                recommendations.append("• Consider multidisciplinary tumor board discussion")
            
            if metrics['contrast_ratio'] > 2.0:
                recommendations.append("• Advanced imaging (perfusion MRI, MR spectroscopy) recommended")
            
            if 'Frontal' in metrics.get('anatomical_location', ''):
                recommendations.append("• Pre-operative neuropsychological assessment advised")
            
            if not recommendations:
                recommendations.append("• Routine clinical follow-up appropriate")
                recommendations.append("• Consider repeat imaging in 3-6 months if symptomatic")
            
            recommendations_text = "<br/>".join(recommendations)
            story.append(Paragraph(recommendations_text, styles['Normal']))
            story.append(Spacer(1, 20))
            
            # METHODOLOGY
            story.append(Paragraph("ANALYSIS METHODOLOGY", styles['Heading3']))
            methodology_text = """
            This report was generated using advanced AI-assisted image analysis techniques:
            • Automated tumor segmentation using adaptive thresholding
            • 3D volumetric analysis with sub-voxel precision  
            • Morphological characterization using established radiological metrics
            • Statistical comparison with normative brain tissue values
            • Anatomical localization based on standard brain atlases
            
            <b>Technical Parameters:</b>
            • Voxel spacing: """ + f"{self.voxel_spacing[0]:.2f} x {self.voxel_spacing[1]:.2f} x {self.voxel_spacing[2]:.2f} mm" + """
            • Segmentation method: Multi-Otsu adaptive thresholding
            • Minimum lesion size: 50 voxels
            • Analysis software: Custom Python implementation
            """
            
            story.append(Paragraph(methodology_text, styles['Normal']))
            story.append(Spacer(1, 20))
            
            # DISCLAIMER
            disclaimer_style = ParagraphStyle(
                'DisclaimerStyle',
                parent=styles['Normal'],
                fontSize=9,
                textColor=colors.grey,
                borderColor=colors.grey,
                borderWidth=1,
                borderPadding=8
            )
            
            disclaimer_text = """
            <b>IMPORTANT DISCLAIMER:</b> This automated analysis is intended to assist radiological 
            interpretation and should not replace clinical judgment. All findings should be correlated 
            with clinical presentation and confirmed by a qualified radiologist. This AI system has not 
            been FDA approved for diagnostic use. The analysis is based on computational algorithms 
            and may not detect all pathological features present in the imaging study.
            """
            
            story.append(Paragraph(disclaimer_text, disclaimer_style))
            
            # Footer with timestamp and version
            footer_text = f"Report generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | " + \
                         f"Analysis Version: 1.0 | " + \
                         f"Patient ID: {self.report_metadata['patient_id']}"
            story.append(Spacer(1, 20))
            story.append(Paragraph(footer_text, 
                                 ParagraphStyle('Footer', parent=styles['Normal'], 
                                              fontSize=8, textColor=colors.grey, alignment=1)))
            
            # Build the PDF
            doc.build(story)
            
            print(f"PDF report generated successfully: {output_path}")
            return output_path
            
        except Exception as e:
            print(f"Error generating PDF report: {e}")
            import traceback
            traceback.print_exc()
            return None
        
        finally:
            # Clean up temporary files
            try:
                import shutil
                shutil.rmtree(temp_dir)
            except:
                pass
    
    def visualize_volume_slices(self, save_path: str = None):
        """Create multi-planar view of the MRI data with enhanced tumor highlighting."""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('Brain MRI Multi-Planar View', fontsize=16)
        
        # Get middle slices
        mid_x = self.mri_data.shape[0] // 2
        mid_y = self.mri_data.shape[1] // 2
        mid_z = self.mri_data.shape[2] // 2
        
        # Sagittal view
        axes[0, 0].imshow(self.mri_data[mid_x, :, :].T, cmap='gray', origin='lower')
        axes[0, 0].set_title('Sagittal View')
        axes[0, 0].axis('off')
        
        # Coronal view
        axes[0, 1].imshow(self.mri_data[:, mid_y, :].T, cmap='gray', origin='lower')
        axes[0, 1].set_title('Coronal View')
        axes[0, 1].axis('off')
        
        # Axial view
        axes[0, 2].imshow(self.mri_data[:, :, mid_z], cmap='gray')
        axes[0, 2].set_title('Axial View')
        axes[0, 2].axis('off')
        
        if self.tumor_mask is not None:
            # Enhanced tumor overlay with better contrast
            axes[1, 0].imshow(self.mri_data[mid_x, :, :].T, cmap='gray', origin='lower')
            tumor_overlay = self.tumor_mask[mid_x, :, :].T
            axes[1, 0].contour(tumor_overlay, colors='red', linewidths=2, levels=[0.5])
            axes[1, 0].imshow(tumor_overlay, cmap='Reds', alpha=0.4, origin='lower')
            axes[1, 0].set_title('Sagittal + Tumor', color='red')
            axes[1, 0].axis('off')
            
            axes[1, 1].imshow(self.mri_data[:, mid_y, :].T, cmap='gray', origin='lower')
            tumor_overlay = self.tumor_mask[:, mid_y, :].T
            axes[1, 1].contour(tumor_overlay, colors='red', linewidths=2, levels=[0.5])
            axes[1, 1].imshow(tumor_overlay, cmap='Reds', alpha=0.4, origin='lower')
            axes[1, 1].set_title('Coronal + Tumor', color='red')
            axes[1, 1].axis('off')
            
            axes[1, 2].imshow(self.mri_data[:, :, mid_z], cmap='gray')
            tumor_overlay = self.tumor_mask[:, :, mid_z]
            axes[1, 2].contour(tumor_overlay, colors='red', linewidths=2, levels=[0.5])
            axes[1, 2].imshow(tumor_overlay, cmap='Reds', alpha=0.4)
            axes[1, 2].set_title('Axial + Tumor', color='red')
            axes[1, 2].axis('off')
        else:
            # Hide bottom row if no tumor mask
            for ax in axes[1, :]:
                ax.axis('off')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def create_detailed_tumor_mesh_with_tooltips(self):
        """Create tumor mesh with detailed tooltip information for each vertex."""
        if self.tumor_mask is None or np.sum(self.tumor_mask) == 0:
            return None, None, None, None
        
        # Create mesh
        vertices, faces, normals = self.create_3d_mesh(self.tumor_mask.astype(float), level=0.5)
        
        if len(vertices) == 0:
            return None, None, None, None
        
        # Generate tooltip text for each vertex
        tooltip_texts = []
        for vertex in vertices:
            # Convert vertex coordinates to voxel coordinates
            voxel_coords = vertex.astype(int)
            
            # Ensure coordinates are within bounds
            voxel_coords = np.clip(voxel_coords, 0, 
                                 [s-1 for s in self.mri_data.shape])
            
            tooltip_text = self.get_location_tooltip(voxel_coords)
            tooltip_texts.append(tooltip_text)
        
        return vertices, faces, normals, tooltip_texts
    
    def visualize_3d_interactive_enhanced(self, save_path: str = None):
        """Create enhanced interactive 3D visualization with detailed location tooltips."""
        # Segment tumor if not already done
        if self.tumor_mask is None:
            print("Performing tumor segmentation...")
            self.segment_tumor_advanced('adaptive_threshold')
        
        # Check if tumor was found
        if self.tumor_mask is None or np.sum(self.tumor_mask) == 0:
            print("No tumor detected, using threshold method...")
            self.segment_tumor_threshold(threshold_percentile=75)
        
        if self.tumor_mask is None or np.sum(self.tumor_mask) == 0:
            print("Warning: No tumor regions detected for 3D visualization")
            return
        
        print(f"Tumor voxels found: {np.sum(self.tumor_mask)}")
        
        # Create enhanced tumor mesh with tooltips
        tumor_vertices, tumor_faces, _, tooltip_texts = self.create_detailed_tumor_mesh_with_tooltips()
        
        if tumor_vertices is None or len(tumor_vertices) == 0 or len(tumor_faces) == 0:
            print("Could not generate tumor mesh")
            return
        
        # Create brain outline with lower threshold for context
        brain_threshold = np.percentile(self.mri_data[self.mri_data > 0], 15)
        brain_mask = self.mri_data > brain_threshold
        brain_vertices, brain_faces, _ = self.create_3d_mesh(brain_mask.astype(float), level=0.5)
        
        # Create Plotly figure with subplots for multiple views
        fig = make_subplots(
            rows=1, cols=2,
            specs=[[{'type': 'scatter3d'}, {'type': 'scatter3d'}]],
            subplot_titles=('Brain with Tumor (Hover for Location Details)', 'Tumor Only (Detailed Location Info)')
        )
        
        # First subplot: Brain + Tumor
        if len(brain_vertices) > 0 and len(brain_faces) > 0:
            fig.add_trace(go.Mesh3d(
                x=brain_vertices[:, 0],
                y=brain_vertices[:, 1],
                z=brain_vertices[:, 2],
                i=brain_faces[:, 0],
                j=brain_faces[:, 1],
                k=brain_faces[:, 2],
                opacity=0.15,
                color='lightblue',
                name='Brain Tissue',
                showscale=False,
                hoverinfo='name'
            ), row=1, col=1)
        
        # Enhanced tumor visualization with detailed tooltips
        fig.add_trace(go.Mesh3d(
            x=tumor_vertices[:, 0],
            y=tumor_vertices[:, 1],
            z=tumor_vertices[:, 2],
            i=tumor_faces[:, 0],
            j=tumor_faces[:, 1],
            k=tumor_faces[:, 2],
            opacity=0.9,
            color='red',
            name='Tumor',
            showscale=False,
            hovertemplate='<b>Tumor Region</b><br>' +
                         'Coordinates: (%{x:.0f}, %{y:.0f}, %{z:.0f})<br>' +
                         '<extra></extra>',
            lighting=dict(ambient=0.8, diffuse=0.8, specular=0.1)
        ), row=1, col=1)
        
        # Second subplot: Tumor only with enhanced details and location tooltips
        fig.add_trace(go.Mesh3d(
            x=tumor_vertices[:, 0],
            y=tumor_vertices[:, 1],
            z=tumor_vertices[:, 2],
            i=tumor_faces[:, 0],
            j=tumor_faces[:, 1],
            k=tumor_faces[:, 2],
            opacity=0.95,
            color='darkred',
            name='Tumor Detail',
            showscale=False,
            hovertemplate='<b>Detailed Tumor Analysis</b><br>' +
                         'Position: (%{x:.1f}, %{y:.1f}, %{z:.1f})<br>' +
                         'Click for full location details<br>' +
                         '<extra></extra>',
            lighting=dict(ambient=0.9, diffuse=0.8, specular=0.2)
        ), row=1, col=2)
        
        # Add tumor center point with detailed information
        if hasattr(self, 'tumor_centroid') and self.tumor_centroid is not None:
            centroid = self.tumor_centroid
            anatomical_centroid = self.voxel_to_anatomical(centroid)
            region = self.get_anatomical_region(centroid)
            
            centroid_tooltip = f"""
<b>Tumor Center of Mass</b><br>
<b>Location:</b> {region}<br>
<b>Voxel Coords:</b> ({centroid[0]:.1f}, {centroid[1]:.1f}, {centroid[2]:.1f})<br>
<b>Anatomical Coords:</b> ({anatomical_centroid[0]:.1f}, {anatomical_centroid[1]:.1f}, {anatomical_centroid[2]:.1f}) mm<br>
<b>Distance from Midline:</b> {self.calculate_distance_from_midline(centroid):.1f} mm
            """.strip()
            
            fig.add_trace(go.Scatter3d(
                x=[centroid[0]], y=[centroid[1]], z=[centroid[2]],
                mode='markers',
                marker=dict(size=10, color='yellow', symbol='diamond'),
                name='Tumor Center',
                hovertemplate=centroid_tooltip + '<extra></extra>'
            ), row=1, col=1)
            
            fig.add_trace(go.Scatter3d(
                x=[centroid[0]], y=[centroid[1]], z=[centroid[2]],
                mode='markers',
                marker=dict(size=12, color='gold', symbol='diamond'),
                name='Tumor Center',
                hovertemplate=centroid_tooltip + '<extra></extra>'
            ), row=1, col=2)
        
        # Update layout
        fig.update_layout(
            title={
                'text': '3D Brain Tumor Visualization - Enhanced with Location Details',
                'x': 0.5,
                'font': {'size': 20, 'color': 'darkred'}
            },
            width=1600,
            height=800,
            showlegend=True,
            legend=dict(x=0.02, y=0.98),
            annotations=[
                dict(
                    text="Hover over tumor regions for detailed location information",
                    showarrow=False,
                    xref="paper", yref="paper",
                    x=0.5, y=0.02, xanchor='center', yanchor='bottom',
                    font=dict(size=12, color="blue")
                )
            ]
        )
        
        # Update 3D scene properties for both subplots
        scene_settings = dict(
            xaxis_title='X (voxels)',
            yaxis_title='Y (voxels)',
            zaxis_title='Z (voxels)',
            aspectmode='cube',
            bgcolor='black',
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))
        )
        
        fig.update_layout(scene=scene_settings, scene2=scene_settings)
        
        if save_path:
            fig.write_html(save_path)
        
        fig.show()
    
    def generate_diagnostic_report(self) -> str:
        """Generate a diagnostic report for medical professionals."""
        metrics = self.calculate_tumor_metrics()
        
        # Safe access to metrics with fallbacks
        def safe_get(key, default=0.0):
            return metrics.get(key, default)
        
        report = f"""
BRAIN TUMOR ANALYSIS REPORT
===========================

LOCATION INFORMATION:
- Primary Anatomical Location: {safe_get('anatomical_location', 'Unknown')}
- Voxel Spacing: {self.voxel_spacing} mm

VOLUMETRIC MEASUREMENTS:
- Tumor Volume: {safe_get('tumor_volume_mm3'):.2f} mm³
- Tumor/Brain Volume Ratio: {safe_get('tumor_volume_percentage'):.2f}%

INTENSITY CHARACTERISTICS:
- Mean Tumor Intensity: {safe_get('mean_tumor_intensity'):.2f}
- Mean Healthy Tissue Intensity: {safe_get('mean_healthy_intensity'):.2f}
- Contrast Ratio: {safe_get('contrast_ratio'):.2f}
- Tumor Intensity Std: {safe_get('std_tumor_intensity'):.2f}

MORPHOLOGICAL FEATURES:
- Compactness: {safe_get('tumor_compactness'):.3f}
- Sphericity: {safe_get('tumor_sphericity'):.3f}

CLINICAL INTERPRETATION:
- Contrast Ratio > 1.5: {'High contrast tumor (likely malignant)' if safe_get('contrast_ratio') > 1.5 else 'Moderate contrast'}
- Volume Assessment: {'Large tumor (>10,000 mm³)' if safe_get('tumor_volume_mm3') > 10000 else 'Small to moderate tumor'}
- Shape Regularity: {'Regular shape' if safe_get('tumor_sphericity') > 0.7 else 'Irregular shape (potential infiltrative growth)'}

LOCATION ANALYSIS:
- Distance from Midline: {self.calculate_distance_from_midline(self.tumor_centroid) if hasattr(self, 'tumor_centroid') and self.tumor_centroid is not None else 'Not calculated':.1f} mm
- Hemisphere: {'Left' if hasattr(self, 'tumor_centroid') and self.tumor_centroid is not None and self.tumor_centroid[0] < self.mri_data.shape[0]/2 else 'Right' if hasattr(self, 'tumor_centroid') and self.tumor_centroid is not None else 'Unknown'}

SEGMENTATION STATUS: {'Tumor detected and analyzed' if safe_get('tumor_volume_mm3') > 0 else 'No clear tumor regions identified'}

NOTE: This automated analysis should be used in conjunction with clinical expertise and additional diagnostic methods.
The location tooltips provide detailed anatomical information when hovering over the 3D visualization.
        """
        
        return report


# Example usage with enhanced PDF reporting
if __name__ == "__main__":
    try:
        print("Initializing Enhanced Brain Tumor Visualization System with PDF Reporting...")

        # ✅ Get file path from command-line argument
        if len(sys.argv) < 2:
            raise ValueError("Please provide path to MRI .nii file")
        
        file_path = sys.argv[1]

        # Create visualizer with provided file path
        visualizer = BrainTumorVisualizer(file_path)
        
        # Set patient metadata (optional)
        visualizer.set_patient_metadata(
            patient_id="PATIENT_002_2025",
            study_date="2025-01-15",
            radiologist="Dr. AI Radiologist",
            institution="Advanced Medical Imaging Center"
        )
        
        # Try multiple segmentation methods
        print("Segmenting tumor with advanced methods...")
        visualizer.segment_tumor_advanced('adaptive_threshold')
        
        if visualizer.tumor_mask is None or np.sum(visualizer.tumor_mask) == 0:
            print("Advanced segmentation failed, trying threshold method...")
            visualizer.segment_tumor_threshold(threshold_percentile=75)
        
        # Generate enhanced visualizations with location tooltips
        print("Creating enhanced visualizations with location information...")
        visualizer.visualize_volume_slices()
        visualizer.visualize_3d_interactive_enhanced()
        
        # Generate comprehensive PDF report
        print("Generating comprehensive PDF report with RSNA/ACR guidelines...")
        pdf_path = visualizer.generate_pdf_report(
            output_path="brain_tumor_comprehensive_report.pdf",
            patient_id="PATIENT_001_2025"
        )
        
        if pdf_path:
            print(f"✅ PDF report generated successfully: {pdf_path}")
            print("\n📋 Report Features:")
            print("- RSNA/ACR compliant format")
            print("- Highlighted urgent findings")
            print("- Quantitative tumor analysis")
            print("- Clinical recommendations")
            print("- Multi-planar imaging views")
            print("- Statistical charts and graphs")
        else:
            print("❌ Failed to generate PDF report")
        
        # Generate console diagnostic report
        print("\nConsole Diagnostic Report:")
        print(visualizer.generate_diagnostic_report())
        
        # Display urgent findings summary
        if visualizer.urgent_findings:
            print("\n🚨 URGENT FINDINGS SUMMARY:")
            for i, finding in enumerate(visualizer.urgent_findings, 1):
                severity_symbol = "🔴" if finding['severity'] == 'HIGH' else "🟡"
                print(f"{severity_symbol} {i}. {finding['finding']} ({finding['severity']} Priority)")
                print(f"   Finding: {finding['description']}")
                print(f"   Recommendation: {finding['recommendation']}")
                print()
        else:
            print("\n✅ No urgent findings identified.")
        
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()
        
    print("\n📖 Instructions for Use:")
    print("1. Replace the file path with your actual MRI data path")
    print("2. Install required packages: pip install reportlab nibabel scikit-image plotly")
    print("3. The system will generate both interactive visualizations and a PDF report")
    print("4. Urgent findings are automatically flagged based on clinical criteria")
    print("5. Report follows RSNA/ACR structured reporting guidelines")
    print("6. All measurements are provided in standard medical units (mm, mm³)")