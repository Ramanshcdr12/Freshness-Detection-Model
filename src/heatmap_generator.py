"""
Mock heatmap generator for visualizing areas affecting freshness decisions.
Creates simulated heatmaps based on color and texture analysis.
"""

import numpy as np
import cv2
from typing import Tuple
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap


class HeatmapGenerator:
    """Generates mock heatmaps to visualize freshness analysis."""
    
    def __init__(self):
        """Initialize heatmap generator."""
        pass
    
    def generate_heatmap(self, image: np.ndarray, features: dict, 
                        freshness: float) -> np.ndarray:
        """
        Generate mock heatmap overlay on image.
        
        Args:
            image: Input image (BGR format)
            features: Dictionary of extracted features
            freshness: Freshness percentage (0-100)
            
        Returns:
            Heatmap overlay as numpy array (same size as image)
        """
        h, w = image.shape[:2]
        heatmap = np.zeros((h, w), dtype=np.float32)
        
        # Convert to RGB for processing
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Create heatmap based on various factors
        
        # 1. Color-based regions (brightness and saturation)
        brightness_map = hsv[:, :, 2].astype(np.float32) / 255.0
        saturation_map = hsv[:, :, 1].astype(np.float32) / 255.0
        
        # Higher brightness and saturation = higher heat (fresher areas)
        color_score = (brightness_map * 0.6 + saturation_map * 0.4)
        
        # 2. Texture-based regions (edge detection)
        edges = cv2.Canny(gray, 50, 150)
        edge_map = (edges > 0).astype(np.float32)
        
        # More edges = lower heat (less fresh areas)
        texture_score = 1.0 - (edge_map * 0.3)
        
        # 3. Color uniformity (variance in local regions)
        # Calculate local color variance
        kernel_size = 15
        kernel = np.ones((kernel_size, kernel_size), np.float32) / (kernel_size * kernel_size)
        
        gray_float = gray.astype(np.float32)
        local_mean = cv2.filter2D(gray_float, -1, kernel)
        local_sq_mean = cv2.filter2D(gray_float ** 2, -1, kernel)
        local_variance = local_sq_mean - local_mean ** 2
        
        # Normalize variance
        variance_norm = 1.0 - np.clip(local_variance / 1000.0, 0, 1)
        
        # Combine factors
        heatmap = (color_score * 0.4 + texture_score * 0.3 + variance_norm * 0.3)
        
        # Adjust based on overall freshness
        # If freshness is low, highlight problematic areas more
        if freshness < 50:
            # Invert heatmap to highlight problematic areas
            heatmap = 1.0 - heatmap
            heatmap = np.power(heatmap, 0.7)  # Enhance contrast
        
        # Apply Gaussian blur for smoother heatmap
        heatmap = cv2.GaussianBlur(heatmap, (21, 21), 0)
        
        # Normalize to 0-1 range
        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-7)
        
        return heatmap
    
    def overlay_heatmap(self, image: np.ndarray, heatmap: np.ndarray, 
                       alpha: float = 0.5) -> np.ndarray:
        """
        Overlay heatmap on image.
        
        Args:
            image: Original image (BGR format)
            heatmap: Heatmap array (0-1 range)
            alpha: Transparency of heatmap overlay (0-1)
            
        Returns:
            Image with heatmap overlay (BGR format)
        """
        # Convert heatmap to colormap
        colormap = plt.cm.get_cmap('jet')
        heatmap_colored = colormap(heatmap)[:, :, :3]  # Remove alpha channel
        heatmap_colored = (heatmap_colored * 255).astype(np.uint8)
        heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_RGB2BGR)
        
        # Overlay on original image
        overlay = cv2.addWeighted(image, 1 - alpha, heatmap_colored, alpha, 0)
        
        return overlay
    
    def create_visualization(self, image: np.ndarray, features: dict, 
                           freshness: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create complete heatmap visualization.
        
        Args:
            image: Input image (BGR format)
            features: Dictionary of extracted features
            freshness: Freshness percentage (0-100)
            
        Returns:
            Tuple of (heatmap array, overlay image)
        """
        heatmap = self.generate_heatmap(image, features, freshness)
        overlay = self.overlay_heatmap(image, heatmap, alpha=0.4)
        
        return heatmap, overlay
    
    def get_heatmap_explanation(self, heatmap: np.ndarray, freshness: float) -> str:
        """
        Generate explanation text for heatmap.
        
        Args:
            heatmap: Heatmap array
            freshness: Freshness percentage
            
        Returns:
            Explanation string
        """
        # Calculate statistics
        if heatmap is None:
            return "Heatmap could not be generated."

        mean_heat = np.mean(heatmap)
        std_heat = np.std(heatmap)
        
        if freshness >= 70:
            explanation = "Heatmap shows predominantly warm colors (red/yellow) indicating good freshness across the surface. "
        elif freshness >= 40:
            explanation = "Heatmap shows mixed colors with some cooler regions (blue/green) indicating moderate freshness. "
        else:
            explanation = "Heatmap shows predominantly cool colors (blue/green) indicating areas of concern. "
        
        if std_heat > 0.2:
            explanation += "Noticeable variation in color suggests uneven freshness or surface irregularities."
        else:
            explanation += "Uniform color distribution suggests consistent freshness across the surface."
        
        return explanation


