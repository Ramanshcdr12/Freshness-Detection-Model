"""
Image preprocessing and feature extraction module for freshness detection.
Uses OpenCV and scikit-image for color, texture, and shape analysis.
"""

import cv2
import numpy as np
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern
from skimage import color, measure
from typing import Dict, Tuple, List
import warnings
warnings.filterwarnings('ignore')


class ImageProcessor:
    """Processes images to extract features for freshness detection."""
    
    def __init__(self, target_size: Tuple[int, int] = (224, 224)):
        """
        Initialize image processor.
        
        Args:
            target_size: Target size for image resizing (width, height)
        """
        self.target_size = target_size
    
    def preprocess_image(self, image: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Preprocess image: resize, normalize, convert color spaces.
        
        Args:
            image: Input image as numpy array (BGR format from OpenCV)
            
        Returns:
            Dictionary containing processed images in different color spaces
        """
        # Resize image
        resized = cv2.resize(image, self.target_size)
        
        # Convert to different color spaces
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)
        lab = cv2.cvtColor(resized, cv2.COLOR_BGR2LAB)
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        
        # Normalize images
        rgb_norm = rgb.astype(np.float32) / 255.0
        hsv_norm = hsv.astype(np.float32)
        hsv_norm[:, :, 0] = hsv_norm[:, :, 0] / 180.0  # Hue: 0-180 -> 0-1
        hsv_norm[:, :, 1:] = hsv_norm[:, :, 1:] / 255.0  # Saturation, Value: 0-255 -> 0-1
        lab_norm = lab.astype(np.float32) / 255.0
        gray_norm = gray.astype(np.float32) / 255.0
        
        return {
            'rgb': rgb,
            'hsv': hsv,
            'lab': lab,
            'gray': gray,
            'rgb_norm': rgb_norm,
            'hsv_norm': hsv_norm,
            'lab_norm': lab_norm,
            'gray_norm': gray_norm
        }
    
    def extract_color_features(self, processed: Dict[str, np.ndarray]) -> Dict[str, float]:
        """
        Extract color features from image.
        
        Args:
            processed: Dictionary of processed images
            
        Returns:
            Dictionary of color features
        """
        rgb = processed['rgb']
        hsv = processed['hsv']
        lab = processed['lab']
        
        features = {}
        
        # RGB channel statistics
        for i, channel in enumerate(['R', 'G', 'B']):
            channel_data = rgb[:, :, i].flatten()
            features[f'{channel}_mean'] = float(np.mean(channel_data))
            features[f'{channel}_std'] = float(np.std(channel_data))
            features[f'{channel}_skew'] = float(self._skewness(channel_data))
            features[f'{channel}_kurtosis'] = float(self._kurtosis(channel_data))
        
        # HSV channel statistics
        for i, channel in enumerate(['H', 'S', 'V']):
            channel_data = hsv[:, :, i].flatten()
            features[f'{channel}_mean'] = float(np.mean(channel_data))
            features[f'{channel}_std'] = float(np.std(channel_data))
        
        # LAB channel statistics
        for i, channel in enumerate(['L', 'A', 'B']):
            channel_data = lab[:, :, i].flatten()
            features[f'{channel}_mean'] = float(np.mean(channel_data))
            features[f'{channel}_std'] = float(np.std(channel_data))
        
        # Color histogram features
        hist_rgb = cv2.calcHist([rgb], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
        features['hist_entropy'] = float(self._entropy(hist_rgb.flatten()))
        
        # Dominant color analysis
        features['dominant_hue'] = float(np.mean(hsv[:, :, 0]))
        features['saturation_mean'] = float(np.mean(hsv[:, :, 1]))
        features['brightness_mean'] = float(np.mean(hsv[:, :, 2]))
        
        # Color diversity (variance in color space)
        rgb_reshaped = rgb.reshape(-1, 3)
        features['color_variance'] = float(np.var(rgb_reshaped, axis=0).mean())
        
        return features
    
    def extract_texture_features(self, processed: Dict[str, np.ndarray]) -> Dict[str, float]:
        """
        Extract texture features using GLCM and LBP.
        
        Args:
            processed: Dictionary of processed images
            
        Returns:
            Dictionary of texture features
        """
        gray = processed['gray']
        features = {}
        
        # Quantize grayscale image for GLCM (0-255 -> 0-15)
        gray_quantized = (gray / 16).astype(np.uint8)
        
        # GLCM (Gray-Level Co-occurrence Matrix) features
        distances = [1, 2]
        angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
        
        glcm = graycomatrix(gray_quantized, distances=distances, angles=angles, 
                           levels=16, symmetric=True, normed=True)
        
        # GLCM properties
        properties = ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation']
        for prop in properties:
            prop_values = graycoprops(glcm, prop)
            features[f'glcm_{prop}_mean'] = float(np.mean(prop_values))
            features[f'glcm_{prop}_std'] = float(np.std(prop_values))
        
        # LBP (Local Binary Pattern) features
        radius = 3
        n_points = 8 * radius
        lbp = local_binary_pattern(gray, n_points, radius, method='uniform')
        
        # LBP histogram features
        hist_lbp, _ = np.histogram(lbp.ravel(), bins=n_points + 2, range=(0, n_points + 2))
        hist_lbp = hist_lbp.astype(float)
        hist_lbp /= (hist_lbp.sum() + 1e-7)  # Normalize
        
        features['lbp_entropy'] = float(self._entropy(hist_lbp))
        features['lbp_uniformity'] = float(np.max(hist_lbp))
        features['lbp_mean'] = float(np.mean(lbp))
        features['lbp_std'] = float(np.std(lbp))
        
        # Edge features
        edges = cv2.Canny(gray, 50, 150)
        features['edge_density'] = float(np.sum(edges > 0) / (edges.shape[0] * edges.shape[1]))
        
        # Gradient features
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        features['gradient_mean'] = float(np.mean(gradient_magnitude))
        features['gradient_std'] = float(np.std(gradient_magnitude))
        
        return features
    
    def extract_shape_features(self, processed: Dict[str, np.ndarray]) -> Dict[str, float]:
        """
        Extract shape features from image.
        
        Args:
            processed: Dictionary of processed images
            
        Returns:
            Dictionary of shape features
        """
        gray = processed['gray']
        features = {}
        
        # Threshold to get binary image
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Find contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if len(contours) > 0:
            # Get largest contour
            largest_contour = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(largest_contour)
            perimeter = cv2.arcLength(largest_contour, True)
            
            # Shape features
            features['area'] = float(area)
            features['perimeter'] = float(perimeter)
            
            if perimeter > 0:
                features['circularity'] = float(4 * np.pi * area / (perimeter ** 2))
            else:
                features['circularity'] = 0.0
            
            # Bounding box features
            x, y, w, h = cv2.boundingRect(largest_contour)
            features['aspect_ratio'] = float(w / (h + 1e-7))
            features['extent'] = float(area / (w * h + 1e-7))
            
            # Convex hull features
            hull = cv2.convexHull(largest_contour)
            hull_area = cv2.contourArea(hull)
            if hull_area > 0:
                features['solidity'] = float(area / hull_area)
            else:
                features['solidity'] = 0.0
        else:
            # Default values if no contours found
            features['area'] = 0.0
            features['perimeter'] = 0.0
            features['circularity'] = 0.0
            features['aspect_ratio'] = 1.0
            features['extent'] = 0.0
            features['solidity'] = 0.0
        
        # Image dimensions
        features['width'] = float(gray.shape[1])
        features['height'] = float(gray.shape[0])
        features['total_pixels'] = float(gray.shape[0] * gray.shape[1])
        
        return features
    
    def extract_all_features(self, image: np.ndarray) -> Dict[str, float]:
        """
        Extract all features from image.
        
        Args:
            image: Input image as numpy array (BGR format)
            
        Returns:
            Dictionary of all extracted features
        """
        processed = self.preprocess_image(image)
        
        color_features = self.extract_color_features(processed)
        texture_features = self.extract_texture_features(processed)
        shape_features = self.extract_shape_features(processed)
        
        # Combine all features
        all_features = {**color_features, **texture_features, **shape_features}
        
        return all_features
    
    def _skewness(self, data: np.ndarray) -> float:
        """Calculate skewness of data."""
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0.0
        n = len(data)
        skew = (n / ((n - 1) * (n - 2))) * np.sum(((data - mean) / std) ** 3)
        return skew
    
    def _kurtosis(self, data: np.ndarray) -> float:
        """Calculate kurtosis of data."""
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0.0
        n = len(data)
        kurt = (n * (n + 1) / ((n - 1) * (n - 2) * (n - 3))) * np.sum(((data - mean) / std) ** 4) - 3 * (n - 1) ** 2 / ((n - 2) * (n - 3))
        return kurt
    
    def _entropy(self, data: np.ndarray) -> float:
        """Calculate entropy of data."""
        data = data[data > 0]  # Remove zeros
        if len(data) == 0:
            return 0.0
        data = data / (data.sum() + 1e-7)  # Normalize
        entropy = -np.sum(data * np.log2(data + 1e-7))
        return entropy


def load_image_from_bytes(image_bytes: bytes) -> np.ndarray:
    """
    Load image from bytes.
    
    Args:
        image_bytes: Image data as bytes
        
    Returns:
        Image as numpy array (BGR format) or None if decoding fails
    """
    try:
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if image is not None and image.size > 0:
            return image
        return None
    except Exception as e:
        print(f"Error decoding image: {e}")
        return None


def load_image_from_path(image_path: str) -> np.ndarray:
    """
    Load image from file path.
    
    Args:
        image_path: Path to image file
        
    Returns:
        Image as numpy array (BGR format)
    """
    image = cv2.imread(image_path)
    return image

