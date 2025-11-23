"""
Freshness analyzer module.
Combines ML predictions with rule-based analysis to determine freshness.
"""

import numpy as np
from typing import Dict, Tuple
from image_processor import ImageProcessor
from model_loader import FreshnessModel
from nutrition_lookup import NutritionLookup


class FreshnessAnalyzer:
    """Analyzes freshness of fruits/vegetables using ML and rule-based methods."""
    
    def __init__(self, model_path: str = "models/freshness_model"):
        """
        Initialize freshness analyzer.
        
        Args:
            model_path: Path to trained model directory
        """
        self.processor = ImageProcessor()
        self.model = FreshnessModel(model_path)
        self.nutrition_lookup = NutritionLookup()
    
    def analyze(self, image: np.ndarray) -> Dict:
        """
        Analyze image to determine item and freshness.
        
        Args:
            image: Input image as numpy array (BGR format)
            
        Returns:
            Dictionary with analysis results
        """
        # Extract features for rule-based analysis and heatmap
        features_dict = self.processor.extract_all_features(image)
        features_array = np.array(list(features_dict.values()))
        
        # Predict item using CNN model (takes image directly)
        item_name, confidence = self.model.predict_item(image)
        
        # Predict freshness using CNN model (takes image directly)
        freshness_ml = self.model.predict_freshness(image, item_name)
        
        # Rule-based freshness adjustment
        freshness_rule = self._rule_based_freshness(features_dict, item_name)
        
        # Combine ML and rule-based predictions
        # Use more weight on ML if confidence is high
        ml_weight = 0.8 if confidence > 0.7 else 0.6
        freshness_final = (freshness_ml * ml_weight + freshness_rule * (1 - ml_weight))
        freshness_final = max(0.0, min(100.0, freshness_final))
        
        # Get nutrition and storage info
        nutrition_info = self.nutrition_lookup.get_nutrition_info(item_name)
        storage_info = self.nutrition_lookup.get_storage_info(item_name, freshness_final)
        benefits = self.nutrition_lookup.get_benefits(item_name)
        recipes = self.nutrition_lookup.get_recipes(item_name)
        
        # Calculate usable days
        usable_days = self._calculate_usable_days(freshness_final, storage_info)
        
        # Generate action suggestion
        action_suggestion = self._generate_action_suggestion(freshness_final, usable_days)
        
        # Determine prediction basis
        prediction_basis = self._get_prediction_basis(features_dict, freshness_final)
        
        return {
            'item_name': item_name,
            'item_display_name': nutrition_info.get('name', item_name) if nutrition_info else item_name,
            'hindi_name': nutrition_info.get('hindi_name', '') if nutrition_info else '',
            'confidence': float(confidence),
            'freshness_percentage': float(freshness_final),
            'freshness_ml': float(freshness_ml),
            'freshness_rule': float(freshness_rule),
            'usable_days_ambient': usable_days['ambient'],
            'usable_days_fridge': usable_days['fridge'],
            'calories': self.nutrition_lookup.get_calories(item_name),
            'benefits': benefits,
            'vitamins': nutrition_info.get('vitamins', []) if nutrition_info else [],
            'minerals': nutrition_info.get('minerals', []) if nutrition_info else [],
            'fiber_g': nutrition_info.get('fiber_g', 0) if nutrition_info else 0,
            'storage_tips': storage_info.get('storage_tips', ''),
            'action_suggestion': action_suggestion,
            'prediction_basis': prediction_basis,
            'recipes': recipes,
            'features': features_dict
        }
    
    def _rule_based_freshness(self, features: Dict, item_name: str) -> float:
        """
        Rule-based freshness estimation using color and texture features.
        
        Args:
            features: Dictionary of extracted features
            item_name: Name of the fruit/vegetable
            
        Returns:
            Estimated freshness percentage (0-100)
        """
        freshness = 75.0  # Base freshness
        
        # Color-based rules
        brightness = features.get('V_mean', 128) / 255.0  # Normalize to 0-1
        saturation = features.get('S_mean', 128) / 255.0
        
        # Higher brightness and saturation generally indicate freshness
        if brightness > 0.6:
            freshness += 10
        elif brightness < 0.4:
            freshness -= 15
        
        if saturation > 0.5:
            freshness += 5
        elif saturation < 0.3:
            freshness -= 10
        
        # Texture-based rules
        # Lower contrast and higher homogeneity might indicate uniform texture (fresh)
        contrast = features.get('glcm_contrast_mean', 0.5)
        homogeneity = features.get('glcm_homogeneity_mean', 0.5)
        
        if contrast < 0.3:
            freshness += 5
        elif contrast > 0.7:
            freshness -= 10
        
        if homogeneity > 0.7:
            freshness += 5
        elif homogeneity < 0.4:
            freshness -= 10
        
        # Edge density (more edges might indicate wrinkles/decay)
        edge_density = features.get('edge_density', 0.1)
        if edge_density > 0.15:
            freshness -= 10
        elif edge_density < 0.05:
            freshness += 5
        
        # Color variance (uniform color might indicate freshness)
        color_variance = features.get('color_variance', 1000)
        if color_variance < 500:
            freshness += 5
        elif color_variance > 2000:
            freshness -= 10
        
        # Clamp to 0-100
        freshness = max(0.0, min(100.0, freshness))
        
        return freshness
    
    def _calculate_usable_days(self, freshness: float, storage_info: Dict) -> Dict:
        """
        Calculate usable days based on freshness percentage.
        
        Args:
            freshness: Freshness percentage (0-100)
            storage_info: Storage information dictionary
            
        Returns:
            Dictionary with usable days for ambient and fridge storage
        """
        base_ambient = storage_info.get('base_days_ambient', 3)
        base_fridge = storage_info.get('base_days_fridge', 10)
        
        # Linear scaling based on freshness
        freshness_factor = freshness / 100.0
        
        days_ambient = max(0, int(base_ambient * freshness_factor))
        days_fridge = max(0, int(base_fridge * freshness_factor))
        
        return {
            'ambient': days_ambient,
            'fridge': days_fridge
        }
    
    def _generate_action_suggestion(self, freshness: float, usable_days: Dict) -> str:
        """
        Generate action suggestion based on freshness.
        
        Args:
            freshness: Freshness percentage (0-100)
            usable_days: Dictionary with usable days
            
        Returns:
            Action suggestion string
        """
        if freshness >= 80:
            return "Use now - Excellent freshness! Best quality."
        elif freshness >= 60:
            days = max(usable_days['ambient'], usable_days['fridge'])
            return f"Use within {days} days - Good freshness. Store in refrigerator for longer shelf life."
        elif freshness >= 40:
            return "Use soon - Moderate freshness. Consider cooking or freezing to preserve."
        elif freshness >= 20:
            return "Use immediately or cook - Low freshness. Best used in cooked dishes."
        else:
            return "Discard - Poor freshness. Not recommended for consumption."
    
    def _get_prediction_basis(self, features: Dict, freshness: float) -> str:
        """
        Generate explanation of prediction basis.
        
        Args:
            features: Dictionary of extracted features
            freshness: Freshness percentage
            
        Returns:
            Explanation string
        """
        factors = []
        
        brightness = features.get('V_mean', 128) / 255.0
        if brightness > 0.6:
            factors.append("bright appearance")
        elif brightness < 0.4:
            factors.append("dull appearance")
        
        saturation = features.get('S_mean', 128) / 255.0
        if saturation > 0.5:
            factors.append("vibrant colors")
        elif saturation < 0.3:
            factors.append("faded colors")
        
        contrast = features.get('glcm_contrast_mean', 0.5)
        if contrast < 0.3:
            factors.append("uniform texture")
        elif contrast > 0.7:
            factors.append("irregular texture")
        
        edge_density = features.get('edge_density', 0.1)
        if edge_density > 0.15:
            factors.append("surface irregularities")
        
        if not factors:
            factors.append("overall appearance")
        
        basis = f"Based on analysis of {', '.join(factors)}"
        
        if freshness >= 70:
            basis += ". The item shows signs of good freshness."
        elif freshness >= 40:
            basis += ". The item shows moderate signs of aging."
        else:
            basis += ". The item shows significant signs of deterioration."
        
        return basis

