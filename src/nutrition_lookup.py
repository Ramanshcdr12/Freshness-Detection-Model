"""
Nutrition data lookup module.
Provides access to nutrition database and storage information.
"""

import json
import os
from typing import Dict, Optional, List


class NutritionLookup:
    """Lookup nutrition and storage information for fruits/vegetables."""
    
    def __init__(self, nutrition_db_path: str = "data/nutrition_db.json",
                 storage_tips_path: str = "data/storage_tips.json"):
        """
        Initialize nutrition lookup.
        
        Args:
            nutrition_db_path: Path to nutrition database JSON file
            storage_tips_path: Path to storage tips JSON file
        """
        self.nutrition_db = self._load_json(nutrition_db_path)
        self.storage_tips = self._load_json(storage_tips_path)
    
    def _load_json(self, filepath: str) -> Dict:
        """Load JSON file."""
        try:
            # Try relative path first
            if os.path.exists(filepath):
                with open(filepath, 'r', encoding='utf-8') as f:
                    return json.load(f)
            # Try from project root
            project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            full_path = os.path.join(project_root, filepath)
            if os.path.exists(full_path):
                with open(full_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            return {}
        except Exception as e:
            print(f"Error loading {filepath}: {e}")
            return {}
    
    def get_nutrition_info(self, item_name: str) -> Optional[Dict]:
        """
        Get nutrition information for an item.
        
        Args:
            item_name: Name of the fruit/vegetable (lowercase, with underscores)
            
        Returns:
            Dictionary with nutrition information or None if not found
        """
        # Try exact match first
        if item_name in self.nutrition_db:
            return self.nutrition_db[item_name]
        
        # Try case-insensitive match
        item_name_lower = item_name.lower().replace(' ', '_')
        if item_name_lower in self.nutrition_db:
            return self.nutrition_db[item_name_lower]
        
        # Try partial match
        for key in self.nutrition_db.keys():
            if item_name_lower in key or key in item_name_lower:
                return self.nutrition_db[key]
        
        return None
    
    def get_storage_info(self, item_name: str, freshness_percentage: float) -> Dict:
        """
        Get storage information based on freshness.
        
        Args:
            item_name: Name of the fruit/vegetable
            freshness_percentage: Freshness percentage (0-100)
            
        Returns:
            Dictionary with storage information
        """
        nutrition_info = self.get_nutrition_info(item_name)
        
        if not nutrition_info:
            return {
                'days_ambient': 0,
                'days_fridge': 0,
                'storage_tips': 'Information not available'
            }
        
        # Calculate usable days based on freshness percentage
        base_days_fresh = nutrition_info.get('storage_days_fresh', 7)
        base_days_ambient = nutrition_info.get('storage_days_ambient', 3)
        base_days_fridge = nutrition_info.get('storage_days_fridge', 10)
        
        # Adjust days based on freshness (linear scaling)
        freshness_factor = freshness_percentage / 100.0
        
        days_ambient = max(0, int(base_days_ambient * freshness_factor))
        days_fridge = max(0, int(base_days_fridge * freshness_factor))
        
        storage_tips = nutrition_info.get('storage_tips', 'Store in cool, dry place.')
        
        return {
            'days_ambient': days_ambient,
            'days_fridge': days_fridge,
            'storage_tips': storage_tips,
            'base_days_fresh': base_days_fresh,
            'base_days_ambient': base_days_ambient,
            'base_days_fridge': base_days_fridge
        }
    
    def get_calories(self, item_name: str, portion_weight_g: Optional[float] = None) -> Dict:
        """
        Get calorie information.
        
        Args:
            item_name: Name of the fruit/vegetable
            portion_weight_g: Weight in grams (optional, uses per_item if not provided)
            
        Returns:
            Dictionary with calorie information
        """
        nutrition_info = self.get_nutrition_info(item_name)
        
        if not nutrition_info:
            return {
                'calories': 0,
                'calories_per_100g': 0,
                'calories_per_item': 0,
                'portion_weight_g': portion_weight_g or 0
            }
        
        calories_per_100g = nutrition_info.get('calories_per_100g', 0)
        calories_per_item = nutrition_info.get('calories_per_item', 0)
        
        if portion_weight_g:
            calories = (calories_per_100g / 100.0) * portion_weight_g
        else:
            calories = calories_per_item
        
        return {
            'calories': round(calories, 1),
            'calories_per_100g': calories_per_100g,
            'calories_per_item': calories_per_item,
            'portion_weight_g': portion_weight_g or 0
        }
    
    def get_benefits(self, item_name: str) -> List[str]:
        """
        Get health benefits for an item.
        
        Args:
            item_name: Name of the fruit/vegetable
            
        Returns:
            List of benefit strings
        """
        nutrition_info = self.get_nutrition_info(item_name)
        
        if not nutrition_info:
            return ['Nutrition information not available']
        
        return nutrition_info.get('benefits', ['No specific benefits listed'])
    
    def get_recipes(self, item_name: str) -> List[str]:
        """
        Get recipe suggestions for an item.
        
        Args:
            item_name: Name of the fruit/vegetable
            
        Returns:
            List of recipe names
        """
        if 'indian_recipes' not in self.storage_tips:
            return []
        
        recipes = self.storage_tips.get('indian_recipes', {})
        
        # Try exact match
        if item_name in recipes:
            return recipes[item_name]
        
        # Try case-insensitive match
        item_name_lower = item_name.lower().replace(' ', '_')
        if item_name_lower in recipes:
            return recipes[item_name_lower]
        
        return []
    
    def get_all_items(self) -> List[str]:
        """Get list of all available items in database."""
        return list(self.nutrition_db.keys())


