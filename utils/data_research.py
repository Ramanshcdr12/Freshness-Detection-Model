"""
Utility script for researching and compiling nutrition data.
This script can be used to update the nutrition database.
"""

import json
import os
from typing import Dict, List


def add_item_to_database(item_data: Dict, db_path: str = "data/nutrition_db.json"):
    """
    Add or update an item in the nutrition database.
    
    Args:
        item_data: Dictionary with item information
        db_path: Path to nutrition database
    """
    # Load existing database
    if os.path.exists(db_path):
        with open(db_path, 'r', encoding='utf-8') as f:
            db = json.load(f)
    else:
        db = {}
    
    # Get item key (use name in lowercase with underscores)
    item_key = item_data.get('name', '').lower().replace(' ', '_')
    
    # Add/update item
    db[item_key] = item_data
    
    # Save database
    with open(db_path, 'w', encoding='utf-8') as f:
        json.dump(db, f, indent=2, ensure_ascii=False)
    
    print(f"Added/updated item: {item_key}")


def validate_database(db_path: str = "data/nutrition_db.json") -> List[str]:
    """
    Validate nutrition database structure.
    
    Args:
        db_path: Path to nutrition database
        
    Returns:
        List of validation errors (empty if valid)
    """
    errors = []
    
    if not os.path.exists(db_path):
        return [f"Database file not found: {db_path}"]
    
    with open(db_path, 'r', encoding='utf-8') as f:
        db = json.load(f)
    
    required_fields = [
        'name', 'calories_per_100g', 'vitamins', 'benefits',
        'storage_days_fresh', 'storage_days_ambient', 'storage_days_fridge'
    ]
    
    for item_key, item_data in db.items():
        for field in required_fields:
            if field not in item_data:
                errors.append(f"{item_key}: Missing field '{field}'")
    
    return errors


if __name__ == "__main__":
    # Example usage
    print("Nutrition Database Research Utility")
    print("Use this script to add or update items in the nutrition database.")
    
    # Validate database
    errors = validate_database()
    if errors:
        print("\nValidation errors found:")
        for error in errors:
            print(f"  - {error}")
    else:
        print("\nDatabase validation passed!")


