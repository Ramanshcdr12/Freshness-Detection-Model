"""
Local storage module for scan history.
Uses JSON file for persistence.
"""

import json
import os
from datetime import datetime
from typing import List, Dict, Optional
import base64


class HistoryStorage:
    """Manages scan history storage in local JSON file."""
    
    def __init__(self, history_file: str = "data/scan_history.json"):
        """
        Initialize history storage.
        
        Args:
            history_file: Path to history JSON file
        """
        self.history_file = history_file
        self._ensure_file_exists()
    
    def _ensure_file_exists(self):
        """Ensure history file exists."""
        os.makedirs(os.path.dirname(self.history_file), exist_ok=True)
        if not os.path.exists(self.history_file):
            with open(self.history_file, 'w', encoding='utf-8') as f:
                json.dump([], f)
    
    def save_scan(self, scan_data: Dict) -> bool:
        """
        Save a scan to history.
        
        Args:
            scan_data: Dictionary containing scan information
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Load existing history
            history = self.load_all_scans()
            
            # Add timestamp if not present
            if 'timestamp' not in scan_data:
                scan_data['timestamp'] = datetime.now().isoformat()
            
            # Add ID if not present
            if 'id' not in scan_data:
                scan_data['id'] = len(history) + 1
            
            # Add to history
            history.append(scan_data)
            
            # Save to file
            with open(self.history_file, 'w', encoding='utf-8') as f:
                json.dump(history, f, indent=2, ensure_ascii=False)
            
            return True
        except Exception as e:
            print(f"Error saving scan: {e}")
            return False
    
    def load_all_scans(self) -> List[Dict]:
        """
        Load all scans from history.
        
        Returns:
            List of scan dictionaries
        """
        try:
            if os.path.exists(self.history_file):
                with open(self.history_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            return []
        except Exception as e:
            print(f"Error loading scans: {e}")
            return []
    
    def get_scan(self, scan_id: int) -> Optional[Dict]:
        """
        Get a specific scan by ID.
        
        Args:
            scan_id: ID of the scan
            
        Returns:
            Scan dictionary or None if not found
        """
        history = self.load_all_scans()
        for scan in history:
            if scan.get('id') == scan_id:
                return scan
        return None
    
    def delete_scan(self, scan_id: int) -> bool:
        """
        Delete a scan from history.
        
        Args:
            scan_id: ID of the scan to delete
            
        Returns:
            True if successful, False otherwise
        """
        try:
            history = self.load_all_scans()
            history = [scan for scan in history if scan.get('id') != scan_id]
            
            with open(self.history_file, 'w', encoding='utf-8') as f:
                json.dump(history, f, indent=2, ensure_ascii=False)
            
            return True
        except Exception as e:
            print(f"Error deleting scan: {e}")
            return False
    
    def clear_history(self) -> bool:
        """
        Clear all scan history.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            with open(self.history_file, 'w', encoding='utf-8') as f:
                json.dump([], f)
            return True
        except Exception as e:
            print(f"Error clearing history: {e}")
            return False
    
    def get_recent_scans(self, limit: int = 10) -> List[Dict]:
        """
        Get most recent scans.
        
        Args:
            limit: Maximum number of scans to return
            
        Returns:
            List of recent scan dictionaries
        """
        history = self.load_all_scans()
        # Sort by timestamp (most recent first)
        history.sort(key=lambda x: x.get('timestamp', ''), reverse=True)
        return history[:limit]
    
    def get_scans_by_item(self, item_name: str) -> List[Dict]:
        """
        Get all scans for a specific item.
        
        Args:
            item_name: Name of the fruit/vegetable
            
        Returns:
            List of scan dictionaries for that item
        """
        history = self.load_all_scans()
        return [scan for scan in history if scan.get('item_name', '').lower() == item_name.lower()]


