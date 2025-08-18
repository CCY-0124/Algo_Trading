"""
strategy_manager.py

Simple and user-friendly strategy recording system for newbies.
Makes it easy to save, load, and manage trading strategies.

Features:
- Simple save/load interface
- Strategy listing and management
- Automatic parameter validation
- User-friendly error messages
"""

import json
import os
from datetime import datetime
from typing import Dict, List, Optional, Any
import logging

def ensure_reports_directory():
    """Ensure the reports directory exists"""
    reports_dir = "reports"
    os.makedirs(reports_dir, exist_ok=True)
    return reports_dir

class StrategyManager:
    """
    Simple strategy manager for newbies to easily save and load strategies.
    
    :param storage_dir: Directory to store strategy files
    """
    
    def __init__(self, storage_dir: str = "config/optimized_params"):
        """
        Initialize the strategy manager.
        
        :param storage_dir: Directory where strategies are stored
        """
        self.storage_dir = storage_dir
        self._ensure_storage_dir()
        self._setup_logging()
    
    def _setup_logging(self):
        """Setup simple logging for user feedback"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
    
    def _ensure_storage_dir(self):
        """Create storage directory if it doesn't exist"""
        os.makedirs(self.storage_dir, exist_ok=True)
    
    def save_strategy(
        self, 
        name: str, 
        parameters: Dict, 
        description: str = "",
        asset: str = "BTC",
        strategy_type: str = "custom"
    ) -> bool:
        """
        Save a strategy with a simple name and description.
        
        :param name: Simple name for the strategy (e.g., "My BTC Strategy")
        :param parameters: Strategy parameters dictionary
        :param description: Optional description of what the strategy does
        :param asset: Asset this strategy is designed for
        :param strategy_type: Type of strategy (e.g., "momentum", "mean_reversion")
        :return: True if saved successfully, False otherwise
        """
        try:
            # Create strategy data
            strategy_data = {
                "name": name,
                "description": description,
                "asset": asset,
                "strategy_type": strategy_type,
                "parameters": parameters,
                "created_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            
            # Create filename from name
            filename = self._name_to_filename(name)
            filepath = os.path.join(self.storage_dir, filename)
            
            # Save to file
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(strategy_data, f, indent=2, ensure_ascii=False)
            
            logging.info(f"Strategy '{name}' saved successfully!")
            return True
            
        except Exception as e:
            logging.error(f"Failed to save strategy '{name}': {str(e)}")
            return False
    
    def load_strategy(self, name: str) -> Optional[Dict]:
        """
        Load a strategy by name.
        
        :param name: Name of the strategy to load
        :return: Strategy data dictionary or None if not found
        """
        try:
            filename = self._name_to_filename(name)
            filepath = os.path.join(self.storage_dir, filename)
            
            if not os.path.exists(filepath):
                logging.error(f"Strategy '{name}' not found!")
                return None
            
            with open(filepath, 'r', encoding='utf-8') as f:
                strategy_data = json.load(f)
            
            logging.info(f"Strategy '{name}' loaded successfully!")
            return strategy_data
            
        except Exception as e:
            logging.error(f"Failed to load strategy '{name}': {str(e)}")
            return None
    
    def list_strategies(self) -> List[Dict]:
        """
        List all saved strategies with basic information.
        
        :return: List of strategy information dictionaries
        """
        strategies = []
        
        try:
            for filename in os.listdir(self.storage_dir):
                if filename.endswith('.json'):
                    filepath = os.path.join(self.storage_dir, filename)
                    
                    with open(filepath, 'r', encoding='utf-8') as f:
                        strategy_data = json.load(f)
                    
                    # Extract basic info for listing
                    strategies.append({
                        "name": strategy_data.get("name", "Unknown"),
                        "description": strategy_data.get("description", ""),
                        "asset": strategy_data.get("asset", "Unknown"),
                        "strategy_type": strategy_data.get("strategy_type", "custom"),
                        "created_date": strategy_data.get("created_date", ""),
                        "last_updated": strategy_data.get("last_updated", "")
                    })
            
            # Sort by last updated date
            strategies.sort(key=lambda x: x.get("last_updated", ""), reverse=True)
            
        except Exception as e:
            logging.error(f"Failed to list strategies: {str(e)}")
        
        return strategies
    
    def delete_strategy(self, name: str) -> bool:
        """
        Delete a strategy by name.
        
        :param name: Name of the strategy to delete
        :return: True if deleted successfully, False otherwise
        """
        try:
            filename = self._name_to_filename(name)
            filepath = os.path.join(self.storage_dir, filename)
            
            if not os.path.exists(filepath):
                logging.error(f"Strategy '{name}' not found!")
                return False
            
            os.remove(filepath)
            logging.info(f"Strategy '{name}' deleted successfully!")
            return True
            
        except Exception as e:
            logging.error(f"Failed to delete strategy '{name}': {str(e)}")
            return False
    
    def update_strategy(
        self, 
        name: str, 
        new_parameters: Dict = None,
        new_description: str = None
    ) -> bool:
        """
        Update an existing strategy.
        
        :param name: Name of the strategy to update
        :param new_parameters: New parameters (optional)
        :param new_description: New description (optional)
        :return: True if updated successfully, False otherwise
        """
        try:
            # Load existing strategy
            strategy_data = self.load_strategy(name)
            if not strategy_data:
                return False
            
            # Update fields if provided
            if new_parameters:
                strategy_data["parameters"] = new_parameters
            if new_description:
                strategy_data["description"] = new_description
            
            strategy_data["last_updated"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            # Save updated strategy
            filename = self._name_to_filename(name)
            filepath = os.path.join(self.storage_dir, filename)
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(strategy_data, f, indent=2, ensure_ascii=False)
            
            logging.info(f"Strategy '{name}' updated successfully!")
            return True
            
        except Exception as e:
            logging.error(f"Failed to update strategy '{name}': {str(e)}")
            return False
    
    def _name_to_filename(self, name: str) -> str:
        """
        Convert strategy name to filename.
        
        :param name: Strategy name
        :return: Safe filename
        """
        # Replace spaces and special characters
        safe_name = name.replace(" ", "_").replace("/", "_").replace("\\", "_")
        safe_name = "".join(c for c in safe_name if c.isalnum() or c in "_-")
        return f"{safe_name}.json"
    
    def print_strategy_list(self):
        """Print a user-friendly list of all strategies"""
        strategies = self.list_strategies()
        
        if not strategies:
            print("No strategies found. Use save_strategy() to create your first strategy!")
            return
        
        print(f"\nFound {len(strategies)} saved strategies:")
        print("=" * 80)
        
        for i, strategy in enumerate(strategies, 1):
            print(f"{i}. {strategy['name']}")
            print(f"   Asset: {strategy['asset']}")
            print(f"   Type: {strategy['strategy_type']}")
            if strategy['description']:
                print(f"   Description: {strategy['description']}")
            print(f"   Created: {strategy['created_date']}")
            print(f"   Updated: {strategy['last_updated']}")
            print("-" * 40)
    
    def get_strategy_parameters(self, name: str) -> Optional[Dict]:
        """
        Get just the parameters from a strategy.
        
        :param name: Strategy name
        :return: Parameters dictionary or None
        """
        strategy_data = self.load_strategy(name)
        if strategy_data:
            return strategy_data.get("parameters", {})
        return None


# Example usage functions for newbies
def quick_save_strategy(name: str, parameters: Dict, description: str = ""):
    """
    Quick function to save a strategy - perfect for newbies!
    
    :param name: Strategy name (e.g., "My BTC Strategy")
    :param parameters: Strategy parameters
    :param description: What this strategy does
    """
    manager = StrategyManager()
    success = manager.save_strategy(name, parameters, description)
    
    if success:
        print(f"Strategy '{name}' saved successfully!")
        print("Use list_strategies() to see all your strategies")
    else:
        print(f"Failed to save strategy '{name}'")


def quick_load_strategy(name: str) -> Optional[Dict]:
    """
    Quick function to load a strategy - perfect for newbies!
    
    :param name: Strategy name
    :return: Strategy parameters or None
    """
    manager = StrategyManager()
    strategy_data = manager.load_strategy(name)
    
    if strategy_data:
        print(f"Loaded strategy '{name}'")
        print(f"Parameters: {strategy_data['parameters']}")
        return strategy_data['parameters']
    else:
        print(f"Strategy '{name}' not found")
        return None


def list_all_strategies():
    """Quick function to list all strategies - perfect for newbies!"""
    manager = StrategyManager()
    manager.print_strategy_list()


if __name__ == "__main__":
    # Example usage for newbies
    print("Strategy Manager - Simple Strategy Recording System")
    print("=" * 50)
    
    # Example: Save a strategy
    example_params = {
        "rolling_window": 20,
        "long_threshold": 0.05,
        "short_threshold": -0.05,
        "lot_size": 0.001
    }
    
    quick_save_strategy(
        "My First BTC Strategy", 
        example_params, 
        "A simple momentum strategy for Bitcoin"
    )
    
    # List all strategies
    list_all_strategies() 