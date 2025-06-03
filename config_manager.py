import yaml
import os
import logging
from typing import Dict, Any
from datetime import datetime

class ConfigManager:
    """Manages configuration loading and validation."""
    
    def __init__(self, config_path: str = "config.yaml"):
        self.config_path = config_path
        self.config = None
        self.logger = logging.getLogger(__name__)
        
    def load_config(self) -> Dict[str, Any]:
        """Load and validate