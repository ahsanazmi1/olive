"""
Policy DSL loader for YAML and JSON formats.

This module provides functionality to load policies from YAML and JSON files.
"""

import json
import logging
import yaml
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from .models import PolicyDSL, PolicyStorage

logger = logging.getLogger(__name__)


class PolicyLoader:
    """Policy loader for YAML and JSON formats."""
    
    def __init__(self, storage: Optional[PolicyStorage] = None):
        self.storage = storage or PolicyStorage()
    
    def load_policy_from_dict(self, policy_data: Dict[str, Any]) -> PolicyDSL:
        """
        Load a policy from a dictionary.
        
        Args:
            policy_data: Policy data dictionary
            
        Returns:
            PolicyDSL object
            
        Raises:
            ValueError: If policy data is invalid
        """
        try:
            # Validate required fields
            required_fields = ["policy_id", "policy_name"]
            for field in required_fields:
                if field not in policy_data:
                    raise ValueError(f"Missing required field: {field}")
            
            # Convert rail type string to enum if present
            if "prefer_rail" in policy_data and policy_data["prefer_rail"]:
                rail_str = policy_data["prefer_rail"].lower()
                if rail_str in ["ach", "debit", "credit"]:
                    policy_data["prefer_rail"] = rail_str
            
            return PolicyDSL(**policy_data)
            
        except Exception as e:
            raise ValueError(f"Invalid policy data: {e}")
    
    def load_policies_from_yaml(self, yaml_content: str) -> List[PolicyDSL]:
        """
        Load policies from YAML content.
        
        Args:
            yaml_content: YAML content string
            
        Returns:
            List of PolicyDSL objects
            
        Raises:
            ValueError: If YAML content is invalid
        """
        try:
            data = yaml.safe_load(yaml_content)
            
            if not isinstance(data, list):
                raise ValueError("YAML content must contain a list of policies")
            
            policies = []
            for policy_data in data:
                if not isinstance(policy_data, dict):
                    raise ValueError("Each policy must be a dictionary")
                
                policy = self.load_policy_from_dict(policy_data)
                policies.append(policy)
            
            return policies
            
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML format: {e}")
        except Exception as e:
            raise ValueError(f"Error loading policies from YAML: {e}")
    
    def load_policies_from_json(self, json_content: str) -> List[PolicyDSL]:
        """
        Load policies from JSON content.
        
        Args:
            json_content: JSON content string
            
        Returns:
            List of PolicyDSL objects
            
        Raises:
            ValueError: If JSON content is invalid
        """
        try:
            data = json.loads(json_content)
            
            if not isinstance(data, list):
                raise ValueError("JSON content must contain a list of policies")
            
            policies = []
            for policy_data in data:
                if not isinstance(policy_data, dict):
                    raise ValueError("Each policy must be a dictionary")
                
                policy = self.load_policy_from_dict(policy_data)
                policies.append(policy)
            
            return policies
            
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON format: {e}")
        except Exception as e:
            raise ValueError(f"Error loading policies from JSON: {e}")
    
    def load_policies_from_file(self, file_path: Union[str, Path]) -> List[PolicyDSL]:
        """
        Load policies from a file (YAML or JSON).
        
        Args:
            file_path: Path to the policy file
            
        Returns:
            List of PolicyDSL objects
            
        Raises:
            ValueError: If file format is unsupported or invalid
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise ValueError(f"Policy file not found: {file_path}")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Determine format by file extension
        if file_path.suffix.lower() in ['.yaml', '.yml']:
            return self.load_policies_from_yaml(content)
        elif file_path.suffix.lower() == '.json':
            return self.load_policies_from_json(content)
        else:
            raise ValueError(f"Unsupported file format: {file_path.suffix}")
    
    def store_policies(self, policies: List[PolicyDSL]) -> None:
        """
        Store policies in the storage backend.
        
        Args:
            policies: List of policies to store
        """
        for policy in policies:
            self.storage.store_policy(policy)
            logger.info(f"Stored policy: {policy.policy_id} ({policy.policy_name})")
    
    def load_and_store_policies_from_file(self, file_path: Union[str, Path]) -> List[PolicyDSL]:
        """
        Load policies from file and store them.
        
        Args:
            file_path: Path to the policy file
            
        Returns:
            List of loaded PolicyDSL objects
        """
        policies = self.load_policies_from_file(file_path)
        self.store_policies(policies)
        return policies


# Global policy loader instance
_policy_loader: Optional[PolicyLoader] = None


def get_policy_loader() -> PolicyLoader:
    """Get the global policy loader instance."""
    global _policy_loader
    if _policy_loader is None:
        _policy_loader = PolicyLoader()
    return _policy_loader


def load_policies_from_yaml(yaml_content: str) -> List[PolicyDSL]:
    """
    Load policies from YAML content.
    
    Args:
        yaml_content: YAML content string
        
    Returns:
        List of PolicyDSL objects
    """
    loader = get_policy_loader()
    return loader.load_policies_from_yaml(yaml_content)


def load_policies_from_json(json_content: str) -> List[PolicyDSL]:
    """
    Load policies from JSON content.
    
    Args:
        json_content: JSON content string
        
    Returns:
        List of PolicyDSL objects
    """
    loader = get_policy_loader()
    return loader.load_policies_from_json(json_content)
