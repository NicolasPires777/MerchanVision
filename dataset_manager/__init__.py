"""
Dataset Manager Package

Provides utilities for creating, loading, and validating video datasets
for machine learning classification tasks.

Key functions:
- create_sample_dataset: Create sample dataset structure
- load_video_dataset: Load and organize video files by class  
- validate_dataset: Check if dataset is ready for training
- get_dataset_statistics: Get detailed dataset statistics
- detect_dataset_classes: Detect existing classes in dataset
- get_dataset_basic_info: Get basic dataset information
"""

from .dataset_creator import create_sample_dataset
from .dataset_lister import (
    load_video_dataset, 
    get_dataset_basic_info,
    detect_dataset_classes
)
from .dataset_validator import (
    validate_dataset,
    get_dataset_statistics
)

__all__ = [
    'create_sample_dataset',
    'load_video_dataset',
    'validate_dataset', 
    'get_dataset_statistics',
    'get_dataset_basic_info',
    'detect_dataset_classes'
]