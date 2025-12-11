"""
Model Manager Package

Provides model training, validation, and diagnostic functionality for video classification.
"""

from .model_trainer import VideoModelTrainer
from .model_validator import ModelValidator, list_available_models

__all__ = [
    'VideoModelTrainer',
    'ModelValidator', 
    'list_available_models'
]

__version__ = '1.0.0'