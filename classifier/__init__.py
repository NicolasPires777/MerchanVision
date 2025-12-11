"""
Classifier Package - Classificação de Vídeos com IA

Responsabilidade: Classificação de vídeos com diferentes estratégias
Contém:
- feature_extractor: Extração de features CNN
- basic_classifier: Classificação básica CNN + ML
- hybrid_classifier: Classificação híbrida CNN + indicadores visuais  
- visual_elements_detector: Detecção de elementos visuais
- realtime_basic_classifier: Classificação básica em tempo real
- realtime_hybrid_classifier: Classificação híbrida em tempo real
"""

# Componentes principais
from .feature_extractor import VideoFeatureExtractor
from .basic_classifier import BasicVideoClassifier
from .hybrid_classifier import HybridVideoClassifier
from .visual_elements_detector import VisualElementsDetector

# Componentes realtime
from .realtime_basic_classifier import RealTimeBasicClassifier
from .realtime_hybrid_classifier import RealTimeHybridClassifier

__version__ = '3.0.0'
__all__ = [
    'VideoFeatureExtractor',
    'BasicVideoClassifier', 
    'HybridVideoClassifier',
    'VisualElementsDetector',
    'RealTimeBasicClassifier',
    'RealTimeHybridClassifier'
]