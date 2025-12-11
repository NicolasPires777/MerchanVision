"""
🔧 Feature Extractor - Extração de Features CNN de Vídeos

Responsabilidade única: Extrair features de vídeos usando EfficientNetB0
Usado por classificadores mais complexos para obter representações vetoriais.
"""

import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
from config import config


class VideoFeatureExtractor:
    """Extrator de features de vídeo usando CNN pré-treinada"""
    
    def __init__(self):
        """Inicializa o extrator de features"""
        self.feature_extractor = None
        
        # Carregar configurações
        try:
            self.video_config = config.get_video_config()
            print(f"✅ Configurações carregadas do .env")
        except:
            # Configurações padrão se não há config
            self.video_config = {
                'frames_per_video': 5,
                'max_duration': 45,
                'min_duration': 8,
                'resize_width': 224,
                'resize_height': 224,
                'frame_interval': 'auto'
            }
            print(f"⚠️ Usando configurações padrão")
        
        self.setup_feature_extractor()
    
    def setup_feature_extractor(self):
        """Configura extrator de features usando EfficientNetB0"""
        print("🔧 Configurando extrator de features CNN...")
        
        # Usar tamanho configurado
        width = self.video_config['resize_width'] 
        height = self.video_config['resize_height']
        
        # EfficientNetB0 como base (leve e eficiente)
        base_model = keras.applications.EfficientNetB0(
            weights='imagenet',
            include_top=False,
            pooling='avg',
            input_shape=(height, width, 3)
        )
        
        self.feature_extractor = base_model
        print("✅ Extrator EfficientNetB0 configurado")
        print(f"📐 Entrada: {width}x{height}, Saída: {base_model.output_shape[1]} features")
    
    def extract_video_features(self, video_path, max_frames=None, sample_rate=None):
        """
        Extrai features agregadas de um vídeo
        
        Args:
            video_path (str): Caminho do vídeo
            max_frames (int): Máximo de frames (padrão: config)
            sample_rate (int): Frame interval (padrão: auto)
        
        Returns:
            np.array: Features agregadas [mean, max, min, std]
        """
        if max_frames is None:
            max_frames = self.video_config['frames_per_video']
        
        # Carregar frames do vídeo
        frames = self._load_video_frames(video_path, max_frames, sample_rate)
        
        if len(frames) == 0:
            print(f"❌ Nenhum frame válido em: {video_path}")
            return None
        
        # Extrair features dos frames
        features = self._extract_features_from_frames(frames)
        
        return features
    
    def extract_features_from_frames(self, frames_array):
        """
        Extrai features de frames já carregados (para tempo real)
        
        Args:
            frames_array (np.array): Array de frames [N, H, W, 3]
        
        Returns:
            np.array: Features agregadas
        """
        if len(frames_array) == 0:
            return None
        
        # Garantir formato correto
        if frames_array.dtype != np.float32:
            frames_array = frames_array.astype(np.float32) / 255.0
        
        return self._extract_features_from_frames(frames_array)
    
    def _load_video_frames(self, video_path, max_frames, sample_rate=None):
        """Carrega frames do vídeo com configurações"""
        cap = cv2.VideoCapture(video_path)
        frames = []
        frame_count = 0
        processed_frames = 0
        
        # Informações do vídeo
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        duration = total_frames / fps
        
        # Validar duração
        if duration < self.video_config['min_duration']:
            print(f"⚠️ Vídeo curto: {duration:.1f}s (mín: {self.video_config['min_duration']}s)")
        elif duration > self.video_config['max_duration']:
            print(f"⚠️ Vídeo longo: {duration:.1f}s (máx: {self.video_config['max_duration']}s)")
        
        # Calcular sample_rate automaticamente
        if sample_rate is None:
            if self.video_config['frame_interval'] == 'auto':
                sample_rate = max(1, total_frames // max_frames)
            else:
                sample_rate = self.video_config['frame_interval']
        
        # Processar frames
        while processed_frames < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_count % sample_rate == 0:
                # Preprocessar frame
                frame = self._preprocess_frame(frame)
                frames.append(frame)
                processed_frames += 1
            
            frame_count += 1
        
        cap.release()
        print(f"📹 Processados {len(frames)} frames de {video_path}")
        
        return np.array(frames) if frames else np.array([])
    
    def _preprocess_frame(self, frame):
        """Preprocessa frame individual"""
        # Redimensionar
        width = self.video_config['resize_width']
        height = self.video_config['resize_height'] 
        frame = cv2.resize(frame, (width, height))
        
        # Converter BGR -> RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Normalizar [0-1]
        frame = frame.astype(np.float32) / 255.0
        
        return frame
    
    def _extract_features_from_frames(self, frames_array):
        """Extrai features e agrega estatísticas"""
        # CNN features para cada frame
        features = self.feature_extractor.predict(frames_array, verbose=0)
        
        # Agregação estatística
        aggregated_features = np.concatenate([
            np.mean(features, axis=0),      # Média temporal
            np.max(features, axis=0),       # Máximos
            np.min(features, axis=0),       # Mínimos  
            np.std(features, axis=0)        # Variabilidade
        ])
        
        print(f"🧠 Features extraídas: {features.shape} -> {aggregated_features.shape}")
        
        return aggregated_features
    
    def get_feature_dimension(self):
        """Retorna dimensão das features agregadas"""
        if self.feature_extractor is None:
            return None
        
        base_features = self.feature_extractor.output_shape[1]
        # 4x devido à agregação (mean, max, min, std)
        return base_features * 4
    
    def get_config(self):
        """Retorna configurações ativas"""
        return {
            'video_config': self.video_config,
            'feature_dimension': self.get_feature_dimension(),
            'model': 'EfficientNetB0'
        }