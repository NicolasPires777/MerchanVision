#!/usr/bin/env python3
"""
🔧 Configurações da IA via Variáveis de Ambiente
Gerencia todas as configurações de treinamento e tempo real
"""

import os
from pathlib import Path
import json
from typing import Dict, Any, Union

class AIConfig:
    """Gerenciador de configurações da IA via .env"""
    
    def __init__(self, env_file=None):
        """Carrega configurações do arquivo .env"""
        if env_file is None:
            # Procurar .env no diretório raiz do projeto
            current_dir = Path(__file__).parent.parent
            env_file = current_dir / ".env"
        
        self.env_file = Path(env_file)
        self.config = {}
        self.load_env_file()
        
    def load_env_file(self):
        """Carrega variáveis do arquivo .env"""
        if not self.env_file.exists():
            print(f"⚠️ Arquivo .env não encontrado: {self.env_file}")
            print("💡 Usando configurações padrão")
            return
            
        with open(self.env_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                
                # Pular comentários e linhas vazias
                if not line or line.startswith('#'):
                    continue
                    
                # Processar linha KEY=VALUE
                if '=' in line:
                    key, value = line.split('=', 1)
                    key = key.strip()
                    value = value.strip()
                    
                    # Remover aspas se existirem
                    if value.startswith('"') and value.endswith('"'):
                        value = value[1:-1]
                    elif value.startswith("'") and value.endswith("'"):
                        value = value[1:-1]
                    
                    # Converter tipos
                    value = self._convert_value(value)
                    self.config[key] = value
                    
        print(f"✅ Configurações carregadas de: {self.env_file}")
        print(f"📊 {len(self.config)} parâmetros configurados")
    
    def _convert_value(self, value: str) -> Union[str, int, float, bool, list]:
        """Converte string do .env para tipo apropriado"""
        # Boolean
        if value.lower() in ('true', 'false'):
            return value.lower() == 'true'
        
        # Integer
        try:
            if '.' not in value and value.isdigit():
                return int(value)
        except ValueError:
            pass
        
        # Float  
        try:
            if '.' in value:
                return float(value)
        except ValueError:
            pass
        
        # List (comma separated)
        if ',' in value:
            return [item.strip() for item in value.split(',')]
            
        # String
        return value
    
    def get(self, key: str, default: Any = None) -> Any:
        """Obtém configuração com fallback para padrão"""
        return self.config.get(key, default)
    
    def get_classifier_config(self) -> Dict[str, Any]:
        """Retorna configuração do classificador"""
        config = {
            'classifier_type': self.get('CLASSIFIER_TYPE', 'rf'),
            'random_state': self.get('RANDOM_STATE', 42)
        }
        
        # Configuração geral
        config.update({
            'n_estimators': self.get('RF_N_ESTIMATORS', 180),
            'max_depth': self.get('RF_MAX_DEPTH', 14),
            'min_samples_split': self.get('RF_MIN_SAMPLES_SPLIT', 4),
            'min_samples_leaf': self.get('RF_MIN_SAMPLES_LEAF', 2),
        })
        
        # Balanceamento de classes
        class_weight = self.get('RF_CLASS_WEIGHT', 'balanced')
        if class_weight != 'None':
            config['class_weight'] = class_weight
            
        return config
    
    def get_video_config(self) -> Dict[str, Any]:
        """Retorna configuração de processamento de vídeo"""
        return {
            'frames_per_video': self.get('FRAMES_PER_VIDEO', 5),
            'max_duration': self.get('MAX_VIDEO_DURATION', 45),
            'min_duration': self.get('MIN_VIDEO_DURATION', 8),
            'resize_width': self.get('RESIZE_WIDTH', 224),
            'resize_height': self.get('RESIZE_HEIGHT', 224),
            'frame_interval': self.get('FRAME_INTERVAL', 'auto')
        }
    
    def get_realtime_config(self) -> Dict[str, Any]:
        """Retorna configuração para tempo real"""
        return {
            'window_seconds': self.get('REALTIME_WINDOW_SECONDS', 3),
            'classification_frequency': self.get('CLASSIFICATION_FREQUENCY', 0.2),
            'min_frames_classify': self.get('MIN_FRAMES_CLASSIFY', 2),
            'target_fps': self.get('TARGET_FPS', 30),
            'frame_buffer_size': self.get('FRAME_BUFFER_SIZE', 20),
            'min_confidence_threshold': self.get('MIN_CONFIDENCE_THRESHOLD', 0.6),
            'prediction_smoothing': self.get('PREDICTION_SMOOTHING', True),
            'smoothing_window': self.get('SMOOTHING_WINDOW', 3)
        }
    
    def get_validation_config(self) -> Dict[str, Any]:
        """Retorna configuração para validação"""
        return {
            'max_videos_validation': self.get('MAX_VIDEOS_VALIDATION', 50),
            'train_test_split': self.get('TRAIN_TEST_SPLIT', 0.8),
            'cross_validation_folds': self.get('CROSS_VALIDATION_FOLDS', 5)
        }
    
    def get_technical_config(self) -> Dict[str, Any]:
        """Retorna configuração técnica"""
        return {
            'use_gpu': self.get('USE_GPU', 'auto'),
            'num_threads': self.get('NUM_THREADS', 'auto'),
            'log_level': self.get('LOG_LEVEL', 'info'),
            'cache_features': self.get('CACHE_FEATURES', True),
            'memory_optimization': self.get('MEMORY_OPTIMIZATION', True)
        }
    
    def get_classes(self) -> list:
        """Retorna lista de classes configuradas"""
        classes = self.get('DEFAULT_CLASSES', ['conteudo', 'merchan'])
        if isinstance(classes, str):
            classes = [c.strip() for c in classes.split(',')]
        return classes
    
    def print_summary(self):
        """Imprime resumo das configurações ativas"""
        print(f"\n🔧 === CONFIGURAÇÕES ATIVAS ===")
            
        print(f"\n🤖 Classificador:")
        clf_config = self.get_classifier_config()
        for key, value in clf_config.items():
            print(f"   {key}: {value}")
            
        print(f"\n🎬 Processamento de Vídeo:")
        video_config = self.get_video_config()
        for key, value in video_config.items():
            print(f"   {key}: {value}")
            
        print(f"\n🚀 Tempo Real:")
        rt_config = self.get_realtime_config()
        for key, value in rt_config.items():
            print(f"   {key}: {value}")
            
        print(f"\n🎯 Classes: {', '.join(self.get_classes())}")

# Instância global para fácil importação
config = AIConfig()

if __name__ == "__main__":
    # Teste das configurações
    config.print_summary()