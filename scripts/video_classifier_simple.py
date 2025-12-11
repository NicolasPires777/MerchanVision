"""
Classificador Flexível de Vídeo usando Modelos Pré-treinados

Este script usa transfer learning com modelos pré-treinados para
classificar vídeos em múltiplas categorias de forma rápida e eficiente.
Suporta: break, conteudo, merchan, e classes customizadas.
"""

import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import os
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report
import argparse
from pathlib import Path
import sys

# Import dataset manager functions
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dataset_manager import load_video_dataset, detect_dataset_classes


def load_dataset_and_extract_features(dataset_path, classifier_instance):
    """
    Carrega dataset usando dataset_manager e extrai features com classifier
    
    Args:
        dataset_path: Caminho para o dataset
        classifier_instance: Instância da classe SimpleVideoClassifier
        
    Returns:
        tuple: (X, y) arrays com features e labels
    """
    print(f"📁 Carregando dataset de: {dataset_path}")
    
    # Detectar classes que realmente existem no dataset
    actual_classes = detect_dataset_classes(dataset_path)
    
    # Usar apenas as classes que existem no dataset e também estão configuradas no classifier
    available_classes = [cls for cls in classifier_instance.classes if cls in actual_classes]
    
    if not available_classes:
        print(f"❌ Nenhuma classe válida encontrada no dataset")
        print(f"   Classes configuradas no classifier: {classifier_instance.classes}")
        print(f"   Classes encontradas no dataset: {actual_classes}")
        return np.array([]), np.array([])
    
    print(f"🎯 Usando classes: {available_classes}")
    
    # Usar dataset_manager para carregar informações do dataset
    dataset_info = load_video_dataset(dataset_path, available_classes, verbose=False)
    
    X, y = [], []
    
    for class_idx, class_name in enumerate(available_classes):
        video_files = dataset_info.get(class_name, [])
        
        if not video_files:
            print(f"⚠️ Nenhum vídeo encontrado para classe: {class_name}")
            continue
        
        print(f"🎬 Processando {len(video_files)} vídeos de '{class_name}'")
        
        for i, video_file in enumerate(video_files):
            print(f"  [{i+1}/{len(video_files)}] {os.path.basename(video_file)}")
            
            try:
                features = classifier_instance.extract_video_features(video_file)
                
                if features is not None:
                    X.append(features)
                    y.append(class_idx)
                    print(f"    ✅ Features extraídas: {features.shape}")
                else:
                    print(f"    ❌ Erro ao extrair features")
            
            except Exception as e:
                print(f"    ❌ Erro: {e}")
    
    X = np.array(X)
    y = np.array(y)
    
    print(f"\n📊 Dataset processado:")
    print(f"  🎬 Total de vídeos: {len(X)}")
    print(f"  📐 Dimensão das features: {X.shape[1] if len(X) > 0 else 0}")
    
    for i, class_name in enumerate(available_classes):
        count = np.sum(y == i)
        percentage = (count / len(y) * 100) if len(y) > 0 else 0
        print(f"  📋 {class_name}: {count} vídeos ({percentage:.1f}%)")
    
    return X, y
import time
import json

# Importar configurações
try:
    from ai_config import config
    CONFIG_AVAILABLE = True
except ImportError:
    print("⚠️ ai_config.py não encontrado, usando configurações padrão")
    CONFIG_AVAILABLE = False

class SimpleVideoClassifier:
    def __init__(self, classes=None):
        """
        Classificador flexível usando features pré-extraídas
        
        Args:
            classes: Lista de classes (padrão: do .env ou ['conteudo', 'merchan'])
        """
        self.feature_extractor = None
        self.classifier = None
        
        # Carregar configurações
        if CONFIG_AVAILABLE:
            self.video_config = config.get_video_config()
            self.classifier_config = config.get_classifier_config()
            
            # Classes do .env ou padrão
            if classes is None:
                self.classes = config.get_classes()
            else:
                self.classes = classes
        else:
            # Configurações padrão se não há .env
            self.video_config = {
                'frames_per_video': 5,
                'max_duration': 45,
                'min_duration': 8,
                'resize_width': 224,
                'resize_height': 224,
                'frame_interval': 'auto'
            }
            self.classifier_config = {
                'classifier_type': 'rf',
                'n_estimators': 180,
                'max_depth': 14,
                'min_samples_split': 4,
                'min_samples_leaf': 2,
                'class_weight': 'balanced',
                'random_state': 42
            }
            self.classes = classes if classes else ['conteudo', 'merchan']
            
        self.setup_feature_extractor()
        print(f"🎯 Classes configuradas: {self.classes}")
        
        # Imprimir configurações ativas
        if CONFIG_AVAILABLE:
            print(f"🔧 Frames por vídeo: {self.video_config['frames_per_video']}")
            print(f"📊 Estimators RF: {self.classifier_config['n_estimators']}")
            print(f"🌳 Profundidade máx: {self.classifier_config['max_depth']}")
    
    def setup_feature_extractor(self):
        """Configura extrator de features usando modelo pré-treinado"""
        print("🔧 Configurando extrator de features...")
        
        # Usar tamanho configurado
        width = self.video_config['resize_width']
        height = self.video_config['resize_height']
        
        # Usar EfficientNetB0 como base (leve e eficiente)
        base_model = keras.applications.EfficientNetB0(
            weights='imagenet',
            include_top=False,
            pooling='avg',
            input_shape=(height, width, 3)
        )
        
        self.feature_extractor = base_model
        print("✅ Extrator de features configurado")
    
    def extract_video_features(self, video_path, max_frames=None, sample_rate=None):
        """
        Extrai features de um vídeo usando configurações do .env
        
        Args:
            video_path: Caminho do vídeo
            max_frames: Máximo de frames a processar (padrão: do .env)
            sample_rate: Pegar 1 frame a cada N frames (padrão: do .env)
        
        Returns:
            Array com features agregadas do vídeo
        """
        # Usar configurações do .env
        if max_frames is None:
            max_frames = self.video_config['frames_per_video']
        
        cap = cv2.VideoCapture(video_path)
        frames = []
        frame_count = 0
        processed_frames = 0
        
        # Obter informações do vídeo
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        duration = total_frames / fps
        
        # Verificar duração
        if duration < self.video_config['min_duration']:
            print(f"⚠️ Vídeo muito curto: {duration:.1f}s (mín: {self.video_config['min_duration']}s)")
        elif duration > self.video_config['max_duration']:
            print(f"⚠️ Vídeo muito longo: {duration:.1f}s (máx: {self.video_config['max_duration']}s)")
        
        # Calcular sample_rate automaticamente se não especificado
        if sample_rate is None:
            if self.video_config['frame_interval'] == 'auto':
                # Distribuir frames uniformemente
                sample_rate = max(1, total_frames // max_frames)
            else:
                sample_rate = self.video_config['frame_interval']
        
        while processed_frames < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_count % sample_rate == 0:
                # Usar tamanho configurado
                width = self.video_config['resize_width']
                height = self.video_config['resize_height']
                frame = cv2.resize(frame, (width, height))
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = frame.astype(np.float32) / 255.0
                frames.append(frame)
                processed_frames += 1
            
            frame_count += 1
        
        cap.release()
        
        if len(frames) == 0:
            return None
        
        # Extrair features de todos os frames
        frames_array = np.array(frames)
        features = self.feature_extractor.predict(frames_array, verbose=0)
        
        # Agregar features (média, máximo, mínimo, desvio padrão)
        aggregated_features = np.concatenate([
            np.mean(features, axis=0),      # Média
            np.max(features, axis=0),       # Máximo
            np.min(features, axis=0),       # Mínimo
            np.std(features, axis=0)        # Desvio padrão
        ])
        
        return aggregated_features
    
    def extract_features_from_frames(self, frames_array):
        """
        Extrai features de frames já carregados (para uso em tempo real)
        
        Args:
            frames_array: Array de frames [N, height, width, 3]
        
        Returns:
            Array com features agregadas
        """
        if len(frames_array) == 0:
            return None
        
        # Extrair features
        features = self.feature_extractor.predict(frames_array, verbose=0)
        
        # Agregar features
        aggregated_features = np.concatenate([
            np.mean(features, axis=0),
            np.max(features, axis=0), 
            np.min(features, axis=0),
            np.std(features, axis=0)
        ])
        
        return aggregated_features
    
    def extract_features_from_frames(self, frames_array):
        """
        Extrai features de frames já carregados (para uso em tempo real)
        
        Args:
            frames_array: Array de frames preprocessados (N, H, W, C)
        
        Returns:
            Array com features agregadas
        """
        if len(frames_array) == 0:
            return None
        
        # Garantir formato correto
        if frames_array.dtype != np.float32:
            frames_array = frames_array.astype(np.float32) / 255.0
        
        # Extrair features
        features = self.feature_extractor.predict(frames_array, verbose=0)
        
        # Agregar features
        aggregated_features = np.concatenate([
            np.mean(features, axis=0),
            np.max(features, axis=0),
            np.min(features, axis=0),
            np.std(features, axis=0)
        ])
        
        return aggregated_features
    
    def train(self, X, y, classifier_type=None):
        """
        Treina classificador usando configurações do .env
        
        Args:
            X: Features dos vídeos
            y: Labels  
            classifier_type: 'rf' ou 'svm' (padrão: do .env)
        """
        # Usar tipo do .env se não especificado
        if classifier_type is None:
            classifier_type = self.classifier_config['classifier_type']
            
        print(f"🚀 Treinando classificador {classifier_type.upper()}...")
        print(f"⚙️ Usando configurações padrão")
        
        if classifier_type == 'rf':
            # Usar configurações do .env
            rf_params = {
                'n_estimators': self.classifier_config['n_estimators'],
                'max_depth': self.classifier_config.get('max_depth'),
                'min_samples_split': self.classifier_config['min_samples_split'],
                'min_samples_leaf': self.classifier_config['min_samples_leaf'],
                'random_state': self.classifier_config['random_state'],
                'n_jobs': -1
            }
            
            # Adicionar class_weight se configurado
            if 'class_weight' in self.classifier_config:
                rf_params['class_weight'] = self.classifier_config['class_weight']
                
            # Remover None values
            rf_params = {k: v for k, v in rf_params.items() if v is not None}
            
            print(f"📊 Parâmetros RF: {rf_params}")
            self.classifier = RandomForestClassifier(**rf_params)
            
        elif classifier_type == 'svm':
            self.classifier = SVC(
                kernel='rbf',
                probability=True,
                random_state=self.classifier_config['random_state'],
                class_weight=self.classifier_config.get('class_weight', 'balanced')
            )
        
        # Treinar
        start_time = time.time()
        self.classifier.fit(X, y)
        train_time = time.time() - start_time
        
        # Avaliar
        train_score = self.classifier.score(X, y)
        
        print(f"✅ Treinamento concluído!")
        print(f"  ⏱️ Tempo: {train_time:.2f}s")
        print(f"  📊 Acurácia (treino): {train_score:.4f}")
        
        return self.classifier
    
    def predict_video(self, video_path):
        """
        Classifica um vídeo
        
        Returns:
            tuple: (classe, confiança)
        """
        if self.classifier is None:
            raise ValueError("Modelo não foi treinado")
        
        print(f"🎬 Analisando: {os.path.basename(video_path)}")
        
        # Extrair features
        features = self.extract_video_features(video_path)
        
        if features is None:
            return "erro", 0.0
        
        # Predição
        features_reshaped = features.reshape(1, -1)
        prediction = self.classifier.predict(features_reshaped)[0]
        probabilities = self.classifier.predict_proba(features_reshaped)[0]
        
        predicted_class = self.classes[prediction]
        confidence = probabilities[prediction]
        
        print(f"🎯 Resultado:")
        print(f"  📋 Classe: {predicted_class}")
        print(f"  💯 Confiança: {confidence:.4f}")
        print(f"  📊 Probabilidades:")
        for i, class_name in enumerate(self.classes):
            print(f"    {class_name}: {probabilities[i]:.4f}")
        
        return predicted_class, confidence
    
    def batch_predict(self, videos_directory):
        """Prediz múltiplos vídeos de uma pasta"""
        if self.classifier is None:
            raise ValueError("Modelo não foi treinado")
        
        video_extensions = ['*.mp4', '*.avi', '*.mov', '*.mkv']
        video_files = []
        for ext in video_extensions:
            video_files.extend(Path(videos_directory).glob(ext))
        
        print(f"🎬 Processando {len(video_files)} vídeos...")
        
        results = {}
        
        for i, video_file in enumerate(video_files):
            print(f"\n[{i+1}/{len(video_files)}] {video_file.name}")
            
            try:
                predicted_class, confidence = self.predict_video(str(video_file))
                results[video_file.name] = {
                    'class': predicted_class,
                    'confidence': confidence
                }
            except Exception as e:
                print(f"❌ Erro: {e}")
                results[video_file.name] = {
                    'class': 'erro',
                    'confidence': 0.0
                }
        
        # Resumo
        print(f"\n📊 Resumo dos resultados:")
        break_count = sum(1 for r in results.values() if r['class'] == 'break')
        content_count = sum(1 for r in results.values() if r['class'] == 'conteudo')
        error_count = sum(1 for r in results.values() if r['class'] == 'erro')
        
        print(f"  📺 Break: {break_count}")
        print(f"  🎬 Conteúdo: {content_count}")
        print(f"  ❌ Erros: {error_count}")
        
        return results
    
    def save_model(self, save_path):
        """Salva o modelo"""
        if self.classifier is None:
            raise ValueError("Nenhum modelo para salvar")
        
        os.makedirs(save_path, exist_ok=True)
        
        # Salvar classificador
        model_path = os.path.join(save_path, "classifier.pkl")
        joblib.dump(self.classifier, model_path)
        
        # Salvar configurações
        config = {
            'classes': self.classes,
            'feature_shape': None  # Será preenchido durante o uso
        }
        
        import json
        config_path = os.path.join(save_path, "config.json")
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"💾 Modelo salvo em: {save_path}")
        print(f"🎯 Classes salvas: {self.classes}")
    
    def load_model(self, load_path):
        """Carrega modelo salvo com suas classes"""
        # Se o caminho termina com .pkl, é um arquivo único
        if load_path.endswith('.pkl'):
            base_path = os.path.dirname(load_path)
            model_file = load_path
        else:
            # É um diretório
            base_path = load_path
            model_file = os.path.join(load_path, "classifier.pkl")
        
        # Verificar se o arquivo existe
        if not os.path.exists(model_file):
            raise FileNotFoundError(f"Modelo não encontrado: {model_file}")
        
        # Carregar configurações primeiro
        config_path = os.path.join(base_path, "config.json")
        if os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    config = json.load(f)
                    self.classes = config.get('classes', ['break', 'conteudo', 'merchan'])
                    print(f"🎯 Classes carregadas: {self.classes}")
            except Exception as e:
                print(f"⚠️ Erro ao carregar config: {e}")
                print("⚠️ Usando classes padrão")
                self.classes = ['break', 'conteudo', 'merchan']
        else:
            print("⚠️ Configuração não encontrada, usando classes padrão")
            self.classes = ['break', 'conteudo', 'merchan']
        
        # Carregar classificador
        try:
            self.classifier = joblib.load(model_file)
            print(f"✅ Modelo carregado de: {model_file}")
            return True
        except Exception as e:
            print(f"❌ Erro ao carregar modelo: {e}")
            return False

def main():
    """Interface de linha de comando"""
    parser = argparse.ArgumentParser(description="Classificador Flexível: Break vs Conteúdo vs Merchan")
    parser.add_argument('command', choices=['train', 'predict', 'batch', 'setup'], 
                       help='Comando a executar')
    parser.add_argument('--dataset', help='Caminho para dataset (train)')
    parser.add_argument('--video', help='Caminho para vídeo (predict)')
    parser.add_argument('--directory', help='Diretório de vídeos (batch)')
    parser.add_argument('--classifier', choices=['rf', 'svm'], 
                       help='Tipo de classificador (padrão: do .env)')
    parser.add_argument('--save', help='Caminho para salvar modelo')
    parser.add_argument('--load', help='Caminho para carregar modelo')
    parser.add_argument('--classes', nargs='+',
                       help='Lista de classes personalizadas (padrão: do .env)')
    
    args = parser.parse_args()
    
    # Imprimir configurações se disponíveis
    if CONFIG_AVAILABLE:
        print(f"🔧 Configurações carregadas do .env")
    
    if args.command == 'setup':
        print("📁 Para criar datasets, use o projeto manager:")
        print("   python3 project_manager.py")
        print("   E selecione a opção '3. 🆕 Criar novo dataset'")
        print()
        print("📁 Ou use o dataset_manager diretamente:")
        print("   python3 dataset_manager/dataset_creator.py create --classes break conteudo merchan")
        return
    
    elif args.command == 'train':
        if not args.dataset:
            print("❌ --dataset é obrigatório para treinar")
            return
        
        # Detectar classes do dataset automaticamente
        detected_classes = []
        if os.path.exists(args.dataset):
            for item in os.listdir(args.dataset):
                item_path = os.path.join(args.dataset, item)
                if os.path.isdir(item_path):
                    detected_classes.append(item)
        
        if detected_classes:
            print(f"🎯 Classes detectadas no dataset: {detected_classes}")
            # Criar classificador com classes detectadas
            classifier = SimpleVideoClassifier(classes=detected_classes)
        else:
            # Usar classes especificadas ou do .env
            classes = args.classes if args.classes else (config.get_classes() if CONFIG_AVAILABLE else ['conteudo', 'merchan'])
            print(f"🎯 Usando classes especificadas: {classes}")
            classifier = SimpleVideoClassifier(classes=classes)
        
        X, y = load_dataset_and_extract_features(args.dataset, classifier)
        
        if len(X) == 0:
            print("❌ Dataset vazio")
            return
        
        classifier.train(X, y, args.classifier)
        
        if args.save:
            classifier.save_model(args.save)
    
    elif args.command == 'predict':
        if not args.video or not args.load:
            print("❌ --video e --load são obrigatórios")
            return
        
        # Classificador carregará as classes do modelo salvo
        classifier = SimpleVideoClassifier()
        classifier.load_model(args.load)
        
        predicted_class, confidence = classifier.predict_video(args.video)
        
        print(f"\n🎯 RESULTADO FINAL:")
        print(f"📋 O vídeo é: {predicted_class.upper()}")
        print(f"💯 Confiança: {confidence:.2%}")
    
    elif args.command == 'batch':
        if not args.directory or not args.load:
            print("❌ --directory e --load são obrigatórios")
            return
        
        classifier = SimpleVideoClassifier()
        classifier.load_model(args.load)
        
        results = classifier.batch_predict(args.directory)

if __name__ == "__main__":
    main()