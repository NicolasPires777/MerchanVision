"""
Model Trainer Module

Provides model training functions for video classification.
"""

import os
import json
import time
import joblib
import numpy as np
import cv2
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import argparse

# Importar TensorFlow/Keras para extração de features
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras.applications import EfficientNetB0
    TENSORFLOW_AVAILABLE = True
except ImportError:
    print("⚠️ TensorFlow não disponível - usando features mockadas")
    TENSORFLOW_AVAILABLE = False


class VideoModelTrainer:
    """Classe para treinamento de modelos de classificação de vídeo"""
    
    def __init__(self, classes=['break', 'conteudo', 'merchan']):
        """
        Inicializa o trainer
        
        Args:
            classes: Lista de classes para classificação
        """
        self.classes = classes
        self.classifier = None
        self.scaler = None  # Inicializar scaler
        self.feature_extractor = None  # Extrator de features para vídeos
        
        # Configurações padrão do classificador
        self.classifier_config = {
            'classifier_type': 'rf',
            'n_estimators': 100,
            'max_depth': None,
            'min_samples_split': 5,
            'min_samples_leaf': 2,
            'random_state': 42,
            'class_weight': 'balanced'
        }
    
    def train_from_dataset(self, dataset_path, classifier_type=None, test_size=0.2):
        """
        Treina modelo diretamente de um dataset path
        
        Args:
            dataset_path: Caminho do dataset
            classifier_type: 'rf' ou 'svm' (padrão: 'rf')  
            test_size: Proporção do conjunto de teste
            
        Returns:
            tuple: (model, scaler)
        """
        # Importar aqui para evitar circular imports
        from dataset_manager import load_video_dataset
        
        # Carregar dataset
        dataset_info = load_video_dataset(dataset_path, verbose=True)
        
        if not dataset_info:
            raise ValueError("Dataset vazio ou inválido")
        
        # Processar vídeos em features REAIS (não mockados)
        # Implementação baseada em video_classifier_simple.py
        print("🎬 Processando vídeos e extraindo features REAIS...")
        
        # Configurar extrator de features
        self._setup_feature_extractor()
        
        X = []
        y = []
        
        for class_name, video_files in dataset_info.items():
            print(f"🎯 Processando classe: {class_name} ({len(video_files)} vídeos)")
            
            for i, video_file in enumerate(video_files):
                print(f"  [{i+1}/{len(video_files)}] {os.path.basename(video_file)}")
                
                try:
                    features = self._extract_video_features(video_file)
                    
                    if features is not None:
                        X.append(features)
                        y.append(class_name)
                        print(f"    ✅ Features extraídas: {features.shape}")
                    else:
                        print(f"    ❌ Erro ao extrair features")
                
                except Exception as e:
                    print(f"    ❌ Erro: {e}")
        
        if len(X) == 0:
            raise ValueError("Nenhum vídeo processado com sucesso")
        
        X = np.array(X)
        y = np.array(y)
        
        # Treinar usando o método existente
        results = self.train(X, y, classifier_type, test_size)
        
        return self.classifier, self.scaler
    
    def train(self, X, y, classifier_type=None, test_size=0.2):
        """
        Treina classificador
        
        Args:
            X: Features dos vídeos
            y: Labels  
            classifier_type: 'rf' ou 'svm' (padrão: 'rf')
            test_size: Proporção do conjunto de teste
            
        Returns:
            dict: Resultados do treinamento
        """
        # Usar tipo padrão se não especificado
        if classifier_type is None:
            classifier_type = self.classifier_config['classifier_type']
            
        print(f"🚀 Treinando classificador {classifier_type.upper()}...")
        print(f"📊 Dataset: {len(X)} amostras, {len(self.classes)} classes")
        
        # Dividir em treino e teste
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        print(f"📈 Treino: {len(X_train)} amostras | Teste: {len(X_test)} amostras")
        
        # Escalonar features
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Criar classificador
        start_time = time.time()
        
        if classifier_type == 'rf':
            # Usar configurações padrão
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
            
            print(f"⚙️ Parâmetros RF: {rf_params}")
            self.classifier = RandomForestClassifier(**rf_params)
            
        elif classifier_type == 'svm':
            print(f"⚙️ Usando SVM com kernel RBF")
            self.classifier = SVC(
                kernel='rbf',
                random_state=42,
                probability=True
            )
            
        else:
            raise ValueError(f"Tipo de classificador não suportado: {classifier_type}")
        
        # Treinar modelo
        print("🔄 Treinando modelo...")
        self.classifier.fit(X_train_scaled, y_train)
        
        train_time = time.time() - start_time
        
        # Avaliar resultados
        train_score = self.classifier.score(X_train_scaled, y_train)
        test_score = self.classifier.score(X_test_scaled, y_test)
        
        print(f"\n✅ Treinamento concluído!")
        print(f"  ⏱️ Tempo: {train_time:.2f}s")
        print(f"  📊 Acurácia (treino): {train_score:.4f}")
        print(f"  📊 Acurácia (teste): {test_score:.4f}")
        
        # Relatório de classificação
        y_pred = self.classifier.predict(X_test_scaled)
        
        # Usar labels do conjunto de teste para evitar problemas de classes faltantes
        unique_labels = sorted(list(set(y_test) | set(y_pred)))
        try:
            report = classification_report(y_test, y_pred, labels=unique_labels, target_names=unique_labels)
            print(f"\n📋 Relatório de classificação:")
            print(report)
        except Exception as e:
            print(f"⚠️ Erro no relatório de classificação: {e}")
            report = "Erro ao gerar relatório"
        
        return {
            'classifier_type': classifier_type,
            'train_score': train_score,
            'test_score': test_score,
            'train_time': train_time,
            'classification_report': report,
            'classes': self.classes
        }
    
    def save_model(self, save_path):
        """
        Salva modelo treinado
        
        Args:
            save_path: Caminho para salvar o modelo
            
        Returns:
            bool: True se salvou com sucesso
        """
        if self.classifier is None:
            print("❌ Nenhum modelo treinado para salvar")
            return False
            
        try:
            # Criar diretório se não existir
            os.makedirs(save_path, exist_ok=True)
            
            # Salvar classificador
            model_path = os.path.join(save_path, "classifier.pkl")
            joblib.dump(self.classifier, model_path)
            
            # Salvar configurações
            config = {
                'classes': self.classes,
                'classifier_config': self.classifier_config,
                'model_type': type(self.classifier).__name__
            }
            
            config_path = os.path.join(save_path, "config.json")
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=2)
            
            print(f"💾 Modelo salvo em: {save_path}")
            return True
            
        except Exception as e:
            print(f"❌ Erro ao salvar modelo: {e}")
            return False
    
    def load_model(self, load_path):
        """
        Carrega modelo treinado
        
        Args:
            load_path: Caminho do modelo
            
        Returns:
            bool: True se carregou com sucesso
        """
        try:
            # Carregar configurações
            config_path = os.path.join(load_path, "config.json")
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    config = json.load(f)
                    self.classes = config.get('classes', self.classes)
                    self.classifier_config = config.get('classifier_config', self.classifier_config)
            
            # Carregar classificador
            model_path = os.path.join(load_path, "classifier.pkl")
            if os.path.exists(model_path):
                self.classifier = joblib.load(model_path)
                print(f"✅ Modelo carregado de: {load_path}")
                return True
            else:
                print(f"❌ Arquivo do modelo não encontrado: {model_path}")
                return False
                
        except Exception as e:
            print(f"❌ Erro ao carregar modelo: {e}")
            return False
    
    def _setup_feature_extractor(self):
        """
        Configura o extrator de features baseado em EfficientNetB0
        """
        if not TENSORFLOW_AVAILABLE:
            print("❌ TensorFlow não disponível - usando features mockadas")
            self.feature_extractor = None
            return
        
        try:
            print("🔧 Configurando extrator de features (EfficientNetB0)...")
            
            # Configurações de vídeo
            width, height = 224, 224
            
            # EfficientNetB0 pré-treinado
            base_model = EfficientNetB0(
                weights='imagenet',
                include_top=False,
                pooling='avg',
                input_shape=(height, width, 3)
            )
            
            self.feature_extractor = base_model
            print("✅ Extrator de features configurado")
            
        except Exception as e:
            print(f"❌ Erro ao configurar extrator: {e}")
            self.feature_extractor = None
    
    def _extract_video_features(self, video_path, max_frames=5):
        """
        Extrai features reais de um vídeo usando CNN pré-treinada
        
        Args:
            video_path: Caminho do vídeo
            max_frames: Máximo de frames a processar
        
        Returns:
            Array com features agregadas do vídeo ou None se erro
        """
        if self.feature_extractor is None:
            # Fallback para features mockadas se TensorFlow não disponível
            print("    ⚠️ Usando features mockadas (TensorFlow indisponível)")
            return np.random.rand(1280)  # EfficientNetB0 produz 1280 features
        
        try:
            cap = cv2.VideoCapture(video_path)
            frames = []
            frame_count = 0
            processed_frames = 0
            
            # Obter informações do vídeo
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS) or 30
            duration = total_frames / fps
            
            # Calcular sample_rate para distribuir frames uniformemente
            sample_rate = max(1, total_frames // max_frames)
            
            while processed_frames < max_frames:
                ret, frame = cap.read()
                if not ret:
                    break
                
                if frame_count % sample_rate == 0:
                    # Redimensionar para 224x224 (EfficientNet)
                    frame = cv2.resize(frame, (224, 224))
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame = frame.astype(np.float32) / 255.0
                    frames.append(frame)
                    processed_frames += 1
                
                frame_count += 1
            
            cap.release()
            
            if len(frames) == 0:
                print("    ⚠️ Nenhum frame válido extraído")
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
            
        except Exception as e:
            print(f"    ❌ Erro ao processar vídeo {video_path}: {e}")
            return None


def train_model_from_dataset(dataset_path, model_name, models_dir="models", classifier_type='rf'):
    """
    Função utilitária para treinar modelo a partir de dataset
    
    Args:
        dataset_path: Caminho do dataset
        model_name: Nome do modelo
        models_dir: Diretório dos modelos
        classifier_type: Tipo do classificador
        
    Returns:
        bool: True se treinou com sucesso
    """
    # Import dataset functions
    try:
        from dataset_manager import detect_dataset_classes, load_video_dataset
    except ImportError:
        print("❌ Erro: dataset_manager não encontrado")
        return False
    
    # Import feature extraction (placeholder - seria implementado conforme necessário)
    print("📁 Carregando dataset...")
    classes = detect_dataset_classes(dataset_path)
    if not classes:
        print(f"❌ Nenhuma classe encontrada no dataset: {dataset_path}")
        return False
    
    print(f"🎯 Classes detectadas: {classes}")
    
    # Aqui você implementaria a extração de features dos vídeos
    # Por agora, vou usar um placeholder
    print("⚠️ Implementar extração de features dos vídeos")
    print("💡 Use load_dataset_and_extract_features do video_classifier_simple.py")
    
    return False


if __name__ == "__main__":
    """Interface de linha de comando para treinamento de modelos"""
    parser = argparse.ArgumentParser(description="Treinador de Modelos para Classificação de Vídeos")
    parser.add_argument('command', choices=['train'], 
                       help='Comando a executar')
    parser.add_argument('--dataset', required=True,
                       help='Caminho do dataset')
    parser.add_argument('--model-name', required=True,
                       help='Nome do modelo')
    parser.add_argument('--models-dir', default='models',
                       help='Diretório dos modelos')
    parser.add_argument('--classifier', choices=['rf', 'svm'], default='rf',
                       help='Tipo de classificador')
    
    args = parser.parse_args()
    
    if args.command == 'train':
        print("🚀 Iniciando treinamento...")
        success = train_model_from_dataset(
            args.dataset, 
            args.model_name, 
            args.models_dir,
            args.classifier
        )
        
        if success:
            print("✅ Treinamento concluído com sucesso!")
        else:
            print("❌ Falha no treinamento")
            exit(1)