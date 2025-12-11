"""
🎬 Basic Video Classifier - Classificação Básica de Vídeos

Responsabilidade única: Classificar vídeos usando features CNN extraídas
Usa o FeatureExtractor para obter features e aplica ML clássico (RF/SVM).
"""

import os
import joblib
import numpy as np
import argparse
import json
import time
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

# Módulos do projeto
from classifier.feature_extractor import VideoFeatureExtractor
from config import config


class BasicVideoClassifier:
    """Classificador básico de vídeos usando features CNN + ML clássico"""
    
    def __init__(self, classes=None):
        """
        Inicializa classificador
        
        Args:
            classes (list): Lista de classes (padrão: do config)
        """
        self.feature_extractor = VideoFeatureExtractor()
        self.classifier = None
        
        # Configurações
        try:
            self.classifier_config = config.get_classifier_config()
            if classes is None:
                self.classes = config.get_classes()
            else:
                self.classes = classes
            print(f"✅ Configurações carregadas do .env")
        except:
            # Configurações padrão
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
            print(f"⚠️ Usando configurações padrão")
        
        print(f"🎯 Classes configuradas: {self.classes}")
        print(f"📊 Classificador: {self.classifier_config['classifier_type'].upper()}")
    
    def extract_video_features(self, video_path):
        """
        Extrai features de um vídeo (delega para FeatureExtractor)
        
        Args:
            video_path (str): Caminho do vídeo
        
        Returns:
            np.array: Features agregadas
        """
        return self.feature_extractor.extract_video_features(video_path)
    
    def extract_features_from_frames(self, frames_array):
        """
        Extrai features de frames já carregados (para tempo real)
        
        Args:
            frames_array (np.array): Array de frames [N, H, W, 3]
        
        Returns:
            np.array: Features agregadas
        """
        return self.feature_extractor.extract_features_from_frames(frames_array)
    
    def train(self, X, y, classifier_type=None):
        """
        Treina classificador usando configurações
        
        Args:
            X (np.array): Features dos vídeos
            y (np.array): Labels correspondentes
            classifier_type (str): 'rf' ou 'svm' (padrão: config)
        
        Returns:
            Classificador treinado
        """
        if classifier_type is None:
            classifier_type = self.classifier_config['classifier_type']
        
        print(f"🚀 Treinando classificador {classifier_type.upper()}...")
        print(f"📊 Dataset: {X.shape[0]} vídeos, {X.shape[1]} features")
        
        # Configurar classificador
        if classifier_type == 'rf':
            rf_params = {
                'n_estimators': self.classifier_config['n_estimators'],
                'max_depth': self.classifier_config.get('max_depth'),
                'min_samples_split': self.classifier_config['min_samples_split'],
                'min_samples_leaf': self.classifier_config['min_samples_leaf'],
                'random_state': self.classifier_config['random_state'],
                'n_jobs': -1
            }
            
            if 'class_weight' in self.classifier_config:
                rf_params['class_weight'] = self.classifier_config['class_weight']
            
            # Remover valores None
            rf_params = {k: v for k, v in rf_params.items() if v is not None}
            
            print(f"⚙️ Parâmetros RF: {rf_params}")
            self.classifier = RandomForestClassifier(**rf_params)
            
        elif classifier_type == 'svm':
            svm_params = {
                'kernel': 'rbf',
                'probability': True,
                'random_state': self.classifier_config['random_state'],
                'class_weight': self.classifier_config.get('class_weight', 'balanced')
            }
            
            print(f"⚙️ Parâmetros SVM: {svm_params}")
            self.classifier = SVC(**svm_params)
        else:
            raise ValueError(f"Classifier type não suportado: {classifier_type}")
        
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
        Classifica um único vídeo
        
        Args:
            video_path (str): Caminho do vídeo
        
        Returns:
            tuple: (classe, confiança)
        """
        if self.classifier is None:
            raise ValueError("Modelo não foi treinado! Use train() primeiro")
        
        print(f"🎬 Analisando: {os.path.basename(video_path)}")
        
        # Extrair features
        features = self.extract_video_features(video_path)
        
        if features is None:
            print(f"❌ Erro ao extrair features")
            return "erro", 0.0
        
        # Predição
        features_reshaped = features.reshape(1, -1)
        prediction = self.classifier.predict(features_reshaped)[0]
        probabilities = self.classifier.predict_proba(features_reshaped)[0]
        
        predicted_class = self.classes[prediction]
        confidence = probabilities[prediction]
        
        # Exibir resultado
        print(f"🎯 Resultado:")
        print(f"  📋 Classe: {predicted_class}")
        print(f"  💯 Confiança: {confidence:.4f}")
        print(f"  📊 Probabilidades:")
        for i, class_name in enumerate(self.classes):
            print(f"    {class_name}: {probabilities[i]:.4f}")
        
        return predicted_class, confidence
    
    def batch_predict(self, videos_directory):
        """
        Prediz múltiplos vídeos de um diretório
        
        Args:
            videos_directory (str): Caminho do diretório
        
        Returns:
            dict: Resultados {arquivo: {class, confidence}}
        """
        if self.classifier is None:
            raise ValueError("Modelo não foi treinado!")
        
        # Encontrar vídeos
        video_extensions = ['*.mp4', '*.avi', '*.mov', '*.mkv', '*.MP4']
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
        
        # Resumo estatístico
        self._print_batch_summary(results)
        
        return results
    
    def _print_batch_summary(self, results):
        """Imprime resumo dos resultados em lote"""
        print(f"\n📊 Resumo dos resultados:")
        
        # Contar por classe
        class_counts = {cls: 0 for cls in self.classes}
        class_counts['erro'] = 0
        
        for result in results.values():
            class_name = result['class']
            class_counts[class_name] = class_counts.get(class_name, 0) + 1
        
        for class_name, count in class_counts.items():
            if count > 0:
                percentage = (count / len(results)) * 100
                print(f"  📋 {class_name}: {count} ({percentage:.1f}%)")
    
    def save_model(self, save_path):
        """
        Salva modelo e configurações
        
        Args:
            save_path (str): Diretório para salvar
        """
        if self.classifier is None:
            raise ValueError("Nenhum modelo para salvar!")
        
        os.makedirs(save_path, exist_ok=True)
        
        # Salvar classificador
        model_path = os.path.join(save_path, "classifier.pkl")
        joblib.dump(self.classifier, model_path)
        
        # Salvar configurações
        config_data = {
            'classes': self.classes,
            'classifier_config': self.classifier_config,
            'feature_dimension': self.feature_extractor.get_feature_dimension()
        }
        
        config_path = os.path.join(save_path, "config.json")
        with open(config_path, 'w') as f:
            json.dump(config_data, f, indent=2)
        
        print(f"💾 Modelo salvo em: {save_path}")
        print(f"  🎯 Classes: {self.classes}")
        print(f"  📐 Features: {config_data['feature_dimension']}")
    
    def load_model(self, load_path):
        """
        Carrega modelo e configurações salvas
        
        Args:
            load_path (str): Caminho do arquivo ou diretório
        
        Returns:
            bool: Sucesso no carregamento
        """
        # Determinar caminhos
        if load_path.endswith('.pkl'):
            base_path = os.path.dirname(load_path)
            model_file = load_path
        else:
            base_path = load_path
            model_file = os.path.join(load_path, "classifier.pkl")
        
        if not os.path.exists(model_file):
            raise FileNotFoundError(f"Modelo não encontrado: {model_file}")
        
        # Carregar configurações
        config_path = os.path.join(base_path, "config.json")
        if os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    saved_config = json.load(f)
                    self.classes = saved_config.get('classes', ['conteudo', 'merchan'])
                    print(f"🎯 Classes carregadas: {self.classes}")
            except Exception as e:
                print(f"⚠️ Erro ao carregar config: {e}")
                self.classes = ['conteudo', 'merchan']
        else:
            print("⚠️ Config não encontrada, usando classes padrão")
            self.classes = ['conteudo', 'merchan']
        
        # Carregar classificador
        try:
            self.classifier = joblib.load(model_file)
            print(f"✅ Modelo carregado de: {model_file}")
            return True
        except Exception as e:
            print(f"❌ Erro ao carregar modelo: {e}")
            return False


# Dataset loading helper (integração com dataset_manager)
def load_dataset_and_extract_features(dataset_path, classifier_instance):
    """
    Carrega dataset e extrai features usando classifier
    
    Args:
        dataset_path (str): Caminho do dataset
        classifier_instance (SimpleVideoClassifier): Instância do classificador
    
    Returns:
        tuple: (X, y) arrays com features e labels
    """
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from dataset_manager import load_video_dataset, detect_dataset_classes
    
    print(f"📁 Carregando dataset de: {dataset_path}")
    
    # Detectar classes reais no dataset
    actual_classes = detect_dataset_classes(dataset_path)
    
    # Usar apenas classes que existem no dataset e no classifier
    available_classes = [cls for cls in classifier_instance.classes if cls in actual_classes]
    
    if not available_classes:
        print(f"❌ Nenhuma classe válida encontrada!")
        print(f"   Configuradas: {classifier_instance.classes}")
        print(f"   No dataset: {actual_classes}")
        return np.array([]), np.array([])
    
    print(f"🎯 Usando classes: {available_classes}")
    
    # Carregar informações do dataset
    dataset_info = load_video_dataset(dataset_path, available_classes, verbose=False)
    
    X, y = [], []
    
    for class_idx, class_name in enumerate(available_classes):
        video_files = dataset_info.get(class_name, [])
        
        if not video_files:
            print(f"⚠️ Nenhum vídeo para: {class_name}")
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
    
    # Estatísticas
    print(f"\n📊 Dataset processado:")
    print(f"  🎬 Total de vídeos: {len(X)}")
    print(f"  📐 Dimensão features: {X.shape[1] if len(X) > 0 else 0}")
    
    for i, class_name in enumerate(available_classes):
        count = np.sum(y == i)
        percentage = (count / len(y) * 100) if len(y) > 0 else 0
        print(f"  📋 {class_name}: {count} vídeos ({percentage:.1f}%)")
    
    return X, y


def main():
    """Interface CLI para classificação básica"""
    parser = argparse.ArgumentParser(description="Basic Video Classifier")
    parser.add_argument('command', choices=['train', 'predict', 'batch'], 
                       help='Comando a executar')
    parser.add_argument('--dataset', help='Caminho do dataset (train)')
    parser.add_argument('--video', help='Caminho do vídeo (predict)')
    parser.add_argument('--directory', help='Diretório de vídeos (batch)')
    parser.add_argument('--classifier', choices=['rf', 'svm'], 
                       help='Tipo de classificador')
    parser.add_argument('--save', help='Caminho para salvar modelo')
    parser.add_argument('--load', help='Caminho para carregar modelo')
    parser.add_argument('--classes', nargs='+',
                       help='Lista de classes personalizadas')
    
    args = parser.parse_args()
    
    if args.command == 'train':
        if not args.dataset:
            print("❌ --dataset obrigatório para train")
            return
        
        # Auto-detectar classes do dataset se não especificadas
        if not args.classes:
            detected_classes = []
            if os.path.exists(args.dataset):
                for item in os.listdir(args.dataset):
                    item_path = os.path.join(args.dataset, item)
                    if os.path.isdir(item_path):
                        detected_classes.append(item)
            classes = detected_classes if detected_classes else ['conteudo', 'merchan']
        else:
            classes = args.classes
        
        print(f"🎯 Classes: {classes}")
        classifier = BasicVideoClassifier(classes=classes)
        
        # Carregar e extrair features
        X, y = load_dataset_and_extract_features(args.dataset, classifier)
        
        if len(X) == 0:
            print("❌ Dataset vazio")
            return
        
        # Treinar
        classifier.train(X, y, args.classifier)
        
        # Salvar se solicitado
        if args.save:
            classifier.save_model(args.save)
    
    elif args.command == 'predict':
        if not args.video or not args.load:
            print("❌ --video e --load obrigatórios")
            return
        
        classifier = BasicVideoClassifier()
        classifier.load_model(args.load)
        
        predicted_class, confidence = classifier.predict_video(args.video)
        
        print(f"\n🎯 RESULTADO FINAL:")
        print(f"📋 Classe: {predicted_class.upper()}")
        print(f"💯 Confiança: {confidence:.2%}")
    
    elif args.command == 'batch':
        if not args.directory or not args.load:
            print("❌ --directory e --load obrigatórios")
            return
        
        classifier = BasicVideoClassifier()
        classifier.load_model(args.load)
        
        results = classifier.batch_predict(args.directory)


if __name__ == "__main__":
    main()