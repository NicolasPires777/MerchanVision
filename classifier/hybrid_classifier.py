#!/usr/bin/env python3
"""
🤖 Hybrid Video Classifier - Classificação Híbrida de Vídeos

Responsabilidade única: Classificação avançada combinando CNN + indicadores visuais
Combina análise de imagem tradicional com detecção específica de elementos comerciais.
"""

import cv2
import numpy as np
import os
import joblib
from pathlib import Path
import json

# Importar módulos do projeto
try:
    from classifier.basic_classifier import BasicVideoClassifier
    from classifier.visual_elements_detector import VisualElementsDetector
    from config import config
    MODULES_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ Erro ao importar módulos: {e}")
    MODULES_AVAILABLE = False

class HybridVideoClassifier:
    """Classificador que combina análise de imagem com detecção de elementos visuais"""
    
    def __init__(self, classes=None):
        """Inicializa classificador híbrido"""
        self.classes = classes if classes else ['conteudo', 'merchan']
        
        # Inicializar componentes
        if MODULES_AVAILABLE:
            self.image_classifier = BasicVideoClassifier(classes=self.classes)
            self.visual_detector = VisualElementsDetector()
        else:
            print("❌ Módulos não disponíveis - modo limitado")
            return
        
        # Configurações do modelo híbrido (usar configurações do .env se disponível)
        self.weights = {
            'image_score': config.get('HYBRID_IMAGE_WEIGHT', 0.5),
            'merchan_score': config.get('HYBRID_MERCHAN_WEIGHT', 0.5),
        }
        
        # Limites para decisão
        self.thresholds = {
            'merchan_min': config.get('HYBRID_MERCHAN_THRESHOLD', 0.25),
            'image_confidence': config.get('HYBRID_IMAGE_CONFIDENCE_THRESHOLD', 0.6),
        }
        
        print(f"🤖 Classificador híbrido inicializado")
        print(f"📊 Pesos: Imagem {self.weights['image_score']:.1f} + Indicadores {self.weights['merchan_score']:.1f}")
    
    def extract_hybrid_features(self, video_path):
        """Extrai features híbridas: EfficientNet + indicadores visuais"""
        if not MODULES_AVAILABLE:
            return None, None
        
        # Features tradicionais do modelo de imagem
        image_features = self.image_classifier.extract_video_features(video_path)
        
        # Indicadores visuais de merchandising
        merchan_analysis = self.visual_detector.analyze_video(video_path, max_frames=8)
        
        return image_features, merchan_analysis
    
    def extract_features_from_frames(self, frames_array):
        """
        Extrai features de frames já carregados (para tempo real)
        
        Args:
            frames_array (np.array): Array de frames [N, H, W, 3]
        
        Returns:
            np.array: Features agregadas do classificador de imagem
        """
        if not MODULES_AVAILABLE:
            return None
        
        return self.image_classifier.extract_features_from_frames(frames_array)
    
    def predict_hybrid(self, video_path, return_details=False):
        """Predição híbrida combinando imagem + indicadores visuais"""
        if not MODULES_AVAILABLE:
            return None
        
        # Extrair features
        image_features, merchan_analysis = self.extract_hybrid_features(video_path)
        
        if image_features is None or merchan_analysis is None:
            print(f"❌ Erro ao extrair features de: {video_path}")
            return None
        
        # Predição do modelo de imagem
        image_probs = self.image_classifier.classifier.predict_proba([image_features])[0]
        image_prediction = self.image_classifier.classifier.predict([image_features])[0]
        image_confidence = max(image_probs)
        
        # Score dos indicadores visuais
        merchan_score = merchan_analysis['avg_merchan_score']
        merchan_elements = merchan_analysis['all_elements_found']
        
        # Lógica híbrida de decisão
        final_prediction = self._make_hybrid_decision(
            image_prediction, image_probs, image_confidence,
            merchan_score, merchan_elements
        )
        
        result = {
            'prediction': self.classes[final_prediction],
            'confidence': self._calculate_hybrid_confidence(image_probs, merchan_score),
            'image_prediction': self.classes[image_prediction],
            'image_confidence': image_confidence,
            'merchan_score': merchan_score,
            'merchan_elements': merchan_elements,
            'decision_logic': self._get_decision_explanation(
                image_prediction, image_confidence, merchan_score, merchan_elements
            )
        }
        
        if return_details:
            result['merchan_analysis'] = merchan_analysis
            result['image_probs'] = image_probs
        
        return result
    
    def _make_hybrid_decision(self, image_pred, image_probs, image_conf, merchan_score, elements):
        """Lógica de decisão híbrida"""
        merchan_idx = 1 if 'merchan' in self.classes else 0
        
        # Regra 1: Se há indicadores visuais fortes, é muito provavelmente merchan
        if merchan_score >= self.thresholds['merchan_min']:
            # Indicadores fortes encontrados
            return merchan_idx
        
        # Regra 2: Se modelo de imagem tem alta confiança, usar sua decisão
        if image_conf >= self.thresholds['image_confidence']:
            return image_pred
        
        # Regra 3: Decisão híbrida ponderada (para casos duvidosos)
        # Combinar probabilidades do modelo de imagem com score de indicadores
        
        # Ajustar probabilidade de merchan baseado nos indicadores
        adjusted_probs = image_probs.copy()
        
        if merchan_idx < len(adjusted_probs):
            # Aumentar probabilidade de merchan baseado no score de indicadores
            merchan_boost = merchan_score * self.weights['merchan_score']
            adjusted_probs[merchan_idx] += merchan_boost
            
            # Normalizar probabilidades
            adjusted_probs = adjusted_probs / np.sum(adjusted_probs)
        
        return np.argmax(adjusted_probs)
    
    def _calculate_hybrid_confidence(self, image_probs, merchan_score):
        """Calcula confiança híbrida"""
        # Confiança base do modelo de imagem
        image_conf = max(image_probs)
        
        # Ajustar confiança baseado nos indicadores visuais
        if merchan_score > 0:
            # Se há indicadores, aumentar confiança
            boost = merchan_score * 0.2  # Boost máximo de 20%
            return min(1.0, image_conf + boost)
        else:
            return image_conf
    
    def _get_decision_explanation(self, image_pred, image_conf, merchan_score, elements):
        """Explica como a decisão foi tomada"""
        explanations = []
        
        if merchan_score >= self.thresholds['merchan_min']:
            explanations.append(f"🛒 Indicadores visuais fortes (score: {merchan_score:.2f})")
            explanations.append(f"   Elementos: {', '.join(elements) if elements else 'Vários'}")
        
        if image_conf >= self.thresholds['image_confidence']:
            class_name = ['conteudo', 'merchan'][image_pred] if image_pred < 2 else 'unknown'
            explanations.append(f"🖼️ Modelo de imagem confiante: {class_name} ({image_conf:.2f})")
        
        if not explanations:
            explanations.append("⚖️ Decisão híbrida ponderada (baixa confiança)")
        
        return explanations
    
    def train_hybrid(self, dataset_path, save_path):
        """Treina modelo híbrido"""
        if not MODULES_AVAILABLE:
            print("❌ Módulos não disponíveis")
            return False
        
        print(f"🚀 Treinando modelo híbrido...")
        print(f"📁 Dataset: {dataset_path}")
        
        # Treinar o componente de imagem normalmente
        X, y = self.image_classifier.load_dataset(dataset_path)
        
        if len(X) == 0:
            print("❌ Dataset vazio")
            return False
        
        self.image_classifier.train(X, y)
        
        # Analisar dataset para otimizar pesos híbridos
        self._optimize_hybrid_weights(dataset_path)
        
        # Salvar modelo híbrido
        self.save_model(save_path)
        
        print(f"✅ Modelo híbrido treinado e salvo em: {save_path}")
        return True
    
    def _optimize_hybrid_weights(self, dataset_path):
        """Otimiza pesos do modelo híbrido baseado no dataset"""
        print(f"⚙️ Otimizando pesos híbridos...")
        
        # Analisar alguns vídeos de cada classe para ajustar pesos
        for class_name in self.classes:
            class_path = os.path.join(dataset_path, class_name)
            if not os.path.exists(class_path):
                continue
            
            videos = [f for f in os.listdir(class_path) if f.endswith(('.mp4', '.avi', '.mov'))]
            
            merchan_scores = []
            for video in videos[:5]:  # Analisar até 5 vídeos por classe
                video_path = os.path.join(class_path, video)
                try:
                    analysis = self.merchan_detector.analyze_video(video_path, max_frames=5)
                    merchan_scores.append(analysis['avg_merchan_score'])
                except Exception:
                    continue
            
            if merchan_scores:
                avg_score = np.mean(merchan_scores)
                print(f"   📊 {class_name}: score médio de indicadores = {avg_score:.2f}")
                
                # Ajustar limites baseado nos dados
                if class_name == 'merchan' and avg_score > 0:
                    # Se vídeos de merchan têm score baixo, diminuir threshold
                    self.thresholds['merchan_min'] = max(0.2, avg_score * 0.8)
    
    def save_model(self, save_path):
        """Salva modelo híbrido"""
        os.makedirs(save_path, exist_ok=True)
        
        # Salvar componente de imagem
        self.image_classifier.save_model(save_path)
        
        # Salvar configurações híbridas
        hybrid_config = {
            'classes': self.classes,
            'network': self.network,
            'weights': self.weights,
            'thresholds': self.thresholds
        }
        
        config_path = os.path.join(save_path, 'hybrid_config.json')
        with open(config_path, 'w') as f:
            json.dump(hybrid_config, f, indent=2)
        
        print(f"💾 Configurações híbridas salvas em: {config_path}")
    
    def load_model(self, model_path):
        """Carrega modelo híbrido"""
        if not MODULES_AVAILABLE:
            return False
        
        # Carregar componente de imagem
        success = self.image_classifier.load_model(model_path)
        
        # Carregar configurações híbridas
        config_path = os.path.join(model_path, 'hybrid_config.json')
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                hybrid_config = json.load(f)
                self.weights = hybrid_config.get('weights', self.weights)
                self.thresholds = hybrid_config.get('thresholds', self.thresholds)
            print(f"✅ Configurações híbridas carregadas")
        
        return success

def main():
    """Teste do classificador híbrido"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Classificador Híbrido de Vídeo")
    parser.add_argument('command', choices=['train', 'predict', 'test'])
    parser.add_argument('--dataset', help='Caminho do dataset (train)')
    parser.add_argument('--video', help='Caminho do vídeo (predict/test)')
    parser.add_argument('--model', help='Caminho para salvar/carregar modelo')
    parser.add_argument('--network', choices=['record', 'sbt'], help='Rede específica')
    
    args = parser.parse_args()
    
    if not MODULES_AVAILABLE:
        print("❌ Módulos necessários não disponíveis")
        return
    
    # Criar classificador
    classifier = HybridVideoClassifier(
        classes=['conteudo', 'merchan'],
        network=args.network
    )
    
    if args.command == 'train':
        if not args.dataset or not args.model:
            print("❌ --dataset e --model são obrigatórios para treinar")
            return
        
        classifier.train_hybrid(args.dataset, args.model)
    
    elif args.command == 'predict':
        if not args.video or not args.model:
            print("❌ --video e --model são obrigatórios para predizer")
            return
        
        # Carregar modelo
        if not classifier.load_model(args.model):
            print("❌ Erro ao carregar modelo")
            return
        
        # Predizer
        result = classifier.predict_hybrid(args.video, return_details=True)
        
        if result:
            print(f"\n🎯 === RESULTADO HÍBRIDO ===")
            print(f"🎬 Vídeo: {args.video}")
            print(f"📊 Predição: {result['prediction']} ({result['confidence']:.2f})")
            print(f"🖼️ Modelo imagem: {result['image_prediction']} ({result['image_confidence']:.2f})")
            print(f"🛒 Score merchan: {result['merchan_score']:.2f}")
            
            if result['merchan_elements']:
                print(f"🔍 Elementos encontrados:")
                for element in result['merchan_elements']:
                    print(f"   - {element}")
            
            print(f"💭 Lógica de decisão:")
            for explanation in result['decision_logic']:
                print(f"   {explanation}")
    
    elif args.command == 'test':
        # Testar apenas detecção de elementos
        if not args.video:
            print("❌ --video é obrigatório para testar")
            return
        
        detector = VisualElementsDetector()
        results = detector.analyze_video(args.video, max_frames=10)
        
        print(f"\n🔍 === TESTE DE DETECÇÃO ===")
        print(f"🎬 Vídeo: {args.video}")
        print(f"📊 Score médio: {results['avg_merchan_score']:.2f}")
        print(f"🎯 Score máximo: {results['max_merchan_score']:.2f}")
        
        if results['all_elements_found']:
            print(f"🛒 Elementos detectados:")
            for element in results['all_elements_found']:
                print(f"   - {element}")
        else:
            print(f"❌ Nenhum indicador visual encontrado")

if __name__ == "__main__":
    main()