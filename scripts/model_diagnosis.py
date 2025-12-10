#!/usr/bin/env python3
"""
Script de Diagnóstico de Modelos de Classificação de Vídeo
Analisa performance detalhada e identifica problemas
"""

import os
import sys
import numpy as np
import json
from pathlib import Path
import argparse
from collections import defaultdict, Counter

# Adicionar paths
current_dir = Path(__file__).parent
sys.path.append(str(current_dir))

try:
    from video_classifier_simple import SimpleVideoClassifier
except ImportError:
    print("❌ Erro: video_classifier_simple.py não encontrado")
    sys.exit(1)

class ModelDiagnosis:
    def __init__(self):
        self.results = {}
        
    def analyze_model(self, model_path, dataset_path, max_videos_per_class=10):
        """Analisa performance detalhada do modelo"""
        
        print(f"\n🔍 === ANÁLISE DO MODELO ===")
        print(f"📁 Modelo: {model_path}")
        print(f"📂 Dataset: {dataset_path}")
        
        # Carregar modelo
        try:
            classifier = SimpleVideoClassifier()
            if os.path.isdir(model_path):
                success = classifier.load_model(model_path)
            else:
                success = classifier.load_model_from_file(f"{model_path}.pkl" if not model_path.endswith('.pkl') else model_path)
            
            if not success:
                print(f"❌ Erro ao carregar modelo: {model_path}")
                return None
                
            print(f"✅ Modelo carregado: {classifier.classes}")
            
        except Exception as e:
            print(f"❌ Erro ao carregar modelo: {e}")
            return None
        
        # Analisar cada classe
        results = {
            'model_path': model_path,
            'dataset_path': dataset_path,
            'classes': classifier.classes,
            'class_analysis': {},
            'confusion_matrix': defaultdict(lambda: defaultdict(int)),
            'overall_stats': {}
        }
        
        total_correct = 0
        total_videos = 0
        confidence_scores = []
        
        for class_name in ['break', 'conteudo', 'merchan']:
            class_path = os.path.join(dataset_path, class_name)
            if not os.path.exists(class_path):
                print(f"⚠️ Pasta não encontrada: {class_path}")
                continue
                
            videos = [f for f in os.listdir(class_path) if f.endswith('.mp4')][:max_videos_per_class]
            
            print(f"\n📺 === CLASSE: {class_name.upper()} ({len(videos)} vídeos) ===")
            
            class_correct = 0
            class_predictions = []
            class_confidences = []
            errors = []
            
            for i, video in enumerate(videos):
                video_path = os.path.join(class_path, video)
                
                try:
                    # Extrair features e classificar
                    features = classifier.extract_video_features(video_path, max_frames=50)
                    if features is None:
                        errors.append(f"Erro ao extrair features: {video}")
                        continue
                    
                    prediction_idx = classifier.classifier.predict([features])[0]
                    probabilities = classifier.classifier.predict_proba([features])[0]
                    predicted_class = classifier.classes[prediction_idx]
                    confidence = max(probabilities)
                    
                    # Estatísticas
                    class_predictions.append(predicted_class)
                    class_confidences.append(confidence)
                    confidence_scores.append(confidence)
                    
                    # Matriz de confusão
                    results['confusion_matrix'][class_name][predicted_class] += 1
                    
                    # Verificar acerto
                    if predicted_class == class_name:
                        class_correct += 1
                        total_correct += 1
                        status = "✅"
                    else:
                        status = "❌"
                        errors.append(f"{video}: {class_name} → {predicted_class} ({confidence:.1%})")
                    
                    total_videos += 1
                    
                    # Exibir progresso
                    short_name = video[:25] + "..." if len(video) > 25 else video
                    print(f"  {status} {short_name:<30} → {predicted_class:<8} ({confidence:.1%})")
                    
                except Exception as e:
                    errors.append(f"Erro ao processar {video}: {e}")
                    print(f"  ⚠️ {video[:30]:<30} → Erro: {e}")
            
            # Estatísticas da classe
            class_accuracy = (class_correct / len(videos) * 100) if videos else 0
            avg_confidence = np.mean(class_confidences) if class_confidences else 0
            prediction_counts = Counter(class_predictions)
            
            results['class_analysis'][class_name] = {
                'accuracy': class_accuracy,
                'avg_confidence': avg_confidence,
                'total_videos': len(videos),
                'correct_predictions': class_correct,
                'prediction_distribution': dict(prediction_counts),
                'errors': errors[:5]  # Top 5 erros
            }
            
            print(f"\n  📊 ESTATÍSTICAS {class_name.upper()}:")
            print(f"     Precisão: {class_accuracy:.1f}% ({class_correct}/{len(videos)})")
            print(f"     Confiança média: {avg_confidence:.1%}")
            print(f"     Predições: {dict(prediction_counts)}")
            
            if errors:
                print(f"  🔀 PRINCIPAIS ERROS:")
                for error in errors[:3]:
                    print(f"     {error}")
        
        # Estatísticas gerais
        overall_accuracy = (total_correct / total_videos * 100) if total_videos else 0
        avg_confidence = np.mean(confidence_scores) if confidence_scores else 0
        
        results['overall_stats'] = {
            'accuracy': overall_accuracy,
            'avg_confidence': avg_confidence,
            'total_videos': total_videos,
            'total_correct': total_correct
        }
        
        print(f"\n🎯 === ESTATÍSTICAS GERAIS ===")
        print(f"📊 Precisão geral: {overall_accuracy:.1f}% ({total_correct}/{total_videos})")
        print(f"🎯 Confiança média: {avg_confidence:.1%}")
        
        # Matriz de confusão
        print(f"\n🔀 === MATRIZ DE CONFUSÃO ===")
        print(f"{'Real \\ Predito':<15}", end="")
        for pred_class in ['break', 'conteudo', 'merchan']:
            print(f"{pred_class:<10}", end="")
        print()
        
        for real_class in ['break', 'conteudo', 'merchan']:
            print(f"{real_class:<15}", end="")
            for pred_class in ['break', 'conteudo', 'merchan']:
                count = results['confusion_matrix'][real_class][pred_class]
                print(f"{count:<10}", end="")
            print()
        
        return results
    
    def compare_models(self, model1_path, model2_path, dataset_path, max_videos_per_class=10):
        """Compara dois modelos"""
        
        print(f"\n🆚 === COMPARAÇÃO DE MODELOS ===")
        
        results1 = self.analyze_model(model1_path, dataset_path, max_videos_per_class)
        results2 = self.analyze_model(model2_path, dataset_path, max_videos_per_class)
        
        if not results1 or not results2:
            return
        
        print(f"\n📈 === COMPARAÇÃO FINAL ===")
        print(f"{'Métrica':<20} {'Modelo 1':<15} {'Modelo 2':<15} {'Diferença':<10}")
        print("-" * 65)
        
        # Precisão geral
        acc1 = results1['overall_stats']['accuracy']
        acc2 = results2['overall_stats']['accuracy']
        diff_acc = acc2 - acc1
        print(f"{'Precisão Geral':<20} {acc1:<15.1f} {acc2:<15.1f} {diff_acc:>+9.1f}")
        
        # Confiança média
        conf1 = results1['overall_stats']['avg_confidence'] * 100
        conf2 = results2['overall_stats']['avg_confidence'] * 100
        diff_conf = conf2 - conf1
        print(f"{'Confiança Média':<20} {conf1:<15.1f} {conf2:<15.1f} {diff_conf:>+9.1f}")
        
        # Por classe
        print(f"\n📊 === PRECISÃO POR CLASSE ===")
        for class_name in ['break', 'conteudo', 'merchan']:
            if class_name in results1['class_analysis'] and class_name in results2['class_analysis']:
                acc1_class = results1['class_analysis'][class_name]['accuracy']
                acc2_class = results2['class_analysis'][class_name]['accuracy']
                diff_class = acc2_class - acc1_class
                print(f"{class_name.upper():<20} {acc1_class:<15.1f} {acc2_class:<15.1f} {diff_class:>+9.1f}")
        
        return results1, results2

def main():
    parser = argparse.ArgumentParser(description="Diagnóstico de modelos de classificação")
    parser.add_argument('--model1', required=True, help='Caminho do primeiro modelo')
    parser.add_argument('--model2', help='Caminho do segundo modelo (opcional)')
    parser.add_argument('--dataset1', required=True, help='Caminho do dataset para modelo1')
    parser.add_argument('--dataset2', help='Caminho do dataset para modelo2 (padrão: mesmo que dataset1)')
    parser.add_argument('--max-videos', type=int, default=10, help='Máximo de vídeos por classe')
    parser.add_argument('--output', help='Arquivo para salvar resultados em JSON')
    
    args = parser.parse_args()
    
    diagnosis = ModelDiagnosis()
    
    if args.model2:
        # Comparar dois modelos
        dataset2 = args.dataset2 if args.dataset2 else args.dataset1
        results = diagnosis.compare_models(args.model1, args.model2, args.dataset1, args.max_videos)
    else:
        # Analisar um modelo
        results = diagnosis.analyze_model(args.model1, args.dataset1, args.max_videos)
    
    if args.output and results:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\n💾 Resultados salvos em: {args.output}")

if __name__ == "__main__":
    main()