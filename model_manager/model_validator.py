"""
Model Validator Module

Provides model validation and diagnostic functions for video classification.
"""

import os
import json
import numpy as np
from pathlib import Path
from collections import defaultdict, Counter
import argparse


class ModelValidator:
    """Classe para validação e diagnóstico de modelos"""
    
    def __init__(self):
        self.results = {}
    
    def validate_model_structure(self, model_path):
        """
        Valida a estrutura básica do modelo
        
        Args:
            model_path: Caminho do modelo
            
        Returns:
            dict: Resultado da validação
        """
        print(f"🔍 Validando estrutura do modelo: {model_path}")
        
        issues = []
        info = {}
        
        # Verificar se o diretório existe
        if not os.path.exists(model_path):
            issues.append(f"Diretório do modelo não existe: {model_path}")
            return {'valid': False, 'issues': issues, 'info': info}
        
        # Verificar arquivo do classificador
        classifier_path = os.path.join(model_path, "classifier.pkl")
        if not os.path.exists(classifier_path):
            issues.append("Arquivo classifier.pkl não encontrado")
        else:
            info['classifier_file'] = classifier_path
            
            # Verificar tamanho do arquivo
            size_mb = os.path.getsize(classifier_path) / (1024 * 1024)
            info['classifier_size_mb'] = round(size_mb, 2)
            
            if size_mb < 0.1:
                issues.append("Arquivo classifier.pkl muito pequeno (pode estar corrompido)")
            elif size_mb > 100:
                issues.append("Arquivo classifier.pkl muito grande (pode haver problema)")
        
        # Verificar arquivo de configuração
        config_path = os.path.join(model_path, "config.json")
        if not os.path.exists(config_path):
            issues.append("Arquivo config.json não encontrado")
        else:
            info['config_file'] = config_path
            
            try:
                with open(config_path, 'r') as f:
                    config = json.load(f)
                    info['classes'] = config.get('classes', [])
                    info['model_type'] = config.get('model_type', 'Unknown')
                    
                    if not info['classes']:
                        issues.append("Classes não definidas na configuração")
                        
            except json.JSONDecodeError:
                issues.append("Arquivo config.json corrompido")
            except Exception as e:
                issues.append(f"Erro ao ler config.json: {e}")
        
        # Verificar outros arquivos opcionais
        optional_files = ['hybrid_config.json', 'training_log.txt']
        for file in optional_files:
            file_path = os.path.join(model_path, file)
            if os.path.exists(file_path):
                info[f'has_{file.replace(".", "_")}'] = True
        
        # Resultado final
        is_valid = len(issues) == 0
        
        if is_valid:
            print("✅ Estrutura do modelo válida")
        else:
            print("❌ Problemas encontrados na estrutura:")
            for issue in issues:
                print(f"  • {issue}")
        
        return {
            'valid': is_valid,
            'issues': issues,
            'info': info
        }
    
    def analyze_model_performance(self, model_path, dataset_path, max_videos_per_class=10):
        """
        Analisa performance detalhada do modelo
        
        Args:
            model_path: Caminho do modelo
            dataset_path: Caminho do dataset de teste
            max_videos_per_class: Máximo de vídeos por classe para teste
            
        Returns:
            dict: Resultados da análise
        """
        print(f"\n🔍 === ANÁLISE DE PERFORMANCE ===")
        print(f"📁 Modelo: {model_path}")
        print(f"📂 Dataset: {dataset_path}")
        
        # Primeiro validar estrutura
        structure_check = self.validate_model_structure(model_path)
        if not structure_check['valid']:
            return {
                'success': False,
                'error': 'Estrutura do modelo inválida',
                'structure_issues': structure_check['issues']
            }
        
        try:
            # Carregar modelo usando VideoModelTrainer
            import sys
            import os
            sys.path.append(os.path.dirname(os.path.abspath(__file__)))
            from model_trainer import VideoModelTrainer
            
            trainer = VideoModelTrainer()
            
            # Carregar modelo
            if not trainer.load_model(model_path):
                return {
                    'success': False,
                    'error': 'Falha ao carregar o modelo'
                }
            
            # Avaliar no dataset
            print(f"🎯 Avaliando modelo no dataset...")
            results = trainer.evaluate_on_dataset(dataset_path, max_videos_per_class)
            
            if not results.get('success'):
                return {
                    'success': False,
                    'error': results.get('error', 'Erro na avaliação')
                }
            
            # Formatar resultados
            performance_results = {
                'success': True,
                'model_info': structure_check['info'],
                'performance': {
                    'accuracy': results['accuracy'],
                    'total_videos': results['total_videos'],
                    'classes_tested': results['classes_tested'],
                    'predictions': results['predictions'],
                    'classification_report': results['classification_report']
                },
                'recommendations': []
            }
            
            # Gerar recomendações
            accuracy = results['accuracy']
            if accuracy < 0.7:
                performance_results['recommendations'].append("Acurácia baixa - considere mais dados de treino ou ajuste de hiperparâmetros")
            elif accuracy > 0.95:
                performance_results['recommendations'].append("Possível overfitting - validar com dataset independente")
            
            if results['total_videos'] < 20:
                performance_results['recommendations'].append("Dataset de teste pequeno - considere mais amostras para validação confiável")
            
            print(f"\n📊 Resultados:")
            print(f"   🎯 Acurácia: {accuracy:.1%}")
            print(f"   📈 Vídeos testados: {results['total_videos']}")
            print(f"   📋 Classes: {', '.join(results['classes_tested'])}")
            
            return performance_results
            
        except Exception as e:
            print(f"❌ Erro na análise: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def compare_models(self, model1_path, model2_path, dataset_path, max_videos_per_class=10):
        """
        Compara performance entre dois modelos
        
        Args:
            model1_path: Caminho do primeiro modelo
            model2_path: Caminho do segundo modelo
            dataset_path: Caminho do dataset de teste
            max_videos_per_class: Máximo de vídeos por classe para teste
            
        Returns:
            dict: Comparação entre os modelos
        """
        print(f"\n🆚 === COMPARAÇÃO DE MODELOS ===")
        
        # Analisar modelo 1
        print(f"\n📊 Analisando Modelo 1...")
        results1 = self.analyze_model_performance(model1_path, dataset_path, max_videos_per_class)
        
        # Analisar modelo 2  
        print(f"\n📊 Analisando Modelo 2...")
        results2 = self.analyze_model_performance(model2_path, dataset_path, max_videos_per_class)
        
        # Comparar resultados
        comparison = {
            'model1': {
                'path': model1_path,
                'results': results1
            },
            'model2': {
                'path': model2_path, 
                'results': results2
            },
            'winner': None,
            'differences': []
        }
        
        # Determinar vencedor (se ambos tiveram sucesso)
        if results1.get('success') and results2.get('success'):
            acc1 = results1.get('performance', {}).get('accuracy', 0)
            acc2 = results2.get('performance', {}).get('accuracy', 0)
            
            if acc1 > acc2:
                comparison['winner'] = 'model1'
                comparison['differences'].append(f"Modelo 1 tem melhor acurácia: {acc1:.4f} vs {acc2:.4f}")
            elif acc2 > acc1:
                comparison['winner'] = 'model2'
                comparison['differences'].append(f"Modelo 2 tem melhor acurácia: {acc2:.4f} vs {acc1:.4f}")
            else:
                comparison['winner'] = 'tie'
                comparison['differences'].append(f"Ambos têm acurácia similar: {acc1:.4f}")
        
        return comparison
    
    def generate_diagnosis_report(self, model_path, dataset_path=None):
        """
        Gera relatório completo de diagnóstico
        
        Args:
            model_path: Caminho do modelo
            dataset_path: Caminho do dataset (opcional)
            
        Returns:
            dict: Relatório completo
        """
        print(f"\n📋 === RELATÓRIO DE DIAGNÓSTICO ===")
        
        report = {
            'model_path': model_path,
            'timestamp': None,  # Seria implementado
            'structure_check': self.validate_model_structure(model_path),
            'performance_analysis': None,
            'recommendations': []
        }
        
        # Análise de performance se dataset fornecido
        if dataset_path:
            report['performance_analysis'] = self.analyze_model_performance(model_path, dataset_path)
        
        # Gerar recomendações
        if not report['structure_check']['valid']:
            report['recommendations'].append("Corrigir problemas na estrutura do modelo")
        
        if dataset_path and report['performance_analysis']:
            if not report['performance_analysis'].get('success'):
                report['recommendations'].append("Investigar problemas na análise de performance")
        
        return report


def list_available_models(models_dir="models"):
    """
    Lista todos os modelos disponíveis
    
    Args:
        models_dir: Diretório dos modelos
        
    Returns:
        list: Lista de modelos encontrados
    """
    if not os.path.exists(models_dir):
        print(f"📂 Diretório de modelos não encontrado: {models_dir}")
        return []
    
    models = []
    for item in os.listdir(models_dir):
        model_path = os.path.join(models_dir, item)
        if os.path.isdir(model_path):
            # Verificar se tem arquivos de modelo
            classifier_file = os.path.join(model_path, "classifier.pkl")
            if os.path.exists(classifier_file):
                models.append({
                    'name': item,
                    'path': model_path,
                    'has_config': os.path.exists(os.path.join(model_path, "config.json"))
                })
    
    return models


if __name__ == "__main__":
    """Interface de linha de comando para validação de modelos"""
    parser = argparse.ArgumentParser(description="Validador de Modelos para Classificação de Vídeos")
    parser.add_argument('command', choices=['validate', 'analyze', 'compare', 'list'], 
                       help='Comando a executar')
    parser.add_argument('--model', 
                       help='Caminho do modelo')
    parser.add_argument('--model2',
                       help='Caminho do segundo modelo (para comparação)')
    parser.add_argument('--dataset',
                       help='Caminho do dataset de teste')
    parser.add_argument('--models-dir', default='models',
                       help='Diretório dos modelos')
    parser.add_argument('--max-videos', type=int, default=10,
                       help='Máximo de vídeos por classe')
    
    args = parser.parse_args()
    
    validator = ModelValidator()
    
    if args.command == 'list':
        print("📋 Listando modelos disponíveis...")
        models = list_available_models(args.models_dir)
        
        if not models:
            print("❌ Nenhum modelo encontrado")
        else:
            print(f"✅ Encontrados {len(models)} modelos:")
            for model in models:
                status = "✅ Completo" if model['has_config'] else "⚠️ Sem config"
                print(f"  🤖 {model['name']}: {status}")
    
    elif args.command == 'validate':
        if not args.model:
            print("❌ --model é obrigatório para validação")
            exit(1)
        
        print("🔍 Validando modelo...")
        result = validator.validate_model_structure(args.model)
        
        if result['valid']:
            print("✅ Modelo válido!")
        else:
            print("❌ Modelo inválido!")
            exit(1)
    
    elif args.command == 'analyze':
        if not args.model:
            print("❌ --model é obrigatório para análise")
            exit(1)
            
        if not args.dataset:
            print("❌ --dataset é obrigatório para análise")
            exit(1)
        
        print("📊 Analisando performance...")
        result = validator.analyze_model_performance(args.model, args.dataset, args.max_videos)
        
        if result.get('success'):
            print("✅ Análise concluída!")
        else:
            print(f"❌ Falha na análise: {result.get('error', 'Erro desconhecido')}")
            exit(1)
    
    elif args.command == 'compare':
        if not args.model or not args.model2:
            print("❌ --model e --model2 são obrigatórios para comparação")
            exit(1)
            
        if not args.dataset:
            print("❌ --dataset é obrigatório para comparação")
            exit(1)
        
        print("🆚 Comparando modelos...")
        result = validator.compare_models(args.model, args.model2, args.dataset, args.max_videos)
        
        winner = result.get('winner')
        if winner:
            print(f"🏆 Vencedor: {winner}")
        else:
            print("❌ Não foi possível determinar vencedor")