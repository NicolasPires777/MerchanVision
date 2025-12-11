#!/usr/bin/env python3
"""
🎬 Video Classification Manager
Sistema completo para classificação de vídeos em Break vs Conteúdo vs Merchan
"""

import sys
import os
import argparse
import subprocess
from pathlib import Path

# Import dataset and model manager functions
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from dataset_manager import (
    create_sample_dataset,
    load_video_dataset,
    validate_dataset,
    get_dataset_statistics,
    get_dataset_basic_info
)
from model_manager import VideoModelTrainer, ModelValidator, list_available_models
from config import config

class VideoClassificationManager:
    """Gerenciador de classificação de vídeo"""
    
    def __init__(self):
        self.base_dir = "/home/nicolas/Zedia/Others/AI-Detector"
        self.models_dir = os.path.join(self.base_dir, "models")
        self.datasets_dir = os.path.join(self.base_dir, "datasets")
        
        # Detectar Python correto (ambiente virtual)
        self.python_cmd = self._detect_python_command()
        
        # Criar diretórios se não existirem
        os.makedirs(self.models_dir, exist_ok=True)
        os.makedirs(self.datasets_dir, exist_ok=True)
    
    def _detect_python_command(self):
        """Detecta o comando Python correto (ambiente virtual se disponível)"""
        venv_python = os.path.join(self.base_dir, ".venv", "bin", "python")
        if os.path.exists(venv_python):
            print(f"🐍 Usando ambiente virtual: {venv_python}")
            return venv_python
        else:
            print("⚠️ Ambiente virtual não encontrado, usando python3")
            print("💡 Recomenda-se ativar o ambiente virtual: source .venv/bin/activate")
            return "python3"
    
    def list_models(self):
        """Lista todos os modelos disponíveis usando model_manager"""
        models = list_available_models(self.models_dir)
        
        print("🤖 Modelos disponíveis:")
        if not models:
            print("  Nenhum modelo encontrado")
            return []
        
        for model in models:
            status = "✅ Completo" if model['has_config'] else "⚠️ Sem config"
            print(f"  🎯 {model['name']}: {status}")
        
        # Retornar apenas os nomes para compatibilidade
        return [model['name'] for model in models]
    
    def list_datasets(self):
        """Lista todos os datasets disponíveis usando dataset_manager"""
        datasets = [d for d in os.listdir(self.datasets_dir) 
                   if os.path.isdir(os.path.join(self.datasets_dir, d)) and 
                   d not in ['__pycache__']]
        
        print("📁 Datasets disponíveis:")
        if not datasets:
            print("  Nenhum dataset encontrado")
            return []
        
        for dataset in datasets:
            dataset_path = os.path.join(self.datasets_dir, dataset)
            
            try:
                # Usar dataset_manager para obter informações básicas (sem verbose output)
                info = get_dataset_basic_info(dataset_path)
                total_videos = info['total_videos']
                classes = info['classes']
                
                if total_videos > 0:
                    # Criar string das classes dinamicamente
                    classes_str = ", ".join([f"{class_name.capitalize()}:{info['class_counts'][class_name]}" 
                                           for class_name in classes])
                    
                    print(f"  📊 {dataset}: {total_videos} vídeos ({classes_str})")
                else:
                    print(f"  📊 {dataset}: 0 vídeos (vazio)")
                    
            except Exception as e:
                print(f"  📊 {dataset}: Status desconhecido (erro: {e})")
        
        return datasets
    
    def create_dataset(self, dataset_name):
        """Cria estrutura de dataset usando dataset_manager"""
        dataset_path = os.path.join(self.datasets_dir, dataset_name)
        
        print(f"📁 Criando dataset: {dataset_name}")
        
        # Perguntar as classes ao usuário
        print("\n🎯 Definir classes do dataset:")
        print("💡 Exemplos comuns:")
        print("   - break, conteudo, merchan")
        print("   - comercial, programa, intervalo")
        print("   - intro, conteudo, creditos")
        print("   - ou qualquer combinação personalizada")
        
        classes_input = input("\nDigite as classes separadas por vírgula: ").strip()
        
        if not classes_input:
            print("⚠️ Nenhuma classe informada, usando padrão: break, conteudo, merchan")
            classes = ['break', 'conteudo', 'merchan']
        else:
            # Processar entrada do usuário
            classes = [cls.strip().lower() for cls in classes_input.split(',') if cls.strip()]
            
            if not classes:
                print("❌ Classes inválidas, usando padrão: break, conteudo, merchan")
                classes = ['break', 'conteudo', 'merchan']
        
        print(f"\n📂 Classes a criar: {', '.join(classes)}")
        
        try:
            # Usar dataset_manager para criar o dataset
            dataset_path = create_sample_dataset(
                custom_classes=classes, 
                base_path=self.datasets_dir,
                dataset_name=dataset_name
            )
            
            print(f"\n🎯 Dataset '{dataset_name}' criado!")
            print(f"📂 Diretório: {dataset_path}")
            print(f"\n📝 Próximos passos:")
            print(f"  1. Adicione vídeos (.mp4, .avi, .mov, .mkv) nas pastas correspondentes:")
            
            for class_name in classes:
                print(f"     - {dataset_path}/{class_name}/")
            
            print(f"  2. Execute o treinamento do modelo")
            print(f"  3. Use 'python3 dataset_manager/dataset_creator.py validate --path {dataset_path}' para validar")
            
        except Exception as e:
            print(f"❌ Erro ao criar dataset: {e}")
            return None
        
        return dataset_path
    
    def validate_dataset_interactive(self, dataset_name=None):
        """Valida dataset usando dataset_manager"""
        if dataset_name is None:
            datasets = self.list_datasets()
            if not datasets:
                print("❌ Nenhum dataset encontrado!")
                return False
            
            print(f"\nDatasets disponíveis: {', '.join(datasets)}")
            dataset_name = input("Nome do dataset para validar: ").strip()
            
            if not dataset_name or dataset_name not in datasets:
                print("❌ Dataset inválido!")
                return False
        
        dataset_path = os.path.join(self.datasets_dir, dataset_name)
        
        print(f"\n🔍 Validando dataset '{dataset_name}'...")
        
        try:
            # Usar dataset_manager para validação
            is_valid = validate_dataset(dataset_path)
            
            if is_valid:
                print(f"✅ Dataset '{dataset_name}' está pronto para treinamento!")
                
                # Mostrar estatísticas
                stats = get_dataset_statistics(dataset_path)
                print(f"\n📊 Resumo:")
                print(f"  🎬 Total: {stats['total_videos']} vídeos")
                print(f"  ⚖️ Balanceamento: {stats['balance_ratio']:.2f}")
                
                for class_name, info in stats['classes'].items():
                    percentage = (info['count'] / stats['total_videos'] * 100) if stats['total_videos'] > 0 else 0
                    print(f"  📋 {class_name.capitalize()}: {info['count']} vídeos ({percentage:.1f}%)")
                
                return True
            else:
                print(f"❌ Dataset '{dataset_name}' precisa de correções!")
                return False
                
        except Exception as e:
            print(f"❌ Erro na validação: {e}")
            return False

    def train_model(self, model_name, dataset_name):
        """Treina um modelo"""
        dataset_path = os.path.join(self.datasets_dir, dataset_name)
        model_path = os.path.join(self.models_dir, model_name)
        
        if not os.path.exists(dataset_path):
            print(f"❌ Dataset não encontrado: {dataset_path}")
            return False
        
        print(f"🚀 Treinando modelo '{model_name}' com dataset '{dataset_name}'...")
        
        try:
            # Usar model_manager ao invés de scripts
            trainer = VideoModelTrainer()
            
            # Treinar diretamente do dataset path
            model, scaler = trainer.train_from_dataset(dataset_path)
            
            # Salvar o modelo (o trainer já tem modelo e scaler internos)
            trainer.save_model(model_path)
            
            print(f"✅ Modelo treinado com sucesso!")
            return True
        except Exception as e:
            print(f"❌ Erro no treinamento: {e}")
            return False
    
    def classify_video(self, model_name, video_path):
        """Classifica um vídeo"""
        model_path = os.path.join(self.models_dir, model_name)
        
        if not os.path.exists(model_path):
            print(f"❌ Modelo não encontrado: {model_path}")
            return
        
        if not os.path.exists(video_path):
            print(f"❌ Vídeo não encontrado: {video_path}")
            return
        
        print(f"🎬 Classificando vídeo: {video_path}")
        
        try:
            # Usar model_manager ao invés de scripts
            trainer = VideoModelTrainer()
            trainer.load_model(model_path)
            
            result = trainer.predict_video(video_path)
            
            print(f"📊 Resultado da classificação:")
            print(f"   🎯 Classe: {result['predicted_class']}")
            print(f"   📈 Confiança: {result['confidence']:.1%}")
            
        except Exception as e:
            print(f"❌ Erro na classificação: {e}")
    
    def realtime_classification(self):
        """Menu de classificação em tempo real"""
        print("\n🔴 === Classificação em Tempo Real ===")
        print("Escolha o tipo de classificação:")
        print("1. 🧠 Classificação Híbrida (RECOMENDADO)")
        print("   🔍 Detecta QR-codes, telefones, preços, etc.")
        print("   📊 Combina análise de imagem + indicadores visuais")
        print("2. 📸 Classificação Tradicional")
        print("   🖼️ Apenas análise de imagem (modelo original)")
        
        classifier_choice = input("\nEscolha o classificador (1-2): ").strip()
        
        print("\nTipo de fonte disponível:")
        print("1. 📹 Arquivo de vídeo local")
        
        rt_choice = "1"  # Apenas arquivo de vídeo local disponível
        
        # Selecionar modelo
        models = self.list_models()
        if not models:
            print("❌ Nenhum modelo treinado encontrado!")
            return
        
        print(f"\nModelos disponíveis: {', '.join(models)}")
        model_name = input("Nome do modelo (padrão: Alpha-v4): ").strip()
        if not model_name:
            model_name = "Alpha-v4"
        
        model_path = os.path.join(self.models_dir, model_name)
        if not os.path.exists(model_path):
            print(f"❌ Modelo não encontrado: {model_path}")
            return
        
        # Determinar script a usar
        if classifier_choice == '1':
            script_name = 'classifier/realtime_hybrid_classifier.py'
            print("🤖 Usando classificador híbrido (detecta indicadores visuais)")
        else:
            script_name = 'classifier/realtime_basic_classifier.py'
            print("📸 Usando classificador básico (apenas imagem)")
        
        # Verificar se o script existe
        if not os.path.exists(script_name):
            print(f"❌ Script não encontrado: {script_name}")
            if classifier_choice == '1':
                print("💡 Execute: python3 classifier/realtime_hybrid_classifier.py --help")
                print("💡 Ou use a opção 2 (classificação básica)")
            return

        if rt_choice == '1':
            # Arquivo local
            video_path = input("Caminho do arquivo de vídeo: ").strip()
            if video_path:
                try:
                    if classifier_choice == '1':
                        # Classificador híbrido
                        subprocess.run([
                            self.python_cmd, script_name,
                            '--model', model_path,
                            '--video', video_path
                        ], check=True)
                    else:
                        # Classificador tradicional
                        subprocess.run([
                            self.python_cmd, script_name,
                            '--model', model_path,
                            '--source', video_path
                        ], check=True)
                except subprocess.CalledProcessError as e:
                    print(f"❌ Erro na classificação: {e}")
        else:
            print("❌ Opção inválida")
            # Webcam
            try:
                if classifier_choice == '1':
                    # Classificador híbrido
                    subprocess.run([
                        self.python_cmd, script_name,
                        '--model', model_path,
                        '--video', '0'
                    ], check=True)
                else:
                    # Classificador tradicional
                    subprocess.run([
                        self.python_cmd, script_name,
                        '--model', model_path,
                        '--source', '0'
                    ], check=True)
            except subprocess.CalledProcessError as e:
                print(f"❌ Erro na classificação: {e}")
                
    def validate_model(self):
        """Valida um modelo"""
        print("\n📊 === Validação de Modelo ===")
        
        # Listar modelos e datasets
        models = self.list_models()
        datasets = self.list_datasets()
        
        if not models:
            print("❌ Nenhum modelo encontrado!")
            return
        
        model_name = input(f"Modelo para validar ({', '.join(models)}): ").strip()
        if not model_name:
            print("❌ Nome do modelo é obrigatório")
            return
        
        model_path = os.path.join(self.models_dir, model_name)
        if not os.path.exists(model_path):
            print(f"❌ Modelo não encontrado: {model_path}")
            return
        
        # Tentar detectar dataset automaticamente
        dataset_name = model_name  # Convenção: mesmo nome
        if dataset_name not in datasets:
            print(f"⚠️ Dataset automático não encontrado: {dataset_name}")
            if datasets:
                dataset_name = input(f"Dataset para usar ({', '.join(datasets)}): ").strip()
            else:
                print("❌ Nenhum dataset encontrado!")
                return
        
        dataset_path = os.path.join(self.datasets_dir, dataset_name)
        
        max_videos = input("Máximo de vídeos por classe (padrão: 10): ").strip()
        max_videos = int(max_videos) if max_videos else 10
        
        print(f"🔍 Validando modelo {model_name} com dataset {dataset_name}...")
        try:
            # Usar model_manager ao invés de scripts
            validator = ModelValidator()
            result = validator.analyze_model_performance(model_path, dataset_path, max_videos)
            
            if result.get('success'):
                print("✅ Validação concluída com sucesso!")
                
                performance = result.get('performance', {})
                if 'accuracy' in performance:
                    print(f"📊 Acurácia: {performance['accuracy']:.1%}")
                
                if result.get('recommendations'):
                    print("\n💡 Recomendações:")
                    for rec in result['recommendations']:
                        print(f"   • {rec}")
            else:
                print(f"❌ Falha na validação: {result.get('error', 'Erro desconhecido')}")
                
        except Exception as e:
            print(f"❌ Erro na validação: {e}")
    
    def compare_models(self):
        """Compara dois modelos"""
        print("\n🆚 === Comparação de Modelos ===")
        
        models = self.list_models()
        datasets = self.list_datasets()
        
        if len(models) < 2:
            print("❌ Necessário pelo menos 2 modelos para comparar!")
            return
        
        print(f"Modelos disponíveis: {', '.join(models)}")
        model1_name = input("Primeiro modelo: ").strip()
        model2_name = input("Segundo modelo: ").strip()
        
        if not model1_name or not model2_name:
            print("❌ Ambos os nomes dos modelos são obrigatórios")
            return
        
        model1_path = os.path.join(self.models_dir, model1_name)
        model2_path = os.path.join(self.models_dir, model2_name)
        
        # Detectar datasets
        dataset1_name = model1_name
        dataset2_name = model2_name
        
        if dataset1_name not in datasets:
            dataset1_name = input(f"Dataset para {model1_name}: ").strip()
        if dataset2_name not in datasets:
            dataset2_name = input(f"Dataset para {model2_name}: ").strip()
        
        dataset1_path = os.path.join(self.datasets_dir, dataset1_name)
        dataset2_path = os.path.join(self.datasets_dir, dataset2_name)
        
        max_videos = input("Máximo de vídeos por classe (padrão: 8): ").strip()
        max_videos = int(max_videos) if max_videos else 8
        
        print(f"🆚 Comparando {model1_name} vs {model2_name}...")
        try:
            # Usar model_manager ao invés de scripts
            validator = ModelValidator()
            
            # Para simplificar, usar o mesmo dataset para ambos
            dataset_path = dataset1_path  # Ou escolher o melhor
            
            result = validator.compare_models(model1_path, model2_path, dataset_path, max_videos)
            
            if result.get('model1', {}).get('results', {}).get('success') and result.get('model2', {}).get('results', {}).get('success'):
                winner = result.get('winner')
                if winner:
                    winner_name = model1_name if winner == 'model1' else model2_name
                    print(f"🏆 Vencedor: {winner_name}")
                
                differences = result.get('differences', [])
                if differences:
                    print("\n📊 Diferenças encontradas:")
                    for diff in differences:
                        print(f"   • {diff}")
                        
            else:
                print("❌ Falha na comparação - um ou ambos os modelos tiveram problemas")
                
        except Exception as e:
            print(f"❌ Erro na comparação: {e}")
    
    def main_menu(self):
        """Menu principal"""
        while True:
            print(f"\n🎬 === Video Classification Manager ===")
            print("1. 📋 Listar modelos")
            print("2. 📁 Listar datasets") 
            print("3. 🆕 Criar novo dataset")
            print("4. 📁 Validar dataset")
            print("5. 🚀 Treinar modelo")
            print("6. 🎯 Classificar vídeo único")
            print("7. 🔴 Classificação em tempo real")
            print("8. 📊 Validar modelo")
            print("9. 🆚 Comparar modelos")
            print("0. Sair")
            
            choice = input("\nEscolha uma opção (0-9): ").strip()
            
            if choice == '0':
                print("👋 Saindo...")
                break
            elif choice == '1':
                self.list_models()
            elif choice == '2':
                self.list_datasets()
            elif choice == '3':
                dataset_name = input("Nome do novo dataset: ").strip()
                if dataset_name:
                    self.create_dataset(dataset_name)
            elif choice == '4':
                self.validate_dataset_interactive()
            elif choice == '5':
                model_name = input("Nome do modelo: ").strip()
                dataset_name = input("Nome do dataset: ").strip()
                if model_name and dataset_name:
                    self.train_model(model_name, dataset_name)
            elif choice == '6':
                model_name = input("Nome do modelo: ").strip()
                video_path = input("Caminho do vídeo: ").strip()
                if model_name and video_path:
                    self.classify_video(model_name, video_path)
            elif choice == '7':
                self.realtime_classification()
            elif choice == '8':
                self.validate_model()
            elif choice == '9':
                self.compare_models()
            else:
                print("❌ Opção inválida")

def main():
    """Função principal"""
    parser = argparse.ArgumentParser(description="Video Classification Manager")
    parser.add_argument('command', nargs='?', choices=['menu', 'list-models', 'list-datasets', 'create', 'train', 'classify', 'validate', 'compare'], 
                       help='Comando a executar (padrão: menu)')
    parser.add_argument('--model', help='Nome do modelo')
    parser.add_argument('--dataset', help='Nome do dataset')
    parser.add_argument('--video', help='Caminho do vídeo')
    
    args = parser.parse_args()
    
    manager = VideoClassificationManager()
    
    if args.command == 'list-models':
        manager.list_models()
    elif args.command == 'list-datasets':
        manager.list_datasets()
    elif args.command == 'create':
        if args.dataset:
            manager.create_dataset(args.dataset)
        else:
            print("❌ Especifique --dataset")
    elif args.command == 'train':
        if args.model and args.dataset:
            manager.train_model(args.model, args.dataset)
        else:
            print("❌ Especifique --model e --dataset")
    elif args.command == 'classify':
        if args.model and args.video:
            manager.classify_video(args.model, args.video)
        else:
            print("❌ Especifique --model e --video")
    elif args.command == 'validate':
        manager.validate_model()
    elif args.command == 'compare':
        manager.compare_models()
    else:
        # Menu interativo por padrão
        manager.main_menu()

if __name__ == "__main__":
    main()