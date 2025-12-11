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

class VideoClassificationManager:
    """Gerenciador de classificação de vídeo"""
    
    def __init__(self):
        self.base_dir = "/home/nicolas/Zedia/Others/AI-Detector"
        self.models_dir = os.path.join(self.base_dir, "models")
        self.datasets_dir = os.path.join(self.base_dir, "datasets")
        self.scripts_dir = os.path.join(self.base_dir, "scripts")
        
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
        """Lista todos os modelos disponíveis"""
        models = [d for d in os.listdir(self.models_dir) 
                 if os.path.isdir(os.path.join(self.models_dir, d))]
        
        print("🤖 Modelos disponíveis:")
        if not models:
            print("  Nenhum modelo encontrado")
            return []
        
        for model in models:
            model_path = os.path.join(self.models_dir, model)
            
            # Verificar arquivos do modelo
            classifier_exists = os.path.exists(os.path.join(model_path, "classifier.pkl"))
            
            if classifier_exists:
                status = "✅ Treinado"
            else:
                status = "❌ Incompleto"
            
            print(f"  🎯 {model}: {status}")
        
        return models
    
    def list_datasets(self):
        """Lista todos os datasets disponíveis"""
        datasets = [d for d in os.listdir(self.datasets_dir) 
                   if os.path.isdir(os.path.join(self.datasets_dir, d)) and 
                   d not in ['__pycache__']]
        
        print("📁 Datasets disponíveis:")
        if not datasets:
            print("  Nenhum dataset encontrado")
            return []
        
        for dataset in datasets:
            dataset_path = os.path.join(self.datasets_dir, dataset)
            
            # Detectar classes dinamicamente
            try:
                classes_info = {}
                total_videos = 0
                
                # Verificar todas as pastas dentro do dataset
                for item in os.listdir(dataset_path):
                    class_path = os.path.join(dataset_path, item)
                    if os.path.isdir(class_path) and item != '__pycache__':
                        # Contar vídeos nesta classe
                        video_count = len([f for f in os.listdir(class_path) 
                                         if f.endswith(('.mp4', '.avi', '.mov', '.mkv'))])
                        if video_count > 0:  # Só mostrar classes que têm vídeos
                            classes_info[item] = video_count
                            total_videos += video_count
                
                if classes_info:
                    # Criar string das classes dinamicamente
                    classes_str = ", ".join([f"{class_name.capitalize()}:{count}" 
                                           for class_name, count in sorted(classes_info.items())])
                    
                    # Determinar status de balanceamento
                    if len(classes_info) >= 2:
                        counts = list(classes_info.values())
                        min_count, max_count = min(counts), max(counts)
                        balance_status = "⚖️ Balanceado" if (max_count - min_count) <= 5 else "⚠️ Desbalanceado"
                    else:
                        balance_status = "➖ Única classe"
                    
                    print(f"  📊 {dataset}: {total_videos} vídeos ({classes_str}) {balance_status}")
                else:
                    print(f"  📊 {dataset}: 0 vídeos (vazio)")
                    
            except Exception as e:
                print(f"  📊 {dataset}: Status desconhecido (erro: {e})")
        
        return datasets
    
    def create_dataset(self, dataset_name):
        """Cria estrutura de dataset"""
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
        
        # Criar estrutura de pastas
        for class_name in classes:
            class_path = os.path.join(dataset_path, class_name)
            os.makedirs(class_path, exist_ok=True)
            print(f"  ✅ {class_path}")
        
        print(f"\n🎯 Dataset '{dataset_name}' criado!")
        print(f"📂 Diretório: {dataset_path}")
        print(f"\n📝 Próximos passos:")
        print(f"  1. Adicione vídeos (.mp4, .avi, .mov, .mkv) nas pastas correspondentes:")
        
        for class_name in classes:
            print(f"     - {dataset_path}/{class_name}/")
        
        print(f"  2. Execute o treinamento do modelo")
        
        return dataset_path
    
    def train_model(self, model_name, dataset_name):
        """Treina um modelo"""
        dataset_path = os.path.join(self.datasets_dir, dataset_name)
        model_path = os.path.join(self.models_dir, model_name)
        
        if not os.path.exists(dataset_path):
            print(f"❌ Dataset não encontrado: {dataset_path}")
            return False
        
        print(f"🚀 Treinando modelo '{model_name}' com dataset '{dataset_name}'...")
        
        try:
            subprocess.run([
                self.python_cmd, 'scripts/video_classifier_simple.py', 'train',
                '--dataset', dataset_path,
                '--save', model_path
            ], check=True)
            print(f"✅ Modelo treinado com sucesso!")
            return True
        except subprocess.CalledProcessError as e:
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
            subprocess.run([
                self.python_cmd, 'scripts/video_classifier_simple.py', 'predict',
                '--video', video_path,
                '--load', model_path
            ], check=True)
        except subprocess.CalledProcessError as e:
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
            script_name = 'scripts/realtime_hybrid_classifier.py'
            print("🤖 Usando classificador híbrido (detecta indicadores visuais)")
        else:
            script_name = 'scripts/realtime_classifier_local.py'
            print("📸 Usando classificador tradicional (apenas imagem)")
        
        # Verificar se o script existe
        if not os.path.exists(script_name):
            print(f"❌ Script não encontrado: {script_name}")
            if classifier_choice == '1':
                print("💡 Execute: python3 scripts/realtime_hybrid_classifier.py --help")
                print("💡 Ou use a opção 2 (classificação tradicional)")
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
            subprocess.run([
                self.python_cmd, 'scripts/model_diagnosis.py',
                '--model1', model_path,
                '--dataset1', dataset_path,
                '--max-videos', str(max_videos)
            ], check=True)
        except subprocess.CalledProcessError as e:
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
            subprocess.run([
                self.python_cmd, 'scripts/model_diagnosis.py',
                '--model1', model1_path,
                '--model2', model2_path,
                '--dataset1', dataset1_path,
                '--dataset2', dataset2_path,
                '--max-videos', str(max_videos)
            ], check=True)
        except subprocess.CalledProcessError as e:
            print(f"❌ Erro na comparação: {e}")
    
    def main_menu(self):
        """Menu principal"""
        while True:
            print(f"\n🎬 === Video Classification Manager ===")
            print("1. 📋 Listar modelos")
            print("2. 📁 Listar datasets") 
            print("3. 🆕 Criar novo dataset")
            print("4. 🚀 Treinar modelo")
            print("5. 🎯 Classificar vídeo único")
            print("6. 🔴 Classificação em tempo real")
            print("7. 📊 Validar modelo")
            print("8. 🆚 Comparar modelos")
            print("0. Sair")
            
            choice = input("\nEscolha uma opção (0-8): ").strip()
            
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
                model_name = input("Nome do modelo: ").strip()
                dataset_name = input("Nome do dataset: ").strip()
                if model_name and dataset_name:
                    self.train_model(model_name, dataset_name)
            elif choice == '5':
                model_name = input("Nome do modelo: ").strip()
                video_path = input("Caminho do vídeo: ").strip()
                if model_name and video_path:
                    self.classify_video(model_name, video_path)
            elif choice == '6':
                self.realtime_classification()
            elif choice == '7':
                self.validate_model()
            elif choice == '8':
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