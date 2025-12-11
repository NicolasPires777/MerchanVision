"""
Dataset Creator Module

Provides dataset creation functions for video classification projects.
"""

import os
import argparse


def create_sample_dataset(custom_classes=None, base_path="video_classification", dataset_name="sample_dataset"):
    """Cria estrutura de exemplo para dataset com classes customizáveis"""
    dataset_path = os.path.join(base_path, dataset_name)
    
    # Usar classes padrão ou customizadas
    if custom_classes is None:
        classes = ['break', 'conteudo', 'merchan']
    else:
        classes = custom_classes
    
    # Criar pastas para cada classe
    for class_name in classes:
        os.makedirs(os.path.join(dataset_path, class_name), exist_ok=True)
        print(f"📁 Pasta criada: {class_name}/")
    
    print(f"📁 Estrutura de dataset criada em: {dataset_path}")
    
    return dataset_path


if __name__ == "__main__":
    """Interface de linha de comando para criação de datasets"""
    parser = argparse.ArgumentParser(description="Criador de Dataset para Classificação de Vídeos")
    parser.add_argument('command', choices=['create'], 
                       help='Comando a executar')
    parser.add_argument('--name', default='sample_dataset',
                       help='Nome do dataset')
    parser.add_argument('--base-path', default='video_classification',
                       help='Caminho base onde criar o dataset')
    parser.add_argument('--classes', nargs='+', default=None,
                       help='Lista de classes (padrão: break, conteudo, merchan)')
    
    args = parser.parse_args()
    
    if args.command == 'create':
        print("🏗️ Criando estrutura de dataset...")
        classes = args.classes if args.classes else ['break', 'conteudo', 'merchan']
        dataset_path = create_sample_dataset(
            custom_classes=classes, 
            base_path=args.base_path,
            dataset_name=args.name
        )
        print(f"✅ Dataset criado com sucesso em: {dataset_path}")