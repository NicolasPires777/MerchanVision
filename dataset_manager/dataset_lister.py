"""
Dataset Lister Module

Provides listing and detection functions for video datasets.
"""

import os
from pathlib import Path


def detect_dataset_classes(dataset_path):
    """
    Detecta automaticamente as classes que existem no dataset
    
    Args:
        dataset_path: Caminho para o dataset
        
    Returns:
        list: Lista de classes encontradas
    """
    if not os.path.exists(dataset_path):
        return []
    
    classes = []
    video_extensions = ('.mp4', '.avi', '.mov', '.mkv')
    
    for item in os.listdir(dataset_path):
        class_path = os.path.join(dataset_path, item)
        if os.path.isdir(class_path) and item != '__pycache__':
            # Verificar se tem vídeos nesta pasta
            has_videos = any(f.lower().endswith(video_extensions) 
                           for f in os.listdir(class_path) 
                           if os.path.isfile(os.path.join(class_path, f)))
            if has_videos:
                classes.append(item)
    
    return sorted(classes)


def load_video_dataset(dataset_path, classes=None, verbose=True):
    """
    Carrega dataset de vídeos e retorna lista de arquivos organizados por classe
    
    Args:
        dataset_path: Caminho para o dataset
        classes: Lista de classes esperadas (None = detectar automaticamente)
        verbose: Se deve imprimir informações detalhadas
        
    Returns:
        dict: {'class_name': [list_of_video_files], ...}
    """
    if verbose:
        print(f"📁 Carregando dataset de: {dataset_path}")
    
    # Detectar classes automaticamente se não especificadas
    if classes is None:
        classes = detect_dataset_classes(dataset_path)
    
    dataset_info = {}
    
    for class_name in classes:
        class_path = os.path.join(dataset_path, class_name)
        
        if not os.path.exists(class_path):
            if verbose:
                print(f"⚠️ Pasta não encontrada: {class_path}")
            dataset_info[class_name] = []
            continue
        
        # Encontrar vídeos
        video_extensions = ['*.mp4', '*.avi', '*.mov', '*.mkv']
        video_files = []
        for ext in video_extensions:
            video_files.extend(Path(class_path).glob(ext))
        
        # Converter para strings
        video_files = [str(f) for f in video_files]
        dataset_info[class_name] = video_files
        
        if verbose:
            print(f"🎬 Encontrados {len(video_files)} vídeos de '{class_name}'")
    
    if verbose:
        total_videos = sum(len(files) for files in dataset_info.values())
        print(f"\n📊 Resumo do dataset:")
        print(f"  🎬 Total de vídeos: {total_videos}")
        
        for class_name, files in dataset_info.items():
            percentage = (len(files) / total_videos * 100) if total_videos > 0 else 0
            print(f"  📋 {class_name}: {len(files)} vídeos ({percentage:.1f}%)")
    
    return dataset_info


def get_dataset_basic_info(dataset_path):
    """
    Retorna informações básicas do dataset sem output verboso
    
    Args:
        dataset_path: Caminho para o dataset
        
    Returns:
        dict: Informações básicas do dataset
    """
    # Detectar classes automaticamente
    classes = detect_dataset_classes(dataset_path)
    
    # Carregar dataset sem output verboso
    dataset_info = load_video_dataset(dataset_path, classes, verbose=False)
    
    basic_info = {
        'classes': classes,
        'total_videos': 0,
        'class_counts': {}
    }
    
    for class_name, files in dataset_info.items():
        count = len(files)
        basic_info['class_counts'][class_name] = count
        basic_info['total_videos'] += count
    
    return basic_info


if __name__ == "__main__":
    """Interface de linha de comando para listagem de datasets"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Listador de Dataset para Classificação de Vídeos")
    parser.add_argument('command', choices=['list', 'info', 'classes'], 
                       help='Comando a executar')
    parser.add_argument('--path', required=True,
                       help='Caminho do dataset')
    parser.add_argument('--verbose', action='store_true',
                       help='Mostrar informações detalhadas')
    
    args = parser.parse_args()
    
    if args.command == 'classes':
        print("📋 Detectando classes...")
        classes = detect_dataset_classes(args.path)
        print(f"Classes encontradas: {', '.join(classes) if classes else 'Nenhuma'}")
    
    elif args.command == 'list':
        print("📁 Listando vídeos do dataset...")
        dataset_info = load_video_dataset(args.path, verbose=args.verbose)
        
        if not args.verbose:
            total_videos = sum(len(files) for files in dataset_info.values())
            print(f"Total de vídeos: {total_videos}")
            for class_name, files in dataset_info.items():
                print(f"  {class_name}: {len(files)} vídeos")
    
    elif args.command == 'info':
        print("ℹ️ Informações básicas do dataset...")
        info = get_dataset_basic_info(args.path)
        
        print(f"📁 Dataset: {args.path}")
        print(f"🎬 Total de vídeos: {info['total_videos']}")
        print(f"📋 Classes: {', '.join(info['classes']) if info['classes'] else 'Nenhuma'}")
        
        for class_name, count in info['class_counts'].items():
            percentage = (count / info['total_videos'] * 100) if info['total_videos'] > 0 else 0
            print(f"  • {class_name}: {count} vídeos ({percentage:.1f}%)")