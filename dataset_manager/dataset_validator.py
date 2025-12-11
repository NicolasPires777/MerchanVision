"""
Dataset Validator Module

Provides validation and statistics functions for video datasets.
"""

import os
import argparse
from pathlib import Path


def validate_dataset(dataset_path, classes=None, min_videos_per_class=5):
    """
    Valida se dataset tem estrutura adequada para treinamento
    
    Args:
        dataset_path: Caminho para o dataset
        classes: Lista de classes esperadas (None = detectar automaticamente)
        min_videos_per_class: Mínimo de vídeos por classe
        
    Returns:
        bool: True se dataset é válido, False caso contrário
    """
    # Import here to avoid circular imports
    try:
        from .dataset_lister import detect_dataset_classes, load_video_dataset
    except ImportError:
        # Fallback for standalone execution
        from dataset_lister import detect_dataset_classes, load_video_dataset
    
    print(f"🔍 Validando dataset: {dataset_path}")
    
    if not os.path.exists(dataset_path):
        print(f"❌ Pasta do dataset não existe: {dataset_path}")
        return False
    
    # Detectar classes automaticamente se não especificadas
    if classes is None:
        classes = detect_dataset_classes(dataset_path)
    
    dataset_info = load_video_dataset(dataset_path, classes)
    
    issues = []
    total_videos = sum(len(files) for files in dataset_info.values())
    
    # Verificar se tem vídeos suficientes
    if total_videos == 0:
        issues.append("Nenhum vídeo encontrado no dataset")
    
    # Verificar cada classe
    for class_name in classes:
        video_count = len(dataset_info.get(class_name, []))
        
        if video_count == 0:
            issues.append(f"Classe '{class_name}' está vazia")
        elif video_count < min_videos_per_class:
            issues.append(f"Classe '{class_name}' tem apenas {video_count} vídeos (mínimo: {min_videos_per_class})")
    
    # Verificar balanceamento
    if total_videos > 0:
        video_counts = [len(files) for files in dataset_info.values()]
        max_count = max(video_counts)
        min_count = min(video_counts)
        
        if max_count > 0 and min_count / max_count < 0.3:  # Desbalanceamento > 70%
            issues.append(f"Dataset muito desbalanceado: {min_count} a {max_count} vídeos por classe")
    
    if issues:
        print("❌ Problemas encontrados no dataset:")
        for issue in issues:
            print(f"  • {issue}")
        
        print("\n💡 Sugestões:")
        print("  • Adicione mais vídeos nas classes com poucos exemplos")
        print("  • Certifique-se de ter ao menos 20 vídeos por classe para melhor performance")
        print("  • Mantenha proporções equilibradas entre as classes")
        return False
    
    print("✅ Dataset válido para treinamento!")
    return True


def get_dataset_statistics(dataset_path, classes=None):
    """
    Retorna estatísticas detalhadas sobre o dataset
    
    Args:
        dataset_path: Caminho para o dataset
        classes: Lista de classes esperadas (None = detectar automaticamente)
        
    Returns:
        dict: Estatísticas do dataset
    """
    # Import here to avoid circular imports
    try:
        from .dataset_lister import detect_dataset_classes, load_video_dataset
    except ImportError:
        # Fallback for standalone execution
        from dataset_lister import detect_dataset_classes, load_video_dataset
    
    # Detectar classes automaticamente se não especificadas
    if classes is None:
        classes = detect_dataset_classes(dataset_path)
    
    dataset_info = load_video_dataset(dataset_path, classes)
    
    stats = {
        'total_videos': 0,
        'classes': {},
        'balance_ratio': 0.0,
        'is_valid': False
    }
    
    # Calcular estatísticas por classe
    video_counts = []
    for class_name in classes:
        count = len(dataset_info.get(class_name, []))
        stats['classes'][class_name] = {
            'count': count,
            'files': dataset_info.get(class_name, [])
        }
        video_counts.append(count)
        stats['total_videos'] += count
    
    # Calcular razão de balanceamento
    if video_counts and max(video_counts) > 0:
        stats['balance_ratio'] = min(video_counts) / max(video_counts)
    
    # Verificar se é válido
    stats['is_valid'] = validate_dataset(dataset_path, classes, min_videos_per_class=5)
    
    return stats


if __name__ == "__main__":
    """Interface de linha de comando para validação de datasets"""
    parser = argparse.ArgumentParser(description="Validador de Dataset para Classificação de Vídeos")
    parser.add_argument('command', choices=['validate', 'stats'], 
                       help='Comando a executar')
    parser.add_argument('--path', required=True,
                       help='Caminho do dataset')
    parser.add_argument('--classes', nargs='+', default=None,
                       help='Lista de classes (None = detectar automaticamente)')
    parser.add_argument('--min-videos', type=int, default=5,
                       help='Mínimo de vídeos por classe para validação')
    
    args = parser.parse_args()
    
    if args.command == 'validate':
        print("🔍 Validando dataset...")
        is_valid = validate_dataset(args.path, args.classes, args.min_videos)
        if is_valid:
            print("✅ Dataset pronto para uso!")
        else:
            print("❌ Dataset precisa de correções")
            exit(1)
    
    elif args.command == 'stats':
        print("📊 Coletando estatísticas...")
        stats = get_dataset_statistics(args.path, args.classes)
        
        print(f"\n📈 Estatísticas do Dataset:")
        print(f"  📁 Caminho: {args.path}")
        print(f"  🎬 Total de vídeos: {stats['total_videos']}")
        print(f"  ⚖️ Taxa de balanceamento: {stats['balance_ratio']:.2f}")
        print(f"  ✅ Válido: {'Sim' if stats['is_valid'] else 'Não'}")
        
        print(f"\n📋 Por classe:")
        for class_name, info in stats['classes'].items():
            percentage = (info['count'] / stats['total_videos'] * 100) if stats['total_videos'] > 0 else 0
            print(f"  • {class_name}: {info['count']} vídeos ({percentage:.1f}%)")