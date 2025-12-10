#!/bin/bash

# Setup unificado para AI-Detector: YOLO11 + Classificação de Vídeo

echo "🎯 === Configuração Completa do AI-Detector ==="
echo

# Verificar Python
if ! command -v python3 &> /dev/null; then
    echo "❌ Python3 não encontrado. Instale Python 3.8+ primeiro."
    exit 1
fi

echo "🐍 Python encontrado: $(python3 --version)"

# Verificar pip
if ! command -v pip3 &> /dev/null; then
    echo "📦 pip3 não encontrado. Instalando..."
    sudo apt update && sudo apt install python3-pip -y
fi

echo "📦 Pip encontrado: $(pip3 --version)"
echo

# Criar ambiente virtual se não existir
if [ ! -d ".venv" ]; then
    echo "🛠️ Criando ambiente virtual..."
    python3 -m venv .venv
    echo "✅ Ambiente virtual criado em .venv/"
fi

# Ativar ambiente virtual
echo "🔄 Ativando ambiente virtual..."
source .venv/bin/activate

# Atualizar pip
echo "⬆️ Atualizando pip..."
pip install --upgrade pip

echo
echo "📚 === Instalando Dependências Principais ==="

# YOLO11 e dependências de detecção
echo "🤖 Instalando YOLO11 (Ultralytics)..."
pip install ultralytics

echo "🔥 Instalando PyTorch..."
pip install torch torchvision torchaudio

echo "📹 Instalando OpenCV completo..."
pip install opencv-python opencv-contrib-python

echo "🔧 Instalando utilitários básicos..."
pip install numpy pyyaml matplotlib pillow

echo
echo "🎬 === Instalando Dependências de Classificação de Vídeo ==="

echo "🧠 Instalando TensorFlow..."
pip install tensorflow

echo "📊 Instalando scikit-learn..."
pip install scikit-learn

echo "💾 Instalando joblib..."
pip install joblib

echo "🎵 Instalando librosa (análise de áudio)..."
pip install librosa

echo
echo "🔍 === Verificando Instalação ==="

python3 -c "
import sys
errors = []

try:
    import torch
    import torchvision
    import ultralytics
    print('✅ PyTorch:', torch.__version__)
    print('✅ YOLO11:', ultralytics.__version__)
    if torch.cuda.is_available():
        print('✅ CUDA disponível - GPU:', torch.cuda.get_device_name())
    else:
        print('ℹ️ CUDA não disponível (CPU only)')
except ImportError as e:
    print('❌ Erro PyTorch/YOLO11:', e)
    errors.append('torch/ultralytics')

try:
    import cv2
    print('✅ OpenCV:', cv2.__version__)
except ImportError as e:
    print('❌ Erro OpenCV:', e)
    errors.append('opencv')

try:
    import tensorflow as tf
    import sklearn
    import joblib
    import numpy as np
    import matplotlib
    print('✅ TensorFlow:', tf.__version__)
    print('✅ Scikit-learn:', sklearn.__version__)
    print('✅ NumPy:', np.__version__)
    print('✅ Matplotlib:', matplotlib.__version__)
    print('✅ Joblib: OK')
except ImportError as e:
    print('❌ Erro classificação:', e)
    errors.append('classification')

try:
    import librosa
    print('✅ Librosa: OK (análise de áudio)')
except ImportError:
    print('⚠️ Librosa não disponível (opcional)')

print()
if errors:
    print(f'❌ Erros encontrados em: {errors}')
    sys.exit(1)
else:
    print('🎉 Todas as dependências instaladas com sucesso!')
"

if [ $? -eq 0 ]; then
    echo
    echo "✅ === Configuração Completa! ==="
    echo
    echo "📝 Próximos passos:"
    echo "1. Ativar ambiente: source .venv/bin/activate"
    echo "2. Executar interface: python project_manager.py"
    echo "3. Para classificação de vídeo: opção 10 no menu"
    echo "4. Para YOLO11: opções 1-9 no menu"
    echo
    echo "🧪 Teste rápido:"
    echo "   python test_local_classifier.py"
    echo
else
    echo "❌ Falha na instalação. Verifique os erros acima."
    exit 1
fi