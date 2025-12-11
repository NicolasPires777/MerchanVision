#!/usr/bin/env python3
"""
Sistema de Classificação de Vídeo em Tempo Real - APENAS VÍDEOS LOCAIS
Classifica entre: break, conteudo, merchan
"""

import cv2
import numpy as np
import time
import threading
import queue
from collections import deque
import argparse
import os
import sys
from pathlib import Path

# Adicionar paths necessários
current_dir = Path(__file__).parent
sys.path.append(str(current_dir))
sys.path.append(str(current_dir.parent / 'common'))

try:
    from video_classifier_simple import SimpleVideoClassifier
except ImportError:
    print("❌ Erro: video_classifier_simple.py não encontrado")
    sys.exit(1)

class RealTimeVideoClassifier:
    def __init__(self, model_path, window_seconds=3, fps_target=30):
        """
        Classificador de vídeo em tempo real para ARQUIVOS LOCAIS APENAS
        
        Args:
            model_path: Caminho para o modelo treinado
            window_seconds: Janela de tempo para classificação (segundos) - REDUZIDO para 3s
            fps_target: FPS alvo para processamento
        """
        self.model_path = model_path
        self.window_seconds = window_seconds
        self.fps_target = fps_target
        self.frame_interval = max(1, fps_target // 8)  # Processa mais frames (1 a cada 4 frames)
        
        # Carregar classificador
        print(f"🤖 Carregando modelo: {model_path}")
        self.classifier = SimpleVideoClassifier()
        
        # Tentar diferentes formatos de caminho
        if os.path.isdir(model_path):
            success = self.classifier.load_model(model_path)
        elif model_path.endswith('.pkl'):
            success = self.classifier.load_model(model_path)
        else:
            if os.path.exists(f"{model_path}.pkl"):
                success = self.classifier.load_model(f"{model_path}.pkl")
            else:
                success = self.classifier.load_model(model_path)
        
        if not success:
            raise Exception(f"❌ Erro ao carregar modelo: {model_path}")
        
        # Buffers para frames e classificações
        self.frame_buffer = deque(maxlen=window_seconds * fps_target)
        self.classification_queue = queue.Queue(maxsize=10)
        self.current_classification = {"class": "inicializando", "confidence": 0.0}
        
        # Estado do processamento
        self.processing = False
        self.frame_count = 0
        
        # Cores para cada classe
        self.class_colors = {
            'break': (0, 255, 255),      # Amarelo
            'conteudo': (0, 255, 0),     # Verde
            'merchan': (255, 0, 255),    # Magenta
            'inicializando': (128, 128, 128)  # Cinza
        }
        
        print(f"✅ Classificador iniciado - Janela: {window_seconds}s, FPS: {fps_target}")
    
    def extract_features_from_buffer(self):
        """Extrai features do buffer de frames atual"""
        if len(self.frame_buffer) < 3:  # Reduzido para 3 frames mínimo (mais responsivo)
            return None
            
        # Pegar frames espaçados do buffer
        buffer_list = list(self.frame_buffer)
        step = max(1, len(buffer_list) // 12)  # Aumentado para 12 frames para mais detalhes
        selected_frames = buffer_list[::step][:12]
        
        # Converter para formato esperado pelo classificador
        frames_array = np.array(selected_frames)
        
        # Extrair features usando o classificador
        try:
            features = self.classifier.extract_features_from_frames(frames_array)
            return features
        except Exception as e:
            print(f"⚠️ Erro ao extrair features: {e}")
            return None
    
    def classification_worker(self):
        """Worker thread para classificação contínua"""
        while self.processing:
            try:
                # Extrair features do buffer atual
                features = self.extract_features_from_buffer()
                
                if features is not None:
                    # Classificar
                    prediction_idx = self.classifier.classifier.predict([features])[0]
                    probabilities = self.classifier.classifier.predict_proba([features])[0]
                    confidence = max(probabilities)
                    
                    # Converter índice para nome da classe
                    prediction_class = self.classifier.classes[prediction_idx]
                    
                    # Atualizar classificação atual
                    self.current_classification = {
                        "class": prediction_class,
                        "confidence": confidence,
                        "timestamp": time.time()
                    }
                
                # Aguardar menos tempo para ser mais responsivo
                time.sleep(0.2)  # Classificar a cada 0.2s (5x por segundo)
                
            except Exception as e:
                print(f"⚠️ Erro na classificação: {e}")
                time.sleep(1)
    
    def draw_classification_overlay(self, frame):
        """Desenha overlay com classificação atual"""
        height, width = frame.shape[:2]
        
        # Configurações do overlay
        class_name = self.current_classification["class"]
        confidence = self.current_classification["confidence"]
        color = self.class_colors.get(class_name, (255, 255, 255))
        
        # Garantir que class_name seja string
        if isinstance(class_name, (int, np.integer)):
            class_name = self.classifier.classes[int(class_name)] if hasattr(self.classifier, 'classes') else str(class_name)
        class_name = str(class_name)  # Força conversão para string
        
        # Caixa de fundo
        overlay_height = 120
        cv2.rectangle(frame, (0, 0), (width, overlay_height), (0, 0, 0), -1)
        cv2.rectangle(frame, (0, 0), (width, overlay_height), color, 3)
        
        # Texto principal
        font = cv2.FONT_HERSHEY_SIMPLEX
        
        # Classe
        class_text = f"CLASSIFICACAO: {class_name.upper()}"
        cv2.putText(frame, class_text, (20, 35), font, 1.2, color, 3)
        
        # Confiança
        conf_text = f"CONFIANCA: {confidence:.1%}"
        cv2.putText(frame, conf_text, (20, 70), font, 0.8, (255, 255, 255), 2)
        
        # Timestamp
        timestamp = time.strftime("%H:%M:%S")
        cv2.putText(frame, timestamp, (width-150, 35), font, 0.7, (255, 255, 255), 2)
        
        # Barra de confiança
        bar_width = int(300 * confidence)
        cv2.rectangle(frame, (20, 85), (20 + bar_width, 105), color, -1)
        cv2.rectangle(frame, (20, 85), (320, 105), color, 2)
        
        # Status do buffer
        buffer_status = f"Buffer: {len(self.frame_buffer)}/{self.frame_buffer.maxlen}"
        cv2.putText(frame, buffer_status, (20, height - 20), font, 0.5, (255, 255, 255), 1)
        
        return frame
    
    def process_video_file(self, video_path):
        """
        Processa arquivo de vídeo local
        
        Args:
            video_path: Caminho do arquivo de vídeo local
        """
        print(f"🎬 Iniciando classificação de vídeo: {video_path}")
        
        # Verificar se é webcam ou stream (não suportados)
        if str(video_path) == '0':
            print("❌ Webcam não suportada nesta versão")
            return
        elif str(video_path).startswith(('srt://', 'http://', 'https://', 'rtmp://', 'rtsp://')):
            print("❌ Streams não suportados nesta versão")
            return
        else:
            print(f"� Tipo: arquivo de vídeo local")
            if not os.path.exists(video_path):
                print(f"❌ Arquivo não encontrado: {video_path}")
                return
                
            cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            print(f"❌ Erro ao abrir arquivo de vídeo: {video_path}")
            return
        
        # Obter informações do vídeo
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print(f"📊 Resolução: {width}x{height}")
        print(f"📊 FPS: {fps:.1f}")
        if video_path != '0' and video_path != 0:
            duration = total_frames / fps if fps > 0 else 0
            print(f"📊 Duração: {duration:.1f}s ({total_frames:.0f} frames)")
        
        # Calcular delay entre frames para manter velocidade original (apenas arquivos locais)
        frame_delay = 1.0 / fps if fps > 0 else 1.0 / 30.0
        
        print(f"⏱️ Delay entre frames: {frame_delay*1000:.1f}ms")
        print(f"📹 Modo arquivo: respeitando FPS original ({fps:.1f})")
        print("💡 Controles: Q=sair, SPACE=pausar, S=screenshot, R=reiniciar")
        
        # Iniciar thread de classificação
        self.processing = True
        classification_thread = threading.Thread(target=self.classification_worker)
        classification_thread.daemon = True
        classification_thread.start()
        
        # Loop principal de processamento
        frame_count = 0
        last_time = time.time()
        frame_start_time = time.time()
        
        try:
            while True:
                ret, frame = cap.read()
                
                if not ret:
                    if video_path == '0' or video_path == 0:
                        print("⚠️ Perda de conexão com vídeo")
                        time.sleep(1)
                        continue
                    else:
                        print("📹 Fim do vídeo")
                        break
                
                # Redimensionar frame se muito grande (para display)
                display_frame = frame.copy()
                if width > 1280:
                    scale = 1280 / width
                    new_width = int(width * scale)
                    new_height = int(height * scale)
                    display_frame = cv2.resize(display_frame, (new_width, new_height))
                
                # Adicionar frame ao buffer para classificação
                if frame_count % self.frame_interval == 0:
                    # Redimensionar para processamento (224x224 para o classificador)
                    small_frame = cv2.resize(frame, (224, 224))
                    self.frame_buffer.append(small_frame)
                
                # Desenhar overlay de classificação
                display_frame = self.draw_classification_overlay(display_frame)
                
                # Mostrar frame
                cv2.imshow('Classificacao de Video Local', display_frame)
                
                # Controle de velocidade - respeitar FPS original do arquivo
                elapsed = time.time() - frame_start_time
                remaining_time = frame_delay - elapsed
                if remaining_time > 0:
                    wait_time = int(remaining_time * 1000)  # Convert to milliseconds
                    key = cv2.waitKey(wait_time) & 0xFF
                else:
                    key = cv2.waitKey(1) & 0xFF
                
                frame_start_time = time.time()  # Reset para próximo frame
                
                # Controles de teclado
                if key == ord('q'):
                    print("👋 Saindo...")
                    break
                elif key == ord(' '):
                    print("⏸️ Pausado - Pressione SPACE novamente para continuar")
                    cv2.waitKey(0)
                elif key == ord('s'):
                    # Screenshot
                    screenshot_name = f"screenshot_{int(time.time())}.jpg"
                    cv2.imwrite(screenshot_name, display_frame)
                    print(f"📸 Screenshot salvo: {screenshot_name}")
                elif key == ord('r'):
                    # Reiniciar vídeo
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    print("🔄 Vídeo reiniciado")
                
                frame_count += 1
                
                # Mostrar FPS a cada segundo
                current_time = time.time()
                if current_time - last_time >= 1.0:
                    fps_actual = frame_count / (current_time - last_time)
                    buffer_status = f"Buffer: {len(self.frame_buffer)}/{self.frame_buffer.maxlen}"
                    print(f"📊 FPS: {fps_actual:.1f} | Classe: {self.current_classification['class']} ({self.current_classification['confidence']:.1%}) | {buffer_status}")
                    frame_count = 0
                    last_time = current_time
        
        except KeyboardInterrupt:
            print("👋 Interrompido pelo usuário")
        
        finally:
            # Cleanup
            self.processing = False
            cap.release()
            cv2.destroyAllWindows()
            print("✅ Recursos liberados")

def main():
    parser = argparse.ArgumentParser(description="Classificador de Vídeo - APENAS ARQUIVOS LOCAIS")
    parser.add_argument('--model', required=True, help="Caminho para o modelo treinado")
    parser.add_argument('--source', required=True, help="Arquivo de vídeo local")
    parser.add_argument('--window', type=int, default=5, 
                       help="Janela de tempo para classificação (segundos)")
    parser.add_argument('--fps', type=int, default=30, 
                       help="FPS alvo para processamento")
    
    args = parser.parse_args()
    
    # Verificar se modelo existe
    if not os.path.exists(args.model) and not os.path.exists(f"{args.model}.pkl") and not os.path.exists(f"{args.model}/classifier.pkl"):
        print(f"❌ Modelo não encontrado: {args.model}")
        return
    
    # Verificar se arquivo de vídeo existe
    if (args.source != '0' and 
        not str(args.source).startswith(('srt://', 'http://', 'https://', 'rtmp://', 'rtsp://')) and 
        not os.path.exists(args.source)):
        print(f"❌ Arquivo de vídeo não encontrado: {args.source}")
        return
    
    try:
        # Criar classificador
        classifier = RealTimeVideoClassifier(
            model_path=args.model,
            window_seconds=args.window,
            fps_target=args.fps
        )
        
        # Processar vídeo
        classifier.process_video_file(args.source)
        
    except Exception as e:
        print(f"❌ Erro: {e}")

if __name__ == "__main__":
    print("🎬 === Classificador de Vídeo LOCAL ===")
    print("Controles:")
    print("  Q - Sair")
    print("  SPACE - Pausar/Retomar")
    print("  S - Screenshot")
    print("  R - Reiniciar vídeo (só arquivos)")
    print()
    
    main()