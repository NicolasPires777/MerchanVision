#!/usr/bin/env python3
"""
Classificador SRT via FFmpeg - Alternativa para quando OpenCV não suporta SRT
"""

import cv2
import numpy as np
import subprocess
import threading
import queue
import time
import argparse
import sys
import os
import json
from pathlib import Path

# Adicionar paths
current_dir = Path(__file__).parent
sys.path.append(str(current_dir))

try:
    from video_classifier_simple import SimpleVideoClassifier
except ImportError:
    print("❌ Erro: video_classifier_simple.py não encontrado")
    sys.exit(1)

class FFmpegSRTClassifier:
    def __init__(self, model_path, srt_url):
        """Classificador que usa FFmpeg para streams SRT"""
        self.model_path = model_path
        self.srt_url = srt_url
        self.classifier = SimpleVideoClassifier()
        
        # Carregar modelo
        print(f"🤖 Carregando modelo: {model_path}")
        if os.path.isdir(model_path):
            success = self.classifier.load_model(model_path)
        else:
            success = self.classifier.load_model_from_file(f"{model_path}.pkl" if not model_path.endswith('.pkl') else model_path)
            
        if not success:
            raise Exception(f"❌ Erro ao carregar modelo: {model_path}")
            
        print("✅ Modelo carregado com sucesso")
        
        # Estado de classificação
        self.current_class = "Aguardando..."
        self.current_confidence = 0.0
        self.frame_buffer = []
        self.buffer_size = 20  # Buffer menor para mais responsividade
        
    def create_ffmpeg_process(self, width=None, height=None):
        """Cria processo FFmpeg para stream SRT"""
        cmd = [
            'ffmpeg',
            '-hide_banner',
            '-loglevel', 'error',
            '-i', self.srt_url,
            '-f', 'rawvideo',
            '-pix_fmt', 'bgr24',
            '-an',  # Sem áudio
            '-sn',  # Sem legendas
        ]
        
        # Se resolução especificada, redimensionar para consistência
        if width and height:
            cmd.extend(['-s', f'{width}x{height}'])
        
        cmd.append('pipe:1')
        
        print(f"🚀 Iniciando FFmpeg para: {self.srt_url}")
        if width and height:
            print(f"🔧 Usando resolução: {width}x{height}")
            
        return subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    
    def classification_worker(self):
        """Worker thread para classificação"""
        while self.processing:
            if len(self.frame_buffer) >= 3:  # Reduzido para 3 frames mínimo
                try:
                    # Pegar últimos frames
                    frames = np.array(self.frame_buffer[-6:], dtype=np.float32) / 255.0  # Últimos 6 frames
                    
                    # Extrair features
                    features = self.classifier.extract_features_from_frames(frames)
                    if features is not None:
                        # Classificar usando o classificador treinado
                        prediction_idx = self.classifier.classifier.predict([features])[0]
                        probabilities = self.classifier.classifier.predict_proba([features])[0]
                        confidence = max(probabilities)
                        
                        # Converter índice para nome da classe
                        class_name = self.classifier.classes[prediction_idx]
                        
                        self.current_class = class_name
                        self.current_confidence = confidence
                except Exception as e:
                    print(f"⚠️ Erro na classificação: {e}")
            
            time.sleep(0.2)  # Classificar a cada 0.2s (5x por segundo)
    
    def get_stream_info(self, url):
        """Detecta informações do stream usando ffprobe"""
        try:
            cmd = [
                'ffprobe',
                '-v', 'quiet',
                '-print_format', 'json',
                '-show_streams',
                '-select_streams', 'v:0',  # Só stream de vídeo
                url
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                data = json.loads(result.stdout)
                if 'streams' in data and len(data['streams']) > 0:
                    stream = data['streams'][0]
                    width = int(stream.get('width', 1280))
                    height = int(stream.get('height', 720))
                    fps = eval(stream.get('r_frame_rate', '30/1'))  # Converter fração para número
                    
                    print(f"📺 Resolução detectada: {width}x{height} @ {fps:.1f}fps")
                    return width, height, fps
                    
        except Exception as e:
            print(f"⚠️ Erro ao detectar resolução: {e}")
            print("💡 Usando valores padrão: 1280x720 @ 30fps")
        
        return 1280, 720, 30.0

    def process_stream(self):
        """Processa stream SRT via FFmpeg"""
        # Primeiro teste se FFmpeg está disponível
        try:
            subprocess.run(['ffmpeg', '-version'], capture_output=True, check=True)
        except (subprocess.CalledProcessError, FileNotFoundError):
            print("❌ FFmpeg não encontrado!")
            print("💡 Instale com: sudo apt install ffmpeg")
            return
        
        # Detectar resolução automaticamente
        print("🔍 Detectando resolução do stream...")
        detected_width, detected_height, fps = self.get_stream_info(self.srt_url)
        
        # Usar resolução fixa para melhor performance e consistência
        width, height = 1920, 1080
        
        # Criar processo FFmpeg
        try:
            process = self.create_ffmpeg_process(width, height)
        except Exception as e:
            print(f"❌ Erro ao criar processo FFmpeg: {e}")
            return
            
        print("🎬 Iniciando classificação de stream SRT via FFmpeg")
        print("💡 Controles: Q para sair, S para screenshot")
        
        # Calcular tamanho do frame
        frame_size = width * height * 3  # 3 bytes por pixel (BGR)
        
        # Iniciar worker de classificação
        self.processing = True
        classifier_thread = threading.Thread(target=self.classification_worker)
        classifier_thread.daemon = True
        classifier_thread.start()
        
        # Calcular tamanho do frame
        frame_size = width * height * 3  # 3 bytes por pixel (BGR)
        
        try:
            while True:
                # Ler frame do FFmpeg
                raw_frame = process.stdout.read(frame_size)
                if len(raw_frame) != frame_size:
                    print("📺 Stream finalizado ou erro de leitura")
                    break
                
                # Converter para numpy array
                frame = np.frombuffer(raw_frame, dtype=np.uint8)
                frame = frame.reshape((height, width, 3))
                
                # Adicionar ao buffer para classificação
                small_frame = cv2.resize(frame, (224, 224))
                self.frame_buffer.append(small_frame)
                if len(self.frame_buffer) > self.buffer_size:
                    self.frame_buffer.pop(0)
                
                # Desenhar overlay
                frame_with_overlay = self.draw_overlay(frame)
                
                # Mostrar frame
                cv2.imshow('Classificacao SRT via FFmpeg', frame_with_overlay)
                
                # Verificar teclas
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("👋 Saindo...")
                    break
                elif key == ord('s'):
                    screenshot_name = f"screenshot_srt_{int(time.time())}.jpg"
                    cv2.imwrite(screenshot_name, frame_with_overlay)
                    print(f"📸 Screenshot salvo: {screenshot_name}")
        
        except KeyboardInterrupt:
            print("👋 Interrompido pelo usuário")
        
        finally:
            # Cleanup
            self.processing = False
            process.terminate()
            cv2.destroyAllWindows()
    
    def draw_overlay(self, frame):
        """Desenha overlay de classificação"""
        height, width = frame.shape[:2]
        
        # Fundo semi-transparente
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (width-10, 120), (0, 0, 0), -1)
        frame = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)
        
        # Cor baseada na classe
        colors = {
            'break': (0, 255, 255),      # Amarelo
            'conteudo': (0, 255, 0),     # Verde
            'merchan': (255, 0, 255)     # Magenta
        }
        color = colors.get(self.current_class.lower(), (255, 255, 255))
        
        # Texto
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame, f"CLASSE: {self.current_class.upper()}", 
                   (20, 35), font, 1.0, color, 2)
        cv2.putText(frame, f"CONFIANCA: {self.current_confidence:.1%}", 
                   (20, 65), font, 0.7, (255, 255, 255), 2)
        
        # Timestamp
        timestamp = time.strftime("%H:%M:%S")
        cv2.putText(frame, timestamp, (width-150, 35), font, 0.6, (255, 255, 255), 2)
        
        return frame

def main():
    parser = argparse.ArgumentParser(description="Classificador SRT via FFmpeg")
    parser.add_argument('--model', required=True, help='Caminho para o modelo')
    parser.add_argument('--srt-url', required=True, help='URL do stream SRT')
    
    args = parser.parse_args()
    
    print("🎬 === Classificador SRT via FFmpeg ===")
    print(f"📡 Stream: {args.srt_url}")
    print(f"🤖 Modelo: {args.model}")
    
    try:
        classifier = FFmpegSRTClassifier(args.model, args.srt_url)
        classifier.process_stream()
    except Exception as e:
        print(f"❌ Erro: {e}")

if __name__ == "__main__":
    main()