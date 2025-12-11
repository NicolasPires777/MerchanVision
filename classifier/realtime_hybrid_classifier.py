#!/usr/bin/env python3
"""
Sistema de Classificação Híbrida de Vídeo em Tempo Real
Combina classificação CNN com detecção de indicadores visuais
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
sys.path.append(str(current_dir.parent))  # Para acessar arquivos na raiz

try:
    # Imports absolutos em vez de relativos
    from classifier.hybrid_classifier import HybridVideoClassifier
    from config import config
    HYBRID_AVAILABLE = True
except ImportError as e:
    print(f"❌ Erro de importação híbrida: {e}")
    HYBRID_AVAILABLE = False

try:
    from classifier.visual_elements_detector import VisualElementsDetector
    MERCHAN_DETECTOR_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ Detector de indicadores não disponível: {e}")
    print("💡 Continuando com classificação apenas por imagem")
    MERCHAN_DETECTOR_AVAILABLE = False

# Fallback para classificação simples
if not HYBRID_AVAILABLE:
    try:
        from classifier.basic_classifier import BasicVideoClassifier
        print("🔄 Usando classificador de imagem tradicional")
    except ImportError as e:
        print(f"❌ Erro crítico de importação: {e}")
        sys.exit(1)

class RealTimeHybridClassifier:
    def __init__(self, model_path, window_seconds=3, fps_target=30):
        """
        Classificador híbrido de vídeo em tempo real
        
        Args:
            model_path: Caminho para o modelo treinado
            window_seconds: Janela de tempo para classificação (segundos)
            fps_target: FPS alvo para processamento
        """
        self.model_path = model_path
        self.window_seconds = window_seconds
        self.fps_target = fps_target
        self.frame_interval = max(1, fps_target // 8)
        
        # Carregar configurações
        print("⚙️ Carregando configurações...")
        if HYBRID_AVAILABLE:
            self.config = config
        else:
            self.config = {}
        
        # Inicializar classificador
        if HYBRID_AVAILABLE:
            print(f"🤖 Carregando modelo híbrido: {model_path}")
            self.classifier = HybridVideoClassifier()
        else:
            print(f"📸 Carregando classificador de imagem: {model_path}")
            self.classifier = BasicVideoClassifier()
        
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
        
        # Inicializar detector de indicadores visuais (opcional)
        if MERCHAN_DETECTOR_AVAILABLE:
            print("🔍 Inicializando detector de indicadores visuais...")
            self.merchan_detector = VisualElementsDetector()
            self.hybrid_mode = True
        else:
            print("⚠️ Modo híbrido desabilitado - apenas análise de imagem")
            self.merchan_detector = None
            self.hybrid_mode = False
        
        # Buffers para frames e classificações
        self.frame_buffer = deque(maxlen=window_seconds * fps_target)
        self.classification_queue = queue.Queue(maxsize=10)
        self.current_classification = {"class": "inicializando", "confidence": 0.0}
        self.current_indicators = {"detected": [], "scores": {}}
        
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
        
        # Cores para indicadores
        self.indicator_colors = {
            'qr_code': (0, 255, 255),     # Cyan
            'phone': (255, 165, 0),       # Laranja
            'email': (255, 0, 255),       # Magenta
            'price': (0, 255, 0),         # Verde
            'address': (255, 255, 0),     # Amarelo
            'commercial_text': (255, 0, 0) # Vermelho
        }
        
        print(f"✅ Classificador iniciado - Janela: {window_seconds}s, FPS: {fps_target}")
        if self.hybrid_mode:
            print(f"🔍 Modo híbrido ativado (imagem + indicadores visuais)")
        else:
            print(f"📸 Modo imagem apenas (híbrido desabilitado)")
    
    def extract_features_and_indicators(self):
        """Extrai features do buffer e detecta indicadores visuais"""
        if len(self.frame_buffer) < 3:
            return None, {}
            
        # Pegar frames espaçados do buffer
        buffer_list = list(self.frame_buffer)
        step = max(1, len(buffer_list) // 12)
        selected_frames = buffer_list[::step][:12]
        
        try:
            # Redimensionar frames para o tamanho esperado (224x224)
            resized_frames = []
            target_size = (224, 224)
            
            for frame in selected_frames:
                # Redimensionar frame mantendo proporção
                resized = cv2.resize(frame, target_size)
                resized_frames.append(resized)
            
            # Extrair features de imagem
            frames_array = np.array(resized_frames)
            
            if HYBRID_AVAILABLE:
                image_features = self.classifier.image_classifier.extract_features_from_frames(frames_array)
            else:
                image_features = self.classifier.extract_features_from_frames(frames_array)
            
            # Detectar indicadores visuais (opcional)
            indicators = {}
            if self.hybrid_mode and self.merchan_detector:
                try:
                    latest_frame = buffer_list[-1]  # Usar frame original para OCR
                    indicators = self.merchan_detector.detect_merchan_indicators(latest_frame)
                except Exception as e:
                    print(f"⚠️ Erro na detecção de indicadores: {e}")
                    indicators = {}
            
            return image_features, indicators
            
        except Exception as e:
            print(f"⚠️ Erro ao extrair features/indicadores: {e}")
            return None, {}
    
    def predict_hybrid_from_features(self, image_features, indicators):
        """Predição híbrida usando features já extraídas"""
        try:
            # Verificar se image_features é válido
            if image_features is None or len(image_features) == 0:
                return None
            
            # Garantir que image_features é um array 1D
            if hasattr(image_features, 'shape'):
                if len(image_features.shape) > 1:
                    image_features = image_features.flatten()
            
            # Converter para numpy array se necessário
            image_features = np.array(image_features).reshape(1, -1)
            
            if HYBRID_AVAILABLE:
                # Usar classificador híbrido
                image_probs = self.classifier.image_classifier.classifier.predict_proba(image_features)[0]
                image_prediction = self.classifier.image_classifier.classifier.predict(image_features)[0]
                classes = self.classifier.classes
            else:
                # Usar classificador simples
                image_probs = self.classifier.classifier.predict_proba(image_features)[0]
                image_prediction = self.classifier.classifier.predict(image_features)[0]
                classes = self.classifier.classes
            
            # CORREÇÃO CRÍTICA: Confiança da classe PREDITA, não máxima!
            image_confidence = image_probs[image_prediction]
            
            # Score dos indicadores visuais (nova lógica otimizada)
            merchan_score = 0.0  # SEM indicadores = 0% (deixa imagem decidir)
            indicator_info = ""
            
            if self.hybrid_mode and indicators:
                # Verificar se há indicadores detectados
                detected_indicators = self.current_indicators.get("detected", [])
                
                if detected_indicators:
                    # NOVA LÓGICA: Calcular score baseado nos tipos específicos
                    score = 0.0
                    
                    # Indicadores principais (80% cada): qr_code, phone, price
                    main_indicators = ['qr_code', 'phone', 'price']
                    main_count = sum(1 for ind in detected_indicators if ind in main_indicators)
                    
                    # Indicadores secundários (50% cada): email, address (site)
                    secondary_indicators = ['email', 'address']
                    secondary_count = sum(1 for ind in detected_indicators if ind in secondary_indicators)
                    
                    # Calcular score total
                    score = (main_count * 0.8) + (secondary_count * 0.5)
                    
                    # Máximo 100%
                    merchan_score = min(score, 1.0)
                    
                    print(f"🎯 Indicadores: {detected_indicators}")
                    print(f"  📍 Principais ({main_count}): +{main_count * 80}%")
                    print(f"  📍 Secundários ({secondary_count}): +{secondary_count * 50}%")
                    print(f"  🎯 Score final: {merchan_score:.1%} merchan")
                    
                    indicator_info = f" (principais:{main_count}, secundários:{secondary_count})"
                else:
                    # NOVO: Ausência de indicadores = boost para CONTEÚDO
                    # Se não há indicadores comerciais, significa que é mais provável ser conteúdo puro
                    merchan_score = -0.8  # NEGATIVO = forte evidência de CONTEÚDO
                    indicator_info = " (sem indicadores → +80% CONTEÚDO)"
                    print(f"📊 SEM INDICADORES → Boost de 80% para CONTEÚDO (score merchan: {merchan_score})")
            
            # Elementos detectados
            detected_elements = []
            if indicators:
                if indicators.get('qr_codes', {}).get('found', False):
                    detected_elements.append('qr_code')
                
                text_details = indicators.get('text_analysis', {}).get('details', {})
                for category, data in text_details.items():
                    if isinstance(data, dict) and data.get('found', False):
                        detected_elements.append(category)
            
            # Lógica híbrida de decisão
            if self.hybrid_mode:
                final_prediction = self._make_smart_hybrid_decision(
                    image_prediction, image_confidence, merchan_score
                )
                final_confidence = self._calculate_smart_confidence(image_probs, merchan_score)
            else:
                # Modo simples - apenas usar predição de imagem
                final_prediction = image_prediction
                final_confidence = image_confidence
            
            return {
                'class': classes[final_prediction],
                'confidence': final_confidence,
                'image_score': image_confidence,
                'merchan_score': merchan_score,
                'explanation': f"Imagem: {image_confidence:.1%}, Indicadores: {merchan_score:.1%}{indicator_info}" if self.hybrid_mode and merchan_score >= 0 else f"Imagem: {image_confidence:.1%}, Boost CONTEÚDO: {abs(merchan_score):.1%}{indicator_info}" if self.hybrid_mode else f"Imagem: {image_confidence:.1%}"
            }
            
        except Exception as e:
            print(f"⚠️ Erro na predição: {e}")
            return None
    
    def _make_smart_hybrid_decision(self, image_prediction, image_confidence, merchan_score):
        """Decisão híbrida inteligente - sempre média entre imagem e indicadores"""
        try:
            # CORREÇÃO CRÍTICA: Baseado no teste real
            # Alpha-v7 PURO diz 87% MERCHAN, mas híbrido lê como 87% CONTEUDO
            # Portanto, predição está INVERTIDA!
            
            if image_prediction == 0:  # Se predição = 0 (conteudo)
                # MAS o modelo REALMENTE está dizendo MERCHAN!
                image_merchan_prob = image_confidence  # 87% confiança = 87% merchan
                predicted_class_name = "MERCHAN"
                real_prediction = 1  # merchan
            else:  # Se predição = 1 (merchan)
                # Modelo REALMENTE está dizendo CONTEUDO
                image_merchan_prob = 1 - image_confidence  # 1 - confiança
                predicted_class_name = "CONTEUDO"
                real_prediction = 0  # conteudo
            
            # PROTEÇÃO: Se modelo tem alta confiança em MERCHAN (>70%), respeitar decisão
            if real_prediction == 1 and image_confidence > 0.7:  # merchan com alta confiança
                print(f"🔒 MERCHAN FORTE: Modelo tem {image_confidence:.1%} confiança → FORÇANDO MERCHAN")
                print(f"🖼️  Imagem: {image_merchan_prob:.1%} merchan (modelo diz: {image_confidence:.1%} {predicted_class_name})")
                print(f"✅ DECISÃO: MERCHAN (confiança alta do modelo)")
                return 1  # merchan
            
            # Lógica normal: combinar imagem (66.6%) + indicadores (33.3%)
            image_weight = float(os.getenv('HYBRID_IMAGE_WEIGHT', '0.666'))
            merchan_weight = float(os.getenv('HYBRID_MERCHAN_WEIGHT', '0.334'))
            
            # NOVA LÓGICA: merchan_score pode ser negativo (boost para conteúdo)
            if merchan_score < 0:
                # Score negativo = boost para conteúdo
                # Converter para boost de conteúdo
                content_boost = abs(merchan_score)  # -0.8 vira +0.8 para conteúdo
                final_merchan_score = (image_merchan_prob * image_weight) - (content_boost * merchan_weight)
                print(f"🖼️  Imagem: {image_merchan_prob:.1%} merchan (modelo diz: {image_confidence:.1%} {predicted_class_name}) [66.6%]")
                print(f"📊 Boost CONTEÚDO: {content_boost:.1%} (sem indicadores comerciais) [33.3%]")
                print(f"⚖️  Final: ({image_merchan_prob:.2f} × 0.666) - ({content_boost:.2f} × 0.334) = {final_merchan_score:.2f}")
            else:
                # Score positivo ou zero = normal
                final_merchan_score = (image_merchan_prob * image_weight) + (merchan_score * merchan_weight)
                print(f"🖼️  Imagem: {image_merchan_prob:.1%} merchan (modelo diz: {image_confidence:.1%} {predicted_class_name}) [66.6%]")
                if merchan_score > 0:
                    print(f"🎯 Indicadores: {merchan_score:.1%} merchan [33.3%]")
                else:
                    print(f"🎯 Indicadores: {merchan_score:.1%} merchan (neutro) [33.3%]")
                print(f"⚖️  Final: ({image_merchan_prob:.2f} × 0.666) + ({merchan_score:.2f} × 0.334) = {final_merchan_score:.2f}")
            
            # Decisão baseada no score final
            if final_merchan_score > 0.5:
                print(f"✅ DECISÃO: MERCHAN (score {final_merchan_score:.1%} > 50%)")
                return 1  # merchan
            else:
                print(f"✅ DECISÃO: CONTEÚDO (score {final_merchan_score:.1%} < 50%)")
                return 0  # conteudo
                
        except Exception as e:
            print(f"⚠️ Erro na decisão híbrida: {e}")
            return image_prediction

    def _calculate_smart_confidence(self, image_probs, merchan_score):
        """Calcula confiança final baseada na média inteligente (com suporte a scores negativos)"""
        try:
            # Confiança da imagem
            image_confidence = max(image_probs)
            
            # Para scores negativos (boost de conteúdo), usar valor absoluto para cálculo de confiança
            indicator_confidence = abs(merchan_score)
            
            # Média simples entre imagem e indicadores (sempre positiva)
            combined_confidence = (image_confidence + indicator_confidence) / 2
            
            return min(combined_confidence, 1.0)
            
        except Exception as e:
            print(f"⚠️ Erro no cálculo de confiança: {e}")
            return max(image_probs)

    def _make_simple_hybrid_decision(self, image_prediction, image_confidence, merchan_score):
        """Decisão híbrida simplificada com foco nos indicadores principais"""
        if not self.hybrid_mode:
            return image_prediction
        
        # Obter classes disponíveis
        if HYBRID_AVAILABLE:
            classes = self.classifier.classes
        else:
            classes = self.classifier.classes
        
        # Verificar se temos merchan como classe
        if 'merchan' not in classes:
            return image_prediction
        
        merchan_idx = classes.index('merchan')
        
        # REGRA ESPECIAL: Se detectou qualquer indicador principal, FORÇAR MERCHAN
        detected_indicators = self.current_indicators.get("detected", [])
        main_indicators = ['qr_code', 'phone', 'price']  # Indicadores principais
        
        # Se detectou QUALQUER indicador principal, classificar como merchan com alta confiança
        if any(indicator in detected_indicators for indicator in main_indicators):
            print(f"🎯 Indicador principal detectado: {[ind for ind in detected_indicators if ind in main_indicators]}")
            return merchan_idx
        
        # Se indicadores visuais são fortes (threshold baixo), forçar merchan
        if merchan_score >= self.config.get('HYBRID_MERCHAN_THRESHOLD', 0.15):
            return merchan_idx
        
        # Se imagem tem MUITO alta confiança, usar sua decisão
        if image_confidence >= self.config.get('HYBRID_IMAGE_CONFIDENCE_THRESHOLD', 0.8):
            return image_prediction
        
        # Combinar scores com pesos (agora favorecendo indicadores 70/30)
        image_weight = self.config.get('HYBRID_IMAGE_WEIGHT', 0.3)
        merchan_weight = self.config.get('HYBRID_MERCHAN_WEIGHT', 0.7)
        
        # Score ponderado favorecendo indicadores
        if merchan_score > 0.05:  # Qualquer indicador detectado
            weighted_score = (image_confidence * image_weight) + (merchan_score * merchan_weight)
            if weighted_score > 0.4:  # Threshold mais baixo
                return merchan_idx
        
        return image_prediction
    
    def _calculate_simple_confidence(self, image_probs, merchan_score):
        """Calcula confiança final simplificada"""
        if not self.hybrid_mode:
            return max(image_probs)
            
        image_weight = self.config.get('HYBRID_IMAGE_WEIGHT', 0.7)
        merchan_weight = self.config.get('HYBRID_MERCHAN_WEIGHT', 0.3)
        
        image_conf = max(image_probs)
        return (image_conf * image_weight) + (merchan_score * merchan_weight)
    
    def classification_worker(self):
        """Worker thread para classificação híbrida contínua"""
        while self.processing:
            try:
                # Extrair features e indicadores
                image_features, indicators = self.extract_features_and_indicators()
                
                if image_features is not None:
                    # Classificação híbrida
                    result = self.predict_hybrid_from_features(image_features, indicators)
                    
                    if result:
                        # Atualizar classificação atual
                        self.current_classification = {
                            "class": result['class'],
                            "confidence": result['confidence'],
                            "image_score": result['image_score'],
                            "merchan_score": result['merchan_score'],
                            "explanation": result['explanation'],
                            "timestamp": time.time()
                        }
                        
                        # Atualizar indicadores detectados
                        detected_indicators = []
                        indicator_scores = {}
                        
                        # Extrair informações dos indicadores de forma segura
                        if indicators:
                            # QR codes
                            if indicators.get('qr_codes', {}).get('found', False):
                                detected_indicators.append('qr_code')
                                qr_count = indicators['qr_codes'].get('count', 0)
                                indicator_scores['qr_code'] = min(qr_count * 1.0, 1.0)  # Peso máximo para QR-codes
                            
                            # Elementos de texto
                            text_analysis = indicators.get('text_analysis', {})
                            if text_analysis.get('found', False):
                                details = text_analysis.get('details', {})
                                
                                text_weights = {
                                    'phone': 1.0,      # MÁXIMO - Telefone é MERCHAN CERTO
                                    'price': 1.0,      # MÁXIMO - Preço é MERCHAN CERTO  
                                    'email': 0.2,      # MÍNIMO - Email não importa
                                    'address': 0.1,    # MÍNIMO - Endereço não importa
                                    'commercial': 0.4  # BAIXO - Texto comercial moderado
                                }
                                
                                for category, data in details.items():
                                    if isinstance(data, dict) and data.get('found', False):
                                        detected_indicators.append(category)
                                        count = data.get('count', 0)
                                        weight = text_weights.get(category, 0.5)
                                        indicator_scores[category] = min(count * weight / 2.0, 1.0)
                        
                        self.current_indicators = {
                            "detected": detected_indicators,
                            "scores": indicator_scores
                        }
                
                # Aguardar antes da próxima classificação
                time.sleep(0.3)  # Classificar a cada 0.3s
                
            except Exception as e:
                print(f"⚠️ Erro na classificação híbrida: {e}")
                time.sleep(1)
    
    def draw_hybrid_overlay(self, frame):
        """Desenha overlay com classificação híbrida e indicadores visuais"""
        height, width = frame.shape[:2]
        
        # Configurações do overlay
        class_name = self.current_classification.get("class", "inicializando")
        confidence = self.current_classification.get("confidence", 0.0)
        image_score = self.current_classification.get("image_score", 0.0)
        merchan_score = self.current_classification.get("merchan_score", 0.0)
        
        color = self.class_colors.get(class_name, (255, 255, 255))
        
        # Garantir que class_name seja string
        class_name = str(class_name)
        
        # Altura do overlay principal (aumentada para incluir mais informações)
        overlay_height = 180
        cv2.rectangle(frame, (0, 0), (width, overlay_height), (0, 0, 0), -1)
        cv2.rectangle(frame, (0, 0), (width, overlay_height), color, 3)
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        
        # Texto principal
        class_text = f"CLASSIFICACAO: {class_name.upper()}"
        cv2.putText(frame, class_text, (20, 35), font, 1.2, color, 3)
        
        # Confiança total
        conf_text = f"CONFIANCA: {confidence:.1%}"
        cv2.putText(frame, conf_text, (20, 70), font, 0.8, (255, 255, 255), 2)
        
        # Scores individuais
        img_text = f"Imagem: {image_score:.1%}"
        cv2.putText(frame, img_text, (20, 100), font, 0.6, (200, 200, 255), 2)
        
        merchan_text = f"Indicadores: {merchan_score:.1%}"
        cv2.putText(frame, merchan_text, (200, 100), font, 0.6, (255, 200, 200), 2)
        
        # Timestamp
        timestamp = time.strftime("%H:%M:%S")
        cv2.putText(frame, timestamp, (width-150, 35), font, 0.7, (255, 255, 255), 2)
        
        # Barra de confiança
        bar_width = int(300 * confidence)
        cv2.rectangle(frame, (20, 115), (20 + bar_width, 135), color, -1)
        cv2.rectangle(frame, (20, 115), (320, 135), color, 2)
        
        # Status do buffer
        buffer_status = f"Buffer: {len(self.frame_buffer)}/{self.frame_buffer.maxlen}"
        cv2.putText(frame, buffer_status, (20, height - 60), font, 0.5, (255, 255, 255), 1)
        
        # Indicadores visuais detectados
        detected = self.current_indicators.get("detected", [])
        scores = self.current_indicators.get("scores", {})
        
        if detected:
            # Título dos indicadores
            cv2.putText(frame, "INDICADORES DETECTADOS:", (20, 160), font, 0.7, (255, 255, 0), 2)
            
            # Lista de indicadores
            indicator_y = 180
            for i, indicator in enumerate(detected):
                if indicator_y > height - 100:  # Não ultrapassar a tela
                    break
                    
                score = scores.get(indicator, 0)
                indicator_color = self.indicator_colors.get(indicator, (255, 255, 255))
                
                # Traduzir nomes dos indicadores para português
                indicator_names = {
                    'qr_code': 'QR-CODE',
                    'phone': 'TELEFONE',
                    'email': 'EMAIL',
                    'price': 'PREÇO',
                    'address': 'ENDEREÇO',
                    'commercial_text': 'TEXTO COMERCIAL'
                }
                
                display_name = indicator_names.get(indicator, indicator.upper())
                
                # Texto do indicador com score
                indicator_text = f"• {display_name} ({score:.1%})"
                cv2.putText(frame, indicator_text, (30, indicator_y + 25), font, 0.6, indicator_color, 2)
                
                # Pequena barra de score
                small_bar_width = int(80 * score)
                cv2.rectangle(frame, (250, indicator_y + 15), (250 + small_bar_width, indicator_y + 25), indicator_color, -1)
                cv2.rectangle(frame, (250, indicator_y + 15), (330, indicator_y + 25), indicator_color, 1)
                
                indicator_y += 30
        else:
            cv2.putText(frame, "Nenhum indicador detectado", (20, 160), font, 0.6, (128, 128, 128), 2)
        
        return frame
    
    def process_video_file(self, video_path):
        """
        Processa arquivo de vídeo local com classificação híbrida
        
        Args:
            video_path: Caminho do arquivo de vídeo, '0' para webcam, ou URL de stream
        """
        print(f"🎬 Iniciando classificação híbrida de vídeo: {video_path}")
        
        # Determinar o tipo de source
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
            print(f"❌ Erro ao abrir vídeo: {video_path}")
            return
        
        # Obter informações do vídeo
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if fps > 0:
            duration = total_frames / fps
            frame_duration = 1.0 / fps  # Tempo entre frames em segundos
            print(f"📊 Vídeo: {fps:.1f} FPS, {total_frames} frames, {duration:.1f}s")
        else:
            frame_duration = 1.0 / 30.0  # Fallback para 30 FPS
            print(f"📊 Vídeo: FPS desconhecido, {total_frames} frames")
            print(f"⚠️ Usando 30 FPS como padrão")
        
        # Iniciar processamento
        self.processing = True
        
        # Iniciar thread de classificação
        classification_thread = threading.Thread(target=self.classification_worker)
        classification_thread.daemon = True
        classification_thread.start()
        
        frame_count = 0
        start_time = time.time()
        last_frame_time = start_time
        
        print("▶️ Pressione 'q' para sair")
        print("🔍 Aguardando detecção de indicadores visuais...")
        print(f"⏱️ Reproduzindo na velocidade original ({fps:.1f} FPS)")
        
        # Verificar se temos ambiente gráfico
        try:
            cv2.namedWindow('Classificação Híbrida em Tempo Real', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('Classificação Híbrida em Tempo Real', 1024, 768)
            display_available = True
        except Exception as e:
            print(f"⚠️ Interface gráfica não disponível: {e}")
            print("🔄 Continuando em modo texto apenas...")
            display_available = False
        
        try:
            while True:
                ret, frame = cap.read()
                
                if not ret:
                    print("🔚 Fim do vídeo ou erro na captura")
                    break
                
                # Redimensionar frame se necessário
                if frame.shape[0] > 720:
                    scale = 720 / frame.shape[0]
                    width = int(frame.shape[1] * scale)
                    frame = cv2.resize(frame, (width, 720))
                
                # Adicionar frame ao buffer (apenas frames espaçados)
                if frame_count % self.frame_interval == 0:
                    self.frame_buffer.append(frame.copy())
                
                # Desenhar overlay com classificação e indicadores
                frame_with_overlay = self.draw_hybrid_overlay(frame)
                
                # Mostrar frame apenas se display disponível
                if display_available:
                    try:
                        cv2.imshow('Classificação Híbrida em Tempo Real', frame_with_overlay)
                    except Exception as e:
                        print(f"⚠️ Erro ao exibir frame: {e}")
                        display_available = False
                
                # Controle de velocidade - respeitar FPS original
                current_time = time.time()
                elapsed_since_last_frame = current_time - last_frame_time
                
                # Calcular tempo que devemos esperar
                time_to_wait = frame_duration - elapsed_since_last_frame
                
                if time_to_wait > 0:
                    # Esperar o tempo necessário para manter FPS correto
                    time.sleep(time_to_wait)
                
                last_frame_time = time.time()
                
                # Controle de teclado
                key = cv2.waitKey(1) & 0xFF if display_available else 0xFF
                if key == ord('q'):
                    print("🛑 Parada solicitada pelo usuário")
                    break
                elif key == ord(' '):
                    print("⏸️ Pausado - Pressione espaço novamente para continuar")
                    while display_available:
                        key = cv2.waitKey(1) & 0xFF
                        if key == ord(' '):
                            break
                        elif key == ord('q'):
                            break
                    if key == ord('q'):
                        break
                
                frame_count += 1
                
                # Estatísticas a cada 100 frames
                if frame_count % 100 == 0:
                    elapsed = time.time() - start_time
                    processing_fps = frame_count / elapsed if elapsed > 0 else 0
                    expected_time = frame_count * frame_duration  # Tempo que deveria ter passado
                    reproduction_fps = frame_count / expected_time if expected_time > 0 else 0
                    
                    # Mostrar informações detalhadas se não há display
                    if not display_available:
                        class_name = self.current_classification.get('class', 'N/A')
                        confidence = self.current_classification.get('confidence', 0)
                        image_score = self.current_classification.get('image_score', 0)
                        merchan_score = self.current_classification.get('merchan_score', 0)
                        
                        # Calcular o que cada componente "votou"
                        image_says_merchan = image_score if class_name == 'merchan' else (1 - image_score)
                        
                        print(f"📈 Frame {frame_count}, Reprodução: {reproduction_fps:.1f} FPS, Processamento: {processing_fps:.1f} FPS")
                        print(f"🖼️  IMAGEM diz: {image_says_merchan:.1%} merchan")
                        print(f"🎯 INDICADORES dizem: {merchan_score:.1%} merchan")
                        print(f"⚖️  MÉDIA FINAL: {(image_says_merchan + merchan_score)/2:.1%} → {class_name.upper()} ({confidence:.1%})")
                        
                        # Mostrar indicadores detectados
                        detected = self.current_indicators.get("detected", [])
                        if detected:
                            print(f"🔍 Indicadores detectados: {', '.join(detected)}")
                        else:
                            print(f"🔍 Indicadores: nenhum (score neutro 50%)")
                        print("─" * 60)
                    else:
                        print(f"📈 Frame {frame_count}, Reprodução: {reproduction_fps:.1f} FPS, Processamento: {processing_fps:.1f} FPS")
                
                # Sair se não há display e passou muito tempo (modo automático)
                if not display_available and frame_count > 1000:  # Parar após ~30s em 30fps
                    print("🔄 Modo automático - parando após análise suficiente")
                    break
        
        except KeyboardInterrupt:
            print("🛑 Interrompido pelo usuário (Ctrl+C)")
        
        finally:
            # Limpeza
            self.processing = False
            cap.release()
            cv2.destroyAllWindows()
            
            # Estatísticas finais
            total_time = time.time() - start_time
            processing_fps = frame_count / total_time if total_time > 0 else 0
            expected_time = frame_count * frame_duration
            reproduction_fps = frame_count / expected_time if expected_time > 0 else 0
            
            print(f"\n📊 Estatísticas finais:")
            print(f"   Frames processados: {frame_count}")
            print(f"   Tempo total: {total_time:.1f}s")
            print(f"   FPS de reprodução: {reproduction_fps:.1f} (alvo: {fps:.1f})")
            print(f"   FPS de processamento: {processing_fps:.1f}")
            print(f"   Última classificação: {self.current_classification.get('class', 'N/A')} ({self.current_classification.get('confidence', 0):.1%})")
            
            if self.current_indicators.get("detected"):
                print(f"   Indicadores detectados: {', '.join(self.current_indicators['detected'])}")

def main():
    parser = argparse.ArgumentParser(description='Classificação Híbrida de Vídeo em Tempo Real')
    parser.add_argument('--model', '-m', required=True,
                       help='Caminho para o modelo treinado (.pkl ou diretório)')
    parser.add_argument('--video', '-v', required=True,
                       help='Caminho do vídeo, "0" para webcam, ou URL de stream')
    parser.add_argument('--window', '-w', type=int, default=3,
                       help='Janela de tempo em segundos (padrão: 3)')
    parser.add_argument('--fps', '-f', type=int, default=30,
                       help='FPS alvo (padrão: 30)')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("🚀 CLASSIFICAÇÃO HÍBRIDA DE VÍDEO EM TEMPO REAL")
    print("=" * 60)
    print(f"📂 Modelo: {args.model}")
    print(f"🎬 Vídeo: {args.video}")
    print(f"⏱️ Janela: {args.window}s")
    print(f"📺 FPS: {args.fps}")
    print("=" * 60)
    
    try:
        # Inicializar classificador híbrido
        classifier = RealTimeHybridClassifier(
            model_path=args.model,
            window_seconds=args.window,
            fps_target=args.fps
        )
        
        # Processar vídeo
        classifier.process_video_file(args.video)
        
    except Exception as e:
        print(f"❌ Erro: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()