#!/usr/bin/env python3
"""
🛒 Detector de Elementos Visuais de Merchandising
Identifica QR-codes, telefones, preços, emails e outros indicadores visuais
"""

import cv2
import numpy as np
import re
import pytesseract
from pathlib import Path
import json
import os

class MerchanVisualDetector:
    """Detecta elementos visuais que indicam merchandising"""
    
    def __init__(self):
        """Inicializa detectores"""
        self.qr_detector = cv2.QRCodeDetector()
        
        # Padrões regex para detectar elementos textuais
        self.patterns = {
            'phone': [
                r'\(\d{2}\)\s*\d{4,5}[-\s]*\d{4}',  # (11) 9999-9999
                r'\d{2}\s*\d{4,5}[-\s]*\d{4}',       # 11 99999-9999
                r'\+55\s*\d{2}\s*\d{4,5}[-\s]*\d{4}', # +55 11 99999-9999
                r'whats?app',                          # WhatsApp
            ],
            'email': [
                r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}',  # email@domain.com
                r'\.com\.br',                           # .com.br
                r'\.com\b',                            # .com
            ],
            'price': [
                r'R\$\s*\d+[,.]?\d*',                  # R$ 99,99
                r'\d+[,.]\d{2}\s*reais',               # 99,99 reais
                r'por\s+R?\$?\s*\d+',                  # por R$ 99
                r'\d+x\s*de\s*R?\$?\s*\d+',           # 12x de R$ 10
                r'desconto',                           # desconto
                r'promo[cç][\w]*',                     # promoção
                r'oferta',                             # oferta
            ],
            'address': [
                r'rua\s+[\w\s]+,?\s*\d+',             # Rua Nome, 123
                r'av\.?\s+[\w\s]+,?\s*\d+',           # Av. Nome, 123
                r'cep\s*:?\s*\d{5}[-.]?\d{3}',        # CEP: 12345-678
                r'\d{5}[-.]?\d{3}',                    # 12345-678
            ],
            'commercial': [
                r'liga\s*j[aá]',                       # liga já
                r'acesse\s*j[aá]',                     # acesse já
                r'compre\s*j[aá]',                     # compre já
                r'cadastre[-\s]se',                    # cadastre-se
                r'clique\s*aqui',                      # clique aqui
                r'saiba\s*mais',                       # saiba mais
                r'entre\s*em\s*contato',               # entre em contato
                r'visite\s*nosso',                     # visite nosso
            ]
        }
    
    def detect_qr_codes(self, frame):
        """Detecta QR codes no frame"""
        try:
            data, bbox, _ = self.qr_detector.detectAndDecode(frame)
            
            if bbox is not None:
                return {
                    'found': True,
                    'count': len(bbox),
                    'data': data if data else 'QR code detectado',
                    'bbox': bbox.tolist() if hasattr(bbox, 'tolist') else None
                }
        except Exception:
            pass
        
        return {'found': False, 'count': 0}
    
    def extract_text_from_frame(self, frame):
        """Extrai texto do frame usando OCR"""
        try:
            # Pré-processamento para melhorar OCR
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Aumentar contraste
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            enhanced = clahe.apply(gray)
            
            # OCR
            text = pytesseract.image_to_string(enhanced, lang='por', config='--psm 6')
            return text.lower().strip()
            
        except Exception as e:
            print(f"⚠️ Erro no OCR: {e}")
            return ""
    
    def analyze_text_patterns(self, text):
        """Analisa texto em busca de padrões de merchandising"""
        results = {}
        total_matches = 0
        
        for category, patterns in self.patterns.items():
            matches = []
            
            for pattern in patterns:
                found = re.findall(pattern, text, re.IGNORECASE)
                if found:
                    matches.extend(found)
            
            results[category] = {
                'found': len(matches) > 0,
                'count': len(matches),
                'matches': matches[:3]  # Primeiros 3 matches
            }
            
            total_matches += len(matches)
        
        return results, total_matches
    
    def detect_merchan_indicators(self, frame):
        """Detecta todos os indicadores de merchandising em um frame"""
        indicators = {
            'qr_codes': self.detect_qr_codes(frame),
            'text_analysis': {'found': False, 'total_matches': 0},
            'merchan_score': 0.0,
            'elements_found': []
        }
        
        # Análise de texto
        text = self.extract_text_from_frame(frame)
        if text:
            text_results, total_matches = self.analyze_text_patterns(text)
            indicators['text_analysis'] = {
                'found': total_matches > 0,
                'total_matches': total_matches,
                'details': text_results
            }
        
        # Calcular score de merchandising
        score = 0.0
        elements_found = []
        
        # QR Code = MUITO forte indicador (peso muito aumentado)
        if indicators['qr_codes']['found']:
            score += 0.8 * min(indicators['qr_codes']['count'], 3)  # QR-codes: peso 0.8 (era 0.6)
            elements_found.append(f"QR-codes ({indicators['qr_codes']['count']})")
        
        # Elementos textuais (pesos MUITO aumentados para principais)
        text_weights = {
            'phone': 0.8,      # MUITO ALTO - Telefone (era 0.5)
            'price': 0.8,      # MUITO ALTO - Preço (era 0.5) 
            'email': 0.1,      # MUITO BAIXO - Email (era 0.15)
            'address': 0.05,   # MUITO BAIXO - Endereço (era 0.1)
            'commercial': 0.2  # BAIXO - Call-to-action (era 0.25)
        }
        
        if indicators['text_analysis']['found']:
            for category, data in indicators['text_analysis']['details'].items():
                if data['found']:
                    weight = text_weights.get(category, 0.1)
                    score += weight * min(data['count'], 2)  # Máximo 2 por categoria
                    elements_found.append(f"{category} ({data['count']})")
        
        # Normalizar score (máximo teórico agora ~4.0, normalizar para 0-1)
        indicators['merchan_score'] = min(score / 4.0, 1.0)
        indicators['elements_found'] = elements_found
        
        return indicators
    
    def analyze_video(self, video_path, max_frames=10):
        """Analisa vídeo inteiro em busca de indicadores de merchandising"""
        cap = cv2.VideoCapture(video_path)
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_interval = max(1, total_frames // max_frames)
        
        results = []
        frame_count = 0
        
        while cap.isOpened() and len(results) < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Analisar apenas frames selecionados
            if frame_count % frame_interval == 0:
                indicators = self.detect_merchan_indicators(frame)
                indicators['frame_number'] = frame_count
                results.append(indicators)
            
            frame_count += 1
        
        cap.release()
        
        # Calcular estatísticas finais
        merchan_scores = [r['merchan_score'] for r in results]
        qr_detections = sum(1 for r in results if r['qr_codes']['found'])
        text_detections = sum(1 for r in results if r['text_analysis']['found'])
        
        all_elements = []
        for r in results:
            all_elements.extend(r['elements_found'])
        
        return {
            'video_path': video_path,
            'frames_analyzed': len(results),
            'avg_merchan_score': np.mean(merchan_scores) if merchan_scores else 0.0,
            'max_merchan_score': max(merchan_scores) if merchan_scores else 0.0,
            'qr_detection_rate': qr_detections / len(results) if results else 0,
            'text_detection_rate': text_detections / len(results) if results else 0,
            'all_elements_found': list(set(all_elements)),
            'frame_details': results
        }

def test_detector():
    """Testa o detector em vídeos"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Testa detector de elementos de merchandising")
    parser.add_argument('--video', required=True, help='Caminho do vídeo')
    parser.add_argument('--frames', type=int, default=10, help='Número de frames para analisar')
    
    args = parser.parse_args()
    
    detector = MerchanVisualDetector()
    
    print(f"🔍 Analisando: {args.video}")
    results = detector.analyze_video(args.video, args.frames)
    
    print(f"\n📊 === RESULTADOS ===")
    print(f"🎬 Frames analisados: {results['frames_analyzed']}")
    print(f"📈 Score médio de merchan: {results['avg_merchan_score']:.2f}")
    print(f"🎯 Score máximo: {results['max_merchan_score']:.2f}")
    print(f"📱 QR-codes detectados: {results['qr_detection_rate']:.1%} dos frames")
    print(f"📝 Texto comercial: {results['text_detection_rate']:.1%} dos frames")
    
    if results['all_elements_found']:
        print(f"🛒 Elementos encontrados:")
        for element in results['all_elements_found']:
            print(f"   - {element}")
    
    # Mostrar frames com maior score
    top_frames = sorted(results['frame_details'], 
                       key=lambda x: x['merchan_score'], 
                       reverse=True)[:3]
    
    print(f"\n🎯 Top 3 frames com mais indicadores:")
    for i, frame_data in enumerate(top_frames, 1):
        score = frame_data['merchan_score']
        elements = ', '.join(frame_data['elements_found']) if frame_data['elements_found'] else 'Nenhum'
        print(f"   {i}. Frame {frame_data['frame_number']}: Score {score:.2f} - {elements}")

if __name__ == "__main__":
    test_detector()