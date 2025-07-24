#!/usr/bin/env python3
"""
SmartNLPProcessor - Doğal dil işleme ve niyet tahmini sınıfı.
"""
import re
import pickle
from difflib import SequenceMatcher

class SmartNLPProcessor:
    def __init__(self):
        # Eş anlamlı kelimeler (synonyms)
        self.synonyms = {
            # Departman isimleri
            'it': ['bilgi işlem', 'bilgi_işlem', 'bilgisayar', 'teknik', 'teknoloji', 'yazılım'],
            'satış': ['sales', 'pazarlama', 'satış_pazarlama', 'ticaret'],
            'muhasebe': ['mali_işler', 'mali işler', 'finans', 'accounting'],
            'ik': ['insan_kaynakları', 'insan kaynakları', 'hr', 'personel', 'human_resources'],
            
            # Eylemler
            'listele': ['göster', 'say', 'çıkar', 'ver', 'bul', 'getir', 'show', 'list'],
            'kaç': ['ne_kadar', 'ne kadar', 'how_many', 'count', 'sayı'],
            'hangi': ['nerede', 'where', 'which', 'kim', 'who'],
            
            # Nesneler
            'çalışan': ['personel', 'employee', 'kişi', 'people', 'worker', 'staff'],
            'departman': ['birim', 'department', 'bölüm', 'unit'],
            'maaş': ['salary', 'ücret', 'gelir', 'para', 'wage'],
        }
        
        # Intent patterns
        self.intent_patterns = {
            'list_all_employees': [
                ['listele', 'çalışan'], ['göster', 'personel'], ['hepsi', 'çalışan'],
                ['tüm', 'çalışan'], ['all', 'employee']
            ],
            'count_employees': [
                ['kaç', 'çalışan'], ['ne_kadar', 'personel'], ['how_many', 'employee']
            ],
            'list_departments': [
                ['listele', 'departman'], ['göster', 'birim'], ['departman', 'neler']
            ],
            'department_analysis': [
                ['departman', 'analiz'], ['birim', 'istatistik'], ['departman', 'grafik']
            ],
            'salary_analysis': [
                ['maaş', 'analiz'], ['salary', 'average'], ['ortalama', 'maaş']
            ]
        }
        
        # Learned patterns (kullanıcı etkileşimlerinden öğrenecek)
        self.learned_patterns = {}
        self.load_learned_patterns()
    
    def save_learned_patterns(self):
        """Öğrenilen patternleri kaydet"""
        try:
            with open('learned_patterns.pkl', 'wb') as f:
                pickle.dump(self.learned_patterns, f)
        except Exception as e:
            print(f"Öğrenilen pattern'ler kaydedilemedi: {e}")
    
    def load_learned_patterns(self):
        """Öğrenilen patternleri yükle"""
        try:
            with open('learned_patterns.pkl', 'rb') as f:
                self.learned_patterns = pickle.load(f)
        except FileNotFoundError:
            self.learned_patterns = {}
        except Exception as e:
            print(f"Öğrenilen pattern'ler yüklenemedi: {e}")
            self.learned_patterns = {}
    
    def normalize_text(self, text):
        """Metni normalize et"""
        text = text.lower()
        text = re.sub(r'[^\w\s]', ' ', text)  # Noktalama işaretlerini kaldır
        text = re.sub(r'\s+', ' ', text)  # Çoklu boşlukları tek yap
        return text.strip()
    
    def expand_synonyms(self, words):
        """Kelimeleri eş anlamlılarıyla genişlet"""
        expanded = []
        for word in words:
            expanded.append(word)
            for key, synonyms in self.synonyms.items():
                if word in synonyms or word == key:
                    expanded.extend([key] + synonyms)
        return list(set(expanded))
    
    def extract_entities(self, text, data_columns):
        """Varlık çıkarımı (NER)"""
        words = self.normalize_text(text).split()
        entities = {
            'departments': [], 'actions': [], 'objects': [], 'names': []
        }
        
        expanded_words = self.expand_synonyms(words)
        
        dept_keywords = ['it', 'satış', 'muhasebe', 'ik']
        for word in expanded_words:
            for dept in dept_keywords:
                if word in self.synonyms.get(dept, []) or word == dept:
                    entities['departments'].append(dept)
        
        action_keywords = ['listele', 'kaç', 'hangi', 'göster']
        for word in expanded_words:
            for action in action_keywords:
                if word in self.synonyms.get(action, []) or word == action:
                    entities['actions'].append(action)
        
        original_words = text.split()
        for word in original_words:
            if word.istitle() and len(word) > 2 and word.lower() not in ['hangi', 'kaç', 'tüm']:
                entities['names'].append(word)
        
        return entities
    
    def calculate_intent_score(self, entities, intent_patterns):
        """Intent skorunu hesapla"""
        scores = {}
        for intent, patterns in intent_patterns.items():
            max_score = 0
            for pattern in patterns:
                score = 0
                present_words = [word for word in pattern if word in entities['actions'] or word in entities['objects'] or word in entities['departments']]
                if len(present_words) > 0:
                    score = len(present_words) / len(pattern)
                if score > max_score:
                    max_score = score
            scores[intent] = max_score
        return scores

    def predict_intent(self, text, data_columns):
        """Intent tahmini"""
        entities = self.extract_entities(text, data_columns)
        all_patterns = {**self.intent_patterns, **self.learned_patterns}
        scores = self.calculate_intent_score(entities, all_patterns)
        
        if not scores:
             return {'intent': 'unknown', 'confidence': 0, 'entities': entities}

        best_intent = max(scores.items(), key=lambda x: x[1])
        
        return {
            'intent': best_intent[0] if best_intent[1] > 0.3 else 'unknown',
            'confidence': best_intent[1],
            'entities': entities
        }
    
    def learn_from_feedback(self, user_query, correct_intent):
        """Kullanıcı geri bildiriminden öğren"""
        normalized_query = self.normalize_text(user_query)
        words = normalized_query.split()
        
        if correct_intent not in self.learned_patterns:
            self.learned_patterns[correct_intent] = []
        
        self.learned_patterns[correct_intent].append(words[:3])  # İlk 3 kelime
        self.save_learned_patterns()
