#!/usr/bin/env python3
"""
Knowledge Processor - Metin ve tablo verilerini işler, anlamsal vektörler oluşturur
ve aranabilir bir FAISS indeksi (AI hafızası) inşa eder.
"""
import os
import numpy as np
import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer
import pickle

class KnowledgeProcessor:
    def __init__(self, model_name='paraphrase-multilingual-MiniLM-L12-v2'):
        """
        Bilgi işlemciyi başlatır ve embedding modelini yükler.
        """
        print("🤖 Knowledge Processor başlatılıyor...")
        # Türkçe'yi de destekleyen güçlü bir model yüklüyoruz.
        self.model = SentenceTransformer(model_name)
        print(f"✅ Embedding modeli '{model_name}' yüklendi.")
        self.index = None # FAISS vektör indeksi
        self.chunks = []  # Orijinal metin parçaları

    def _chunk_text(self, text: str, chunk_size=1000, overlap=150) -> list[str]:
        """
        Uzun bir metni, anlam bütünlüğünü koruyacak şekilde daha küçük parçalara ayırır.
        """
        if not isinstance(text, str):
            return []
            
        chunks = []
        start = 0
        while start < len(text):
            end = start + chunk_size
            chunks.append(text[start:end])
            start += chunk_size - overlap
        return chunks

    def process_knowledge_base(self, knowledge_base: dict, force_rebuild=False):
        """
        Tüm bilgi tabanını (metinler, tablolar) işler, bir vektör indeksi oluşturur ve kaydeder.
        
        :param knowledge_base: Dosya adlarını ve içeriklerini (metin veya DataFrame) tutan sözlük.
        :param force_rebuild: Kayıtlı bir indeks olsa bile yeniden oluşturmaya zorlar.
        """
        index_path = 'faiss_index.bin'
        chunks_path = 'chunks.pkl'

        # Eğer yeniden oluşturma zorlanmadıysa ve kayıtlı dosyalar varsa, onları yükle
        if not force_rebuild and os.path.exists(index_path) and os.path.exists(chunks_path):
            print("💾 Kayıtlı AI hafızası (indeks ve metin parçaları) bulunuyor, yükleniyor...")
            self.index = faiss.read_index(index_path)
            with open(chunks_path, 'rb') as f:
                self.chunks = pickle.load(f)
            print("✅ AI hafızası başarıyla yüklendi.")
            return

        print("🧠 Yeni bir AI hafızası oluşturuluyor...")
        all_chunks = []
        
        for filename, content in knowledge_base.items():
            if isinstance(content, str): # PDF, DOCX metinleri
                text_chunks = self._chunk_text(content)
                # Her parçaya nereden geldiğini belirtmek için kaynak bilgisi ekliyoruz.
                all_chunks.extend([f"Kaynak: {filename}\nİçerik: {chunk}" for chunk in text_chunks])

            elif isinstance(content, pd.DataFrame): # Excel, CSV verileri
                # DataFrame'in her satırını okunabilir bir metne dönüştür
                for _, row in content.iterrows():
                    row_text = ", ".join([f"{col}: {val}" for col, val in row.items()])
                    all_chunks.append(f"Kaynak: {filename}\nİçerik: {row_text}")
        
        if not all_chunks:
            print("⚠️ İşlenecek veri bulunamadı. Hafıza oluşturulamadı.")
            return

        print(f"📄 Toplam {len(all_chunks)} adet metin parçası (chunk) oluşturuldu.")
        print("⏳ Vektörler (embeddings) oluşturuluyor... Bu işlem biraz zaman alabilir.")
        
        # Tüm metin parçalarını vektörlere dönüştür
        embeddings = self.model.encode(all_chunks, show_progress_bar=True)
        
        print("⚡ FAISS vektör indeksi oluşturuluyor...")
        d = embeddings.shape[1]  # Vektör boyutu
        self.index = faiss.IndexFlatL2(d)
        self.index.add(np.array(embeddings, dtype='float32'))
        self.chunks = all_chunks

        # Gelecekte hızlı yükleme için indeksi ve metin parçalarını kaydet
        print(f"💾 AI hafızası '{index_path}' ve '{chunks_path}' dosyalarına kaydediliyor.")
        faiss.write_index(self.index, index_path)
        with open(chunks_path, 'wb') as f:
            pickle.dump(self.chunks, f)
            
        print("✅ Yeni AI hafızası başarıyla oluşturuldu ve kaydedildi.")

    def search(self, query: str, k=5) -> list[str]:
        """
        Verilen bir sorgu için anlamsal olarak en alakalı metin parçalarını bulur.
        
        :param query: Kullanıcının sorduğu soru.
        :param k: Döndürülecek en iyi sonuç sayısı.
        :return: En alakalı metin parçalarının listesi.
        """
        if self.index is None:
            return ["AI hafızası henüz oluşturulmadı."]
            
        query_embedding = self.model.encode([query])
        distances, indices = self.index.search(np.array(query_embedding, dtype='float32'), k)
        
        # En alakalı metin parçalarını döndür
        results = [self.chunks[i] for i in indices[0]]
        return results