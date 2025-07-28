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
from fuzzywuzzy import fuzz

class KnowledgeProcessor:
    def __init__(self, model_name='paraphrase-multilingual-MiniLM-L12-v2'):
        """
        Bilgi işlemciyi başlatır ve embedding modelini yükler.
        """
        print("🤖 Knowledge Processor başlatılıyor...")
        self.model = SentenceTransformer(model_name)
        print(f"✅ Embedding modeli '{model_name}' yüklendi.")
        self.index = None
        self.chunks = []
        self.store_variations = {}

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
        """
        index_path = 'faiss_index.bin'
        chunks_path = 'chunks.pkl'

        # Check if files exist and are not corrupted, unless forced to rebuild
        if not force_rebuild and os.path.exists(index_path) and os.path.exists(chunks_path):
            print("💾 Kayıtlı AI hafızası (indeks ve metin parçaları) bulunuyor, yükleniyor...")
            try:
                self.index = faiss.read_index(index_path)
                with open(chunks_path, 'rb') as f:
                    self.chunks = pickle.load(f)
                print("✅ AI hafızası başarıyla yüklendi.")
                return
            except (EOFError, pickle.UnpicklingError) as e:
                print(f"❌ Kayıtlı dosya bozuk veya boş: {e}. Yeni hafıza oluşturulacak.")
                force_rebuild = True

        print("🧠 Yeni bir AI hafızası oluşturuluyor...")
        all_chunks = []
        
        for filename, content in knowledge_base.items():
            if isinstance(content, str):
                text_chunks = self._chunk_text(content)
                all_chunks.extend([f"Kaynak: {filename}\nİçerik: {chunk}" for chunk in text_chunks])

            elif isinstance(content, pd.DataFrame):
                for _, row in content.iterrows():
                    row_text = ", ".join([f"{col}: {val}" for col, val in row.items()])
                    all_chunks.append(f"Kaynak: {filename}\nİçerik: {row_text}")
                
                store_column = None
                for col in content.columns:
                    if 'mağaza' in col.lower() or 'store' in col.lower():
                        store_column = col
                        break
                
                if store_column:
                    for index, row in content.iterrows():
                        store_name = str(row[store_column]).strip()
                        if store_name and store_name.lower() != 'nan':
                            row_data = {col: row[col] for col in content.columns if col != store_column}
                            row_text = f"Mağaza: {store_name}, Veriler: {row_data}"
                            all_chunks.append(f"Kaynak: {filename}\nİçerik: {row_text}")
                            self.store_variations[store_name] = store_name
                            # Add simplified store name variations
                            base_name = store_name.split('-')[0].strip()
                            if base_name != store_name:
                                self.store_variations[base_name] = store_name
        
        if not all_chunks:
            print("⚠️ İşlenecek veri bulunamadı. Hafıza oluşturulamadı.")
            return

        print(f"📄 Toplam {len(all_chunks)} adet metin parçası (chunk) oluşturuldu.")
        print("⏳ Vektörler (embeddings) oluşturuluyor...")
        
        embeddings = self.model.encode(all_chunks, show_progress_bar=True)
        
        print("⚡ FAISS vektör indeksi oluşturuluyor...")
        d = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(d)
        self.index.add(np.array(embeddings, dtype='float32'))
        self.chunks = all_chunks

        print(f"💾 AI hafızası '{index_path}' ve '{chunks_path}' dosyalarına kaydediliyor.")
        try:
            faiss.write_index(self.index, index_path)
            with open(chunks_path, 'wb') as f:
                pickle.dump(self.chunks, f)
            print("✅ Yeni AI hafızası başarıyla oluşturuldu ve kaydedildi.")
        except Exception as e:
            print(f"❌ Hafıza kaydedilirken hata: {e}")

    def search(self, query: str, k=5) -> list[str]:
        """
        Verilen bir sorgu için anlamsal olarak en alakalı metin parçalarını bulur.
        """
        if self.index is None:
            return ["AI hafızası henüz oluşturulmadı."]
            
        query_words = query.lower().split()
        for store_name in self.store_variations.values():
            for word in query_words:
                if fuzz.partial_ratio(word, store_name.lower()) > 80:
                    query += f" {store_name}"
                    break
        
        query_embedding = self.model.encode([query])
        distances, indices = self.index.search(np.array(query_embedding, dtype='float32'), k)
        
        results = [self.chunks[i] for i in indices[0]]
        return results