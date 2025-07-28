#!/usr/bin/env python3
"""
Universal AI Assistant - Tüm dosya türlerini (Excel, Word, PDF, CSV) okuyup anlayan gelişmiş sistem
"""
import os
import json
import requests
import pandas as pd
import numpy as np
from flask import Flask, render_template, request, jsonify
from datetime import datetime
import re
from typing import Dict, List, Optional, Union
from thefuzz import fuzz

from data_loader import UniversalDataLoader
from knowledge_processor import KnowledgeProcessor
from nlp_processor import SmartNLPProcessor

print("🔧 DEBUG: main.py başlatılıyor...")

try:
    from data_loader import UniversalDataLoader
    print("✅ data_loader import edildi")
except Exception as e:
    print(f"❌ data_loader import hatası: {e}")

try:
    from knowledge_processor import KnowledgeProcessor
    print("✅ knowledge_processor import edildi")
except Exception as e:
    print(f"❌ knowledge_processor import hatası: {e}")

try:
    from nlp_processor import SmartNLPProcessor
    print("✅ nlp_processor import edildi")
except Exception as e:
    print(f"❌ nlp_processor import hatası: {e}")

print(f"🔧 Çalışma dizini: {os.getcwd()}")
try:
    print(f"🔧 Dizin içeriği: {os.listdir('.')}")
except Exception as e:
    print(f"❌ Dizin listelenemedi: {e}")

if os.path.exists('company_data'):
    print(f"✅ company_data klasörü mevcut")
    try:
        print(f"📁 İçeriği: {os.listdir('company_data')}")
    except Exception as e:
        print(f"❌ company_data içeriği listelenemedi: {e}")
else:
    print("❌ company_data klasörü yok!")

class UniversalAISystem:
    def __init__(self, data_directory: str):
        self.data_directory = data_directory
        self.data_loader = UniversalDataLoader()
        self.knowledge_proc = KnowledgeProcessor()
        self.nlp_proc = SmartNLPProcessor()
        self.knowledge_base = {}
        self.structured_data = {}
        self.text_documents = {}
        self.data_insights = {}
        self.initialize_system()
    
    def initialize_system(self):
        print("🚀 Universal AI Assistant başlatılıyor...")
        print("=" * 60)
        
        self.load_all_data()
        
        if self.knowledge_base:
            print(f"\n📚 Toplam {len(self.knowledge_base)} dosya yüklendi:")
            print(f"   📊 Yapılandırılmış veri: {len(self.structured_data)} dosya")
            print(f"   📄 Metin dokümanı: {len(self.text_documents)} dosya")
            
            self.knowledge_proc.process_knowledge_base(self.knowledge_base)
            
            if self.knowledge_proc.index is not None:
                print("✅ AI hafızası başarıyla oluşturuldu!")
                # Add store names to NLP processor
                for filename, insights in self.data_insights.items():
                    if 'store_rows' in insights:
                        self.nlp_proc.add_store_names(list(insights['store_rows'].keys()))
                        print(f"DEBUG: Indexed stores in {filename}: {list(insights['store_rows'].keys())}")
                self.analyze_data_insights()
                self.test_system_capabilities()
            else:
                print("❌ AI hafızası oluşturulamadı!")
        else:
            print("❌ Hiçbir dosya yüklenemedi!")
    
    def load_all_data(self):
        print(f"📂 '{self.data_directory}' klasöründen dosyalar yükleniyor...")
        
        if not os.path.exists(self.data_directory):
            print(f"❌ Veri klasörü bulunamadı: '{self.data_directory}'")
            return
        
        file_stats = {'excel': 0, 'csv': 0, 'pdf': 0, 'word': 0, 'other': 0}
        
        for filename in os.listdir(self.data_directory):
            file_path = os.path.join(self.data_directory, filename)
            
            if not os.path.isfile(file_path):
                continue
            
            print(f"\n📁 İşleniyor: {filename}")
            file_ext = os.path.splitext(filename)[1].lower()
            
            try:
                data = self.data_loader.load_data(file_path)
                
                if data is not None:
                    self.knowledge_base[filename] = data
                    
                    if file_ext in ['.xlsx', '.xls']:
                        self._process_excel_data(filename, data)
                        file_stats['excel'] += 1
                    elif file_ext == '.csv':
                        self._process_csv_data(filename, data)
                        file_stats['csv'] += 1
                    elif file_ext == '.pdf':
                        self._process_pdf_data(filename, data)
                        file_stats['pdf'] += 1
                    elif file_ext == '.docx':
                        self._process_word_data(filename, data)
                        file_stats['word'] += 1
                    else:
                        file_stats['other'] += 1
                    
                    print(f"   ✅ Başarıyla yüklendi!")
                else:
                    print(f"   ❌ Yüklenemedi!")
                    
            except Exception as e:
                print(f"   ❌ Hata: {str(e)}")
        
        print(f"\n📈 YÜKLEME İSTATİSTİKLERİ:")
        print(f"   📊 Excel dosyası: {file_stats['excel']}")
        print(f"   📋 CSV dosyası: {file_stats['csv']}")
        print(f"   📄 PDF dosyası: {file_stats['pdf']}")
        print(f"   📝 Word dosyası: {file_stats['word']}")
        print(f"   📦 Diğer: {file_stats['other']}")
    
    def _process_excel_data(self, filename: str, data: pd.DataFrame):
        try:
            insights = {
                'type': 'excel',
                'shape': data.shape,
                'columns': list(data.columns),
                'numeric_columns': list(data.select_dtypes(include=[np.number]).columns),
                'date_columns': [],
                'text_columns': list(data.select_dtypes(include=['object']).columns),
                'sample_data': data.head(3).to_dict('records') if len(data) > 0 else [],
                'summary': {},
                'store_rows': {},
                'department_counts': {}
            }
            
            for col in data.columns:
                if any(keyword in col.lower() for keyword in ['tarih', 'date', 'gun', 'day', 'time']):
                    insights['date_columns'].append(col)
            
            for col in insights['numeric_columns']:
                insights['summary'][col] = {
                    'min': float(data[col].min()) if not data[col].empty else 0,
                    'max': float(data[col].max()) if not data[col].empty else 0,
                    'mean': float(data[col].mean()) if not data[col].empty else 0,
                    'sum': float(data[col].sum()) if not data[col].empty else 0
                }
            
            store_column = None
            for col in data.columns:
                if 'mağaza' in col.lower() or 'store' in col.lower():
                    store_column = col
                    break
            
            if store_column:
                data[store_column] = data[store_column].astype(str).str.strip()  # Ensure store names are strings and clean
                for index, row in data.iterrows():
                    store_name = row[store_column]
                    if store_name and store_name.lower() != 'nan':
                        insights['store_rows'][store_name] = {
                            col: row[col] for col in data.columns if col != store_column
                        }
                print(f"DEBUG: Loaded {len(insights['store_rows'])} stores from {filename}: {list(insights['store_rows'].keys())}")
            
            dept_column = None
            for col in data.columns:
                if 'departman' in col.lower():
                    dept_column = col
                    break
            
            if dept_column:
                insights['department_counts'] = data[dept_column].value_counts().to_dict()
            
            self.structured_data[filename] = data
            self.data_insights[filename] = insights
            
            print(f"   📊 {data.shape[0]} satır, {data.shape[1]} sütun")
            print(f"   🔢 Sayısal sütun: {len(insights['numeric_columns'])}")
            print(f"   📅 Tarih sütunu: {len(insights['date_columns'])}")
            if store_column:
                print(f"   🏬 Mağaza sütunu: {store_column}, {len(insights['store_rows'])} mağaza indekslendi")
            if dept_column:
                print(f"   👥 Departman sütunu: {dept_column}, {len(insights['department_counts'])} departman bulundu")
            
        except Exception as e:
            print(f"   ⚠️ Excel analiz hatası: {e}")
    
    def _process_csv_data(self, filename: str, data: pd.DataFrame):
        self._process_excel_data(filename, data)
        self.data_insights[filename]['type'] = 'csv'
    
    def _process_pdf_data(self, filename: str, data: str):
        insights = {
            'type': 'pdf',
            'char_count': len(data),
            'word_count': len(data.split()),
            'line_count': len(data.split('\n')),
            'has_numbers': bool(re.search(r'\d+', data)),
            'has_dates': bool(re.search(r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}', data)),
            'preview': data[:200] + "..." if len(data) > 200 else data
        }
        self.text_documents[filename] = data
        self.data_insights[filename] = insights
        print(f"   📄 {insights['word_count']} kelime, {insights['line_count']} satır")
    
    def _process_word_data(self, filename: str, data: str):
        self._process_pdf_data(filename, data)
        self.data_insights[filename]['type'] = 'word'
    
    def analyze_data_insights(self):
        print(f"\n🧠 VERİ ANALİZİ:")
        if self.structured_data:
            total_rows = sum(df.shape[0] for df in self.structured_data.values())
            print(f"   📊 Toplam veri satırı: {total_rows:,}")
            all_columns = []
            for insights in self.data_insights.values():
                if insights['type'] in ['excel', 'csv']:
                    all_columns.extend(insights['columns'])
            common_patterns = {}
            for col in all_columns:
                col_lower = col.lower()
                for pattern in ['tarih', 'date', 'satis', 'sales', 'tutar', 'amount', 'magaza', 'store', 'departman']:
                    if pattern in col_lower:
                        common_patterns[pattern] = common_patterns.get(pattern, 0) + 1
            if common_patterns:
                print(f"   🔍 Ortak veri türleri: {dict(list(common_patterns.items())[:5])}")
        if self.text_documents:
            total_words = sum(self.data_insights[f]['word_count'] 
                             for f in self.text_documents.keys() 
                             if f in self.data_insights)
            print(f"   📝 Toplam metin kelimesi: {total_words:,}")
    
    def test_system_capabilities(self):
        print(f"\n🔧 SİSTEM YETENEKLERİ TESTİ:")
        test_queries = [
            "kaç adet excel dosyası var",
            "hangi dosyalarda satış verisi bulunuyor",
            "toplam kaç satır veri var",
            "en büyük sayısal değer nedir",
            "tarih sütunu olan dosyalar hangileri",
            "İSTİNYEPARK - NSP mağazasının 2025 satış verisi nedir",
            "sales.xlsx dosyasındaki tüm mağazaların büyüme oranlarını karşılaştır",
            "Departman grafiği göster"
        ]
        for query in test_queries:
            print(f"🔍 Test: '{query}'")
            results = self.knowledge_proc.search(query, k=2)
            if results and "AI hafızası henüz oluşturulmadı" not in results[0]:
                print("  ✅ İlgili bilgi bulundu!")
            else:
                print("  ❌ Sonuç yok")
    
    def _generate_smart_answer(self, query: str, context_chunks: List[str]):
        print(f"🤖 Akıllı cevap üretiliyor: '{query[:50]}...'")
        query_lower = query.lower()
        is_data_query = any(keyword in query_lower for keyword in [
            'kaç', 'toplam', 'en çok', 'en büyük', 'en küçük', 'ortalama', 'maksimum', 'minimum',
            'satış', 'sales', 'tutar', 'amount', 'mağaza', 'store', 'tarih', 'date', 'departman', 'büyüme', 'growth'
        ])
        is_chart_query = 'grafik' in query_lower or 'göster' in query_lower
        is_growth_query = 'büyüme' in query_lower or 'growth' in query_lower

        # Use NLP processor to extract entities
        entities = self.nlp_proc.predict_intent(query, self.data_insights)
        store_name = entities['entities']['stores'][0] if entities['entities']['stores'] else None

        enriched_context = context_chunks[:5]
        response_data = {'response': '', 'chart': None}

        # Handle growth rate comparison for all stores
        if is_growth_query and 'sales.xlsx' in self.data_insights:
            sales_insights = self.data_insights['sales.xlsx']
            if 'store_rows' in sales_insights:
                growth_rates = {}
                for store, data in sales_insights['store_rows'].items():
                    try:
                        sales_2024 = float(data.get('Ciro 2024 (TRY-KDV siz)', 0))
                        sales_2025 = float(data.get('Ciro 2025 (TRY-KDV siz)', 0))
                        growth_rate = float(data.get('Ciro % Büyüme(24den25e)', 0))
                        if sales_2024 > 0:  # Verify growth rate
                            calculated_growth = ((sales_2025 - sales_2024) / sales_2024) * 100
                            growth_rates[store] = calculated_growth
                        else:
                            growth_rates[store] = growth_rate  # Use provided growth rate if 2024 sales are 0
                    except (ValueError, TypeError) as e:
                        print(f"DEBUG: Error processing store {store}: {e}")
                        growth_rates[store] = 0  # Default to 0 if data is invalid
                
                print(f"DEBUG: Growth rates calculated for {len(growth_rates)} stores: {growth_rates}")
                
                if growth_rates:
                    # Sort stores by growth rate
                    sorted_growth = sorted(growth_rates.items(), key=lambda x: x[1], reverse=True)
                    top_5_stores = sorted_growth[:5]  # Top 5 stores
                    
                    response_text = (
                        f"En çok büyüme gösteren 5 mağaza:\n" +
                        "\n".join([f"{i+1}. **{store}**: %{rate:.2f}" for i, (store, rate) in enumerate(top_5_stores)])
                    )
                    if len(sorted_growth) > 5:
                        response_text += "\n... (Diğer mağazalar için tam liste talep edebilirsiniz.)"

                    # Generate chart for top 5 stores
                    if is_chart_query:
                        response_data['chart'] = {
                            'type': 'bar',
                            'title': '2024-2025 En Çok Büyüyen 5 Mağaza',
                            'data': {
                                'labels': [store for store, _ in top_5_stores],
                                'data': [rate for _, rate in top_5_stores],
                                'backgroundColor': ['#4CAF50', '#2196F3', '#FFC107', '#F44336', '#9C27B0'],
                                'borderColor': ['#388E3C', '#1976D2', '#FFB300', '#D32F2F', '#7B1FA2'],
                                'borderWidth': 1
                            },
                            'options': {
                                'scales': {
                                    'y': {
                                        'title': {'display': True, 'text': 'Büyüme Oranı (%)'},
                                        'beginAtZero': True
                                    },
                                    'x': {
                                        'title': {'display': True, 'text': 'Mağaza Adı'}
                                    }
                                }
                            }
                        }

                    response_data['response'] = response_text
                    return response_data
                else:
                    response_data['response'] = "Büyüme oranı hesaplanacak yeterli veri bulunamadı."
                    return response_data

        # Handle department chart query
        if is_chart_query and 'departman' in query_lower:
            department_data = {}
            for filename, insights in self.data_insights.items():
                if insights['type'] in ['excel', 'csv'] and 'department_counts' in insights:
                    department_data.update(insights['department_counts'])
            if department_data:
                response_data['chart'] = {
                    'type': 'pie',
                    'title': 'Departmanlara Göre Çalışan Sayısı',
                    'data': {
                        'labels': list(department_data.keys()),
                        'data': list(department_data.values()),
                        'backgroundColor': ['#4CAF50', '#2196F3', '#FFC107', '#F44336'],
                        'borderColor': ['#388E3C', '#1976D2', '#FFB300', '#D32F2F'],
                        'borderWidth': 1
                    }
                }
                response_data['response'] = (
                    "Departmanlara göre çalışan sayıları:\n" +
                    "\n".join([f"{dept}: {count} çalışan" for dept, count in department_data.items()])
                )
                return response_data
            else:
                response_data['response'] = "Departman verisi bulunamadı."
                return response_data

        # Handle specific store queries
        if is_data_query and store_name:
            for filename, insights in self.data_insights.items():
                if insights['type'] in ['excel', 'csv'] and 'store_rows' in insights:
                    if store_name in insights['store_rows']:
                        store_data = insights['store_rows'][store_name]
                        enriched_context.append(
                            f"MAĞAZA: {store_name}\n" +
                            "\n".join([f"{key}: {value}" for key, value in store_data.items()])
                        )

        # General query processing with API
        context_string = "\n\n".join(enriched_context)
        data_summary = self._create_data_summary()
        prompt = f"""Sen evrensel bir veri analisti ve doküman uzmanısın. Farklı türdeki dosyaları (Excel, Word, PDF, CSV) analiz edip kullanıcının sorularına cevap veriyorsun.

KULLANICININ SORUSU: "{query}"

MEVCUT BİLGİLER VE VERİLER:
{context_string}

VERİ ÖZETİ:
{data_summary}

GÖREVLER:
1. Eğer sorgu sayısal/istatistiksel ise, eldeki verilerden hesaplamalar yap
2. Eğer sorguda bir mağaza adı varsa, o mağazanın satırındaki verileri (ör. satış, büyüme) döndür
3. Eğer sorgu büyüme oranı içeriyorsa, tüm mağazaların 2024-2025 büyüme oranlarını hesapla ve en yükseğini belirt
4. Eğer sorgu grafik içeriyorsa, uygun verileri özetle
5. Mümkünse spesifik sayılar, dosya isimleri ve detaylar sun
6. Türkçe, anlaşılır ve doğrudan cevap ver
7. Eğer tam bilgi yoksa, eldeki bilgilere dayanarak en iyi tahmini yap

CEVAP:"""
        api_key = "AIzaSyD-nx8kwlZz2nmVLgcrN2rgcwQKId0Duh8"
        api_url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={api_key}"
        
        payload = {"contents": [{"role": "user", "parts": [{"text": prompt}]}]}
        headers = {'Content-Type': 'application/json'}
        try:
            response = requests.post(api_url, headers=headers, json=payload, timeout=60)
            response.raise_for_status()
            result = response.json()
            if 'error' in result:
                response_data['response'] = f"Cevap üretilirken hata: {result['error'].get('message', 'Bilinmeyen hata')}"
                return response_data
            if ('candidates' in result and result['candidates'] and
                'content' in result['candidates'][0] and
                'parts' in result['candidates'][0]['content'] and
                result['candidates'][0]['content']['parts']):
                generated_text = result['candidates'][0]['content']['parts'][0]['text']
                print("✅ Akıllı cevap üretildi!")
                response_data['response'] = generated_text.strip()
                return response_data
            else:
                response_data['response'] = "Bu soru için uygun cevap üretemiyorum."
                return response_data
        except Exception as e:
            print(f"❌ LLM hatası: {e}")
            response_data['response'] = f"Cevap üretilirken teknik bir sorun oluştu: {str(e)}"
            return response_data
    
    def _create_data_summary(self):
        summary_parts = []
        file_types = {}
        for filename, insights in self.data_insights.items():
            file_type = insights['type']
            file_types[file_type] = file_types.get(file_type, 0) + 1
        summary_parts.append(f"Dosya türleri: {dict(file_types)}")
        if self.structured_data:
            total_rows = sum(df.shape[0] for df in self.structured_data.values())
            total_cols = sum(df.shape[1] for df in self.structured_data.values())
            summary_parts.append(f"Toplam veri: {total_rows} satır, {total_cols} sütun")
            numeric_summaries = []
            for filename, insights in self.data_insights.items():
                if insights['type'] in ['excel', 'csv'] and 'summary' in insights:
                    for col, stats in insights['summary'].items():
                        numeric_summaries.append(f"{filename}-{col}: {stats}")
            if numeric_summaries:
                summary_parts.append(f"Sayısal veriler: {numeric_summaries[:3]}")
        if self.text_documents:
            total_words = sum(self.data_insights[f]['word_count'] 
                             for f in self.text_documents.keys() 
                             if f in self.data_insights)
            summary_parts.append(f"Toplam metin: {total_words} kelime")
        return " | ".join(summary_parts)
    
    def process_universal_query(self, user_query: str):
        print(f"\n🔍 Evrensel sorgu işleniyor: '{user_query}'")
        relevant_chunks = self.knowledge_proc.search(user_query, k=32)  # Ensure all stores are covered
        if not relevant_chunks or "AI hafızası henüz oluşturulmadı" in relevant_chunks[0]:
            return {'response': "Sistemde henüz veri yüklenmemiş veya arama yapılamıyor.", 'chart': None}
        print(f"📋 {len(relevant_chunks)} ilgili bilgi parçası bulundu")
        final_response = self._generate_smart_answer(user_query, relevant_chunks)
        return final_response
    
    def get_system_status(self):
        return {
            'total_files': len(self.knowledge_base),
            'structured_data_files': len(self.structured_data),
            'text_document_files': len(self.text_documents),
            'ai_memory_ready': self.knowledge_proc.index is not None,
            'total_chunks': len(self.knowledge_proc.chunks) if self.knowledge_proc.chunks else 0,
            'file_types': {filename: insights['type'] for filename, insights in self.data_insights.items()},
            'data_summary': self._create_data_summary() if self.data_insights else "Veri yok"
        }

app = Flask(__name__)

print("🌟 UNIVERSAL AI ASSISTANT BAŞLATILIYOR...")
DATA_FOLDER = "company_data"
universal_system = UniversalAISystem(DATA_FOLDER)

if universal_system.knowledge_proc.index is not None:
    print("✅ UNIVERSAL AI ASSISTANT HAZIR!")
else:
    # Bu mesajı görüyorsanız, Render'daki dosya yollarında veya dosya içeriğinde bir sorun olabilir.
    print("❌ SISTEM BAŞLATILAMADI! Lütfen 'company_data' klasörünü ve dosyalarını kontrol edin.")


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/status')
def status():
    # 'universal_system' artık global alanda tanımlandığı için 'global' anahtar kelimesine gerek yok.
    if universal_system is None:
        return jsonify({'error': 'Sistem henüz başlatılmadı'}), 503
    return jsonify(universal_system.get_system_status())

@app.route('/api/query', methods=['POST'])
def query():
    # 'universal_system' artık global alanda tanımlandığı için 'global' anahtar kelimesine gerek yok.
    if universal_system is None or universal_system.knowledge_proc.index is None:
        return jsonify({'error': 'Sistem henüz hazır değil.'}), 503
    
    data = request.get_json()
    user_query = data.get('query', '').strip()
    if not user_query:
        return jsonify({'error': 'Boş sorgu'}), 400
    
    try:
        result = universal_system.process_universal_query(user_query)
        return jsonify(result)
    except Exception as e:
        print(f"❌ Sorgu hatası: {e}")
        return jsonify({'error': f'Hata: {str(e)}'}), 500

# Bu blok, dosyayı doğrudan `python main.py` ile çalıştırdığınızda (lokal geliştirme için) devreye girer.
# Gunicorn gibi production sunucuları bu bloğu çalıştırmaz, bunun yerine yukarıdaki 'app' nesnesini kullanır.
if __name__ == '__main__':
    print("\n" + "=" * 60)
    print("🌐 Lokal geliştirme sunucusu başlatılıyor...")
    print("   Tarayıcı: http://localhost:5000")
    print("   Durum: http://localhost:5000/api/status")
    print("=" * 60)
    # Production'da Gunicorn zaten kendi ayarlarını kullanacağı için buradaki host/port/debug
    # sadece lokal geliştirme içindir. debug=False kullanmak daha güvenlidir.
    app.run(host='0.0.0.0', port=5000, debug=False)