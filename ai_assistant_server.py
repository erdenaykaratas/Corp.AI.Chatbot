#!/usr/bin/env python3
"""
AI Assistant Server - RAG mimarisi ve LLM ile şirket verilerini anlar ve yanıtlar.
"""
import os
import json
import requests
import pandas as pd
from flask import Flask, render_template, request, jsonify

# Gerekli sınıfları ayrı dosyalardan import ediyoruz
from data_loader import UniversalDataLoader
from knowledge_processor import KnowledgeProcessor

class AIAssistantServer:
    """
    Tüm veri kaynaklarını yöneten, RAG ve LLM kullanarak sorguları işleyen ana sunucu sınıfı.
    """
    def __init__(self, data_directory: str):
        self.data_directory = data_directory
        self.data_loader = UniversalDataLoader()
        self.knowledge_proc = KnowledgeProcessor()
        self.knowledge_base = {}
        self.load_knowledge_base()

        if self.knowledge_base:
            self.knowledge_proc.process_knowledge_base(self.knowledge_base)
        else:
            print("❌ Bilgi tabanı boş olduğu için AI hafızası oluşturulamadı.")

    def load_knowledge_base(self):
        print(f"📚 Bilgi tabanı '{self.data_directory}' klasöründen yükleniyor...")
        if not os.path.exists(self.data_directory):
            print(f"❌ Veri klasörü bulunamadı: '{self.data_directory}'. Lütfen oluşturun.")
            return
        for filename in os.listdir(self.data_directory):
            file_path = os.path.join(self.data_directory, filename)
            if os.path.isfile(file_path):
                data = self.data_loader.load_data(file_path)
                if data is not None:
                    self.knowledge_base[filename] = data
        if self.knowledge_base:
            print(f"✅ Bilgi tabanı başarıyla yüklendi. {len(self.knowledge_base)} dosya işlendi.")

    def _generate_answer_with_llm(self, query, context_chunks):
        """
        Büyük Dil Modeli (LLM) kullanarak bulunan bilgilere göre bir cevap üretir.
        (Hata yönetimi ile sağlamlaştırılmış versiyon)
        """
        print("🤖 LLM ile cevap üretiliyor...")
        
        context_string = "\n---\n".join(context_chunks)
        prompt = f"""
        Sen bir şirket içi yardım asistanısın. Görevin, sana verilen bilgileri kullanarak kullanıcının sorusuna kısa, net ve doğrudan bir cevap vermektir.
        
        Kullanıcının Sorusu: "{query}"
        
        Bu soruya cevap vermek için kullanabileceğin bilgiler:
        ---
        {context_string}
        ---
        
        Yukarıdaki bilgileri dikkatlice analiz et ve sadece bu bilgilere dayanarak soruyu cevapla. Eğer verilen bilgiler soruyu cevaplamak için yetersizse, "Bu soruya cevap verecek yeterli bilgiye sahip değilim." de. Kendi bilgini veya varsayımlarını katma.
        """

        api_key = "AIzaSyD-nx8kwlZz2nmVLgcrN2rgcwQKId0Duh8" # Canvas ortamı tarafından otomatik sağlanacaktır.
        api_url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={api_key}"
        
        payload = {"contents": [{"role": "user", "parts": [{"text": prompt}]}]}
        headers = {'Content-Type': 'application/json'}

        try:
            response = requests.post(api_url, headers=headers, json=payload, timeout=60)
            response.raise_for_status() # HTTP 4xx/5xx hatalarını kontrol et
            
            result = response.json()
            
            if 'error' in result:
                error_details = result['error']
                print(f"❌ LLM API Hatası: {error_details.get('message', 'Bilinmeyen hata')}")
                return "Cevap üretilirken modelden bir hata alındı. Detaylar için sunucu loglarını kontrol edin."

            if ('candidates' in result and result['candidates'] and
                'content' in result['candidates'][0] and
                'parts' in result['candidates'][0]['content'] and
                result['candidates'][0]['content']['parts']):
                
                generated_text = result['candidates'][0]['content']['parts'][0]['text']
                return generated_text.strip()
            else:
                print(f"⚠️ LLM'den geçerli bir aday cevap alınamadı. Yanıt: {result}")
                return "Model bu soru için bir cevap üretemedi. Lütfen sorunuzu farklı bir şekilde ifade etmeyi deneyin."

        except requests.exceptions.HTTPError as e:
            # HTTP hatalarını daha spesifik olarak yakala
            print(f"❌ LLM API Hatası: Status {e.response.status_code}, Yanıt: {e.response.text}")
            if e.response.status_code == 403:
                return "Cevap üretilemedi. API anahtarı geçersiz veya gerekli izinlere sahip değil gibi görünüyor. (Hata Kodu: 403)"
            elif e.response.status_code == 429:
                return "API kullanım limiti aşıldı. Lütfen bir süre bekleyip tekrar deneyin. (Hata Kodu: 429)"
            else:
                return f"Cevap üretilirken bir sunucu hatası oluştu. (Hata Kodu: {e.response.status_code})"
        except requests.exceptions.RequestException as e:
            # Genel ağ/bağlantı hatalarını yakala
            print(f"❌ LLM API'sine bağlanırken ağ hatası oluştu: {e}")
            return "Cevap üretilirken bir ağ sorunu oluştu. Lütfen daha sonra tekrar deneyin."
        except json.JSONDecodeError:
            print(f"❌ LLM API'sinden gelen yanıt JSON formatında değil. Yanıt: {response.text}")
            return "Cevap üretilirken modelden beklenmedik bir formatta yanıt alındı."

    def process_query_with_rag(self, user_query: str):
        """
        RAG mimarisini kullanarak sorguyu işler.
        """
        print(f"🔍 RAG ile sorgu işleniyor: '{user_query}'")
        
        relevant_chunks = self.knowledge_proc.search(user_query, k=5)

        if not relevant_chunks or "AI hafızası henüz oluşturulmadı" in relevant_chunks[0]:
            return "AI hafızası boş veya bir sorun oluştu. Lütfen sunucu loglarını kontrol edin."

        final_response = self._generate_answer_with_llm(user_query, relevant_chunks)
        
        return final_response

# --- Flask Uygulaması ---
app = Flask(__name__)
server = None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/query', methods=['POST'])
def query():
    global server
    if server is None or server.knowledge_proc.index is None:
        return jsonify({'error': 'Sunucu veya AI hafızası hazır değil.'}), 503
    
    data = request.get_json()
    user_query = data.get('query', '')
    
    if not user_query.strip():
        return jsonify({'error': 'Boş sorgu'}), 400
    
    try:
        result = server.process_query_with_rag(user_query)
        return jsonify({'response': result})
    except Exception as e:
        print(f"HATA: {e}")
        return jsonify({'error': f'Sorgu işlenirken bir hata oluştu: {str(e)}'}), 500

if __name__ == '__main__':
    DATA_FOLDER = "company_data"
    server = AIAssistantServer(DATA_FOLDER)
    
    if server.knowledge_proc.index is not None:
        print("🚀 Sunucu başarıyla başlatıldı!")
        print("🌐 Tarayıcıda http://localhost:5000 adresine gidin")
        app.run(debug=True, host='0.0.0.0', port=5000)
    else:
        print("❌ Sunucu başlatılamadı. 'company_data' klasörünü ve AI hafıza dosyalarını kontrol edin.")
