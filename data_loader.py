#!/usr/bin/env python3
"""
Universal Data Loader - Farklı formatlardaki (Excel, CSV, PDF, DOCX) dosyaları okur.
"""
import os
import pandas as pd
import PyPDF2
import docx
from typing import Any, Dict, List, Union

class UniversalDataLoader:
    """
    Farklı dosya formatlarından veri okumak için birleşik bir arayüz sağlar.
    Desteklenen formatlar: .xlsx, .csv, .pdf, .docx
    """

    def __init__(self):
        """
        DataLoader sınıfını başlatır.
        """
        print("🚀 Universal Data Loader başlatıldı.")

    def load_excel(self, file_path: str) -> pd.DataFrame:
        """
        Bir Excel dosyasını okur ve pandas DataFrame olarak döndürür.
        """
        try:
            df = pd.read_excel(file_path)
            print(f"✅ Excel dosyası başarıyla okundu: {file_path}")
            return df
        except Exception as e:
            print(f"❌ Excel dosyası okunurken hata oluştu: {file_path} - Hata: {e}")
            return pd.DataFrame() # Hata durumunda boş DataFrame döndür

    def load_csv(self, file_path: str) -> pd.DataFrame:
        """
        Bir CSV dosyasını okur ve pandas DataFrame olarak döndürür.
        """
        try:
            df = pd.read_csv(file_path)
            print(f"✅ CSV dosyası başarıyla okundu: {file_path}")
            return df
        except Exception as e:
            print(f"❌ CSV dosyası okunurken hata oluştu: {file_path} - Hata: {e}")
            return pd.DataFrame()

    def load_pdf(self, file_path: str) -> str:
        """
        Bir PDF dosyasının metin içeriğini okur.
        """
        text_content = ""
        try:
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page in pdf_reader.pages:
                    text_content += page.extract_text() + "\n"
            print(f"✅ PDF dosyası başarıyla okundu: {file_path}")
            return text_content
        except Exception as e:
            print(f"❌ PDF dosyası okunurken hata oluştu: {file_path} - Hata: {e}")
            return "" # Hata durumunda boş metin döndür

    def load_docx(self, file_path: str) -> str:
        """
        Bir Word (.docx) dosyasının metin içeriğini okur.
        """
        text_content = ""
        try:
            doc = docx.Document(file_path)
            for para in doc.paragraphs:
                text_content += para.text + "\n"
            print(f"✅ DOCX dosyası başarıyla okundu: {file_path}")
            return text_content
        except Exception as e:
            print(f"❌ DOCX dosyası okunurken hata oluştu: {file_path} - Hata: {e}")
            return ""

    def load_data(self, file_path: str) -> Union[pd.DataFrame, str, None]:
        """
        Dosya uzantısını kontrol eder ve uygun yükleyici fonksiyonunu çağırır.
        """
        if not os.path.exists(file_path):
            print(f"❌ Dosya bulunamadı: {file_path}")
            return None

        _, file_extension = os.path.splitext(file_path)
        file_extension = file_extension.lower()

        if file_extension == '.xlsx':
            return self.load_excel(file_path)
        elif file_extension == '.csv':
            return self.load_csv(file_path)
        elif file_extension == '.pdf':
            return self.load_pdf(file_path)
        elif file_extension == '.docx':
            return self.load_docx(file_path)
        else:
            print(f"⚠️ Desteklenmeyen dosya formatı: {file_extension}. Sadece .xlsx, .csv, .pdf, .docx desteklenmektedir.")
            return None

# --- Bu script'i doğrudan çalıştırmak için test bölümü ---
if __name__ == '__main__':
    # Bu bölüm, sınıfın doğru çalışıp çalışmadığını test etmek içindir.
    # Projenizin ana klasöründe 'test_data' adında bir klasör oluşturup
    # içine farklı formatlarda dosyalar koyarak test edebilirsiniz.
    
    loader = UniversalDataLoader()
    
    # Örnek bir test dosyası yolu (bu dosyaların sisteminizde olması gerekir)
    test_files_dir = "test_data"
    if not os.path.exists(test_files_dir):
        os.makedirs(test_files_dir)
        print(f"'{test_files_dir}' adında test klasörü oluşturuldu. Lütfen içine test dosyaları ekleyin.")

    # Örnek dosyalar (bu dosyaları kendiniz oluşturmalısınız)
    # ornek_excel = os.path.join(test_files_dir, "test.xlsx")
    # ornek_pdf = os.path.join(test_files_dir, "rapor.pdf")
    
    # --- Örnek Kullanım ---
    # excel_data = loader.load_data(ornek_excel)
    # if not excel_data.empty:
    #     print("\n--- Excel Verisi ---")
    #     print(excel_data.head())

    # pdf_text = loader.load_data(ornek_pdf)
    # if pdf_text:
    #     print("\n--- PDF Metni (ilk 200 karakter) ---")
    #     print(pdf_text[:200])

    print("\nTest tamamlandı. 'UniversalDataLoader' sınıfını projelerinizde import ederek kullanabilirsiniz.")