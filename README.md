# CyberSecurity AI Assistant 🤖🔒

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Dünyanın ilk tam entegre siber güvenlik yapay zeka asistanı. Zaafiyet tespitinden exploit geliştirmeye, güvenlik raporlamasından otomatik yama üretimine kadar tüm süreçleri yönetebilen AI destekli güvenlik platformu.

## 🌟 Öne Çıkan Özellikler

- **Akıllı Sohbet Arayüzü**  
  Doğal dilde siber güvenlik sorularına uzman seviyesinde cevaplar
- **Kod Üretim Motoru**  
  Güvenli kod şablonları ve otomatik zaafiyet düzeltme
- **Entegre Tarama Araçları**  
  Nmap, OpenVAS ve Metasploit ile otomatize testler
- **Dinamik Senaryo Üretici**  
  Gerçekçi saldırı senaryoları ve savunma stratejileri
- **Akıllı Raporlama**  
  Yönetici ve teknik ekiplere özel otomatik raporlar

## 🛠 Teknik Özellikler

| Bileşen              | Teknoloji Stack                     |
|----------------------|-------------------------------------|
| NLP Motoru           | BERT-base + CodeBERT                |
| Kod Analizi          | AST Parser + SemGrep Entegrasyonu   |
| Ağ Tarama            | Scapy + Nmap API                    |
| Zaafiyet Veritabanı  | CVE/NVD + OWASP Top 10 2023         |
| Raporlama            | LaTeX + Jinja2 Templating           |

## 📦 Kurulum

### Önkoşullar
- Python 3.8+
- NVIDIA GPU (CUDA 11.7+ önerilir)
- Docker Engine

### Adımlar
```bash
# Repoyu klonla
git clone https://github.com/sizin-kullanici-adiniz/cyber-ai.git
cd cyber-ai

# Sanal ortam oluştur ve aktif et
python -m venv .venv
source .venv/bin/activate  # Linux/MacOS
.venv\Scripts\activate     # Windows

# Gereksinimleri yükle
pip install -r requirements.txt

# Model dosyalarını indir
python setup.py download-models

# Ortam değişkenlerini ayarla
cp .env.example .env



