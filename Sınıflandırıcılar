
!pip install datasets transformers evaluate googletrans fastapi uvicorn fuzzywuzzy

import os
import subprocess
import warnings
import numpy as np
import pandas as pd
import requests
import torch
import re
import json
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder
from transformers import (
    AutoTokenizer,
    AutoModel,
    Trainer,
    TrainingArguments,
    EvalPrediction
)
from datasets import Dataset, concatenate_datasets
from torch import nn
from google.colab import drive
from tqdm import tqdm

# Google Drive bağlantısı
drive.mount('/content/drive')

# Uyarıları gizleme
warnings.filterwarnings("ignore")
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

# Sabitler
MODEL_NAME = "bert-base-multilingual-uncased"
SEED = 42
torch.manual_seed(SEED)

# Veri Toplama ve İşleme ######################################################

def clean_text(text):
    """Metin temizleme fonksiyonu"""
    text = str(text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def get_exploitdb_data():
    """ExploitDB verilerini indir ve işle"""
    if not os.path.exists("exploitdb"):
        subprocess.run(["git", "clone", "https://gitlab.com/exploit-database/exploitdb.git"], check=True)

    df = pd.read_csv(
        "exploitdb/files_exploits.csv",
        usecols=['description', 'type'],
        encoding='latin-1'
    )
    df = df.rename(columns={'description': 'text', 'type': 'severity'})
    df['text'] = df['text'].apply(clean_text)
    df = df[df['severity'].isin(['dos', 'remote', 'local'])].copy()
    return df

def get_nvd_data(max_results=5000):
    """NVD verilerini API'den çek"""
    base_url = "https://services.nvd.nist.gov/rest/json/cves/2.0"
    entries = []
    start_index = 0
    
    with tqdm(total=max_results, desc="NVD Verileri İndiriliyor") as pbar:
        while start_index < max_results:
            params = {
                'startIndex': start_index,
                'resultsPerPage': 2000
            }
            response = requests.get(base_url, params=params)
            if response.status_code != 200:
                break

            data = response.json()
            for vuln in data['vulnerabilities']:
                cve = vuln['cve']
                severity = cve.get('impact', {}).get('baseMetricV3', {}).get('cvssV3', {}).get('baseSeverity', 'LOW')
                desc = next((d['value'] for d in cve['descriptions'] if d['lang'] == 'en'), "")
                entries.append({
                    'text': clean_text(desc),
                    'severity': severity.upper()
                })
            
            start_index += data['resultsPerPage']
            pbar.update(data['resultsPerPage'])
            if start_index >= data['totalResults']:
                break

    return pd.DataFrame(entries)

def get_cisa_data():
    """CISA verilerini dosyadan oku"""
    file_path = "/content/drive/MyDrive/yapay/data/CISA.json"
    
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    entries = []
    for vuln in data['vulnerabilities']:
        description = vuln.get('description', '')
        entries.append({
            'text': clean_text(description),
            'is_critical': 1
        })
    
    return pd.DataFrame(entries)

def prepare_all_data():
    """Tüm veri kaynaklarını birleştir"""
    exploit_df = get_exploitdb_data()
    nvd_df = get_nvd_data()
    cisa_df = get_cisa_data()

    # Severity etiketlerini birleştir
    exploit_df['severity'] = exploit_df['severity'].map({
        'dos': 'MEDIUM',
        'remote': 'HIGH',
        'local': 'HIGH'
    })
    
    severity_df = pd.concat([
        exploit_df[['text', 'severity']],
        nvd_df[['text', 'severity']]
    ], ignore_index=True)

    # Severity kodlaması
    severity_order = ['LOW', 'MEDIUM', 'HIGH', 'CRITICAL']
    le = LabelEncoder().fit(severity_order)
    severity_df['label'] = le.transform(severity_df['severity'])
    severity_df['task_type'] = 0  # Severity görevi

    # CISA verileri
    cisa_df = cisa_df.dropna(subset=['text'])
    cisa_df = cisa_df[cisa_df['text'].str.strip() != '']
    cisa_df['label'] = cisa_df['is_critical'].astype(int)
    cisa_df['task_type'] = 1  # CISA görevi

    # Dataset oluşturma
    severity_ds = Dataset.from_pandas(severity_df[['text', 'label', 'task_type']])
    cisa_ds = Dataset.from_pandas(cisa_df[['text', 'label', 'task_type']])

    full_ds = concatenate_datasets([severity_ds, cisa_ds]).shuffle(seed=SEED)
    return full_ds.train_test_split(test_size=0.2)

# Model Mimarisi ##############################################################

class MultiTaskBERT(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = AutoModel.from_pretrained(MODEL_NAME)
        self.config = self.bert.config
        
        # Görev spesifik başlıklar
        self.severity_head = nn.Linear(self.config.hidden_size, 4)  # 4 sınıf
        self.cisa_head = nn.Linear(self.config.hidden_size, 2)      # 2 sınıf
        
        self.dropout = nn.Dropout(0.1)

    def forward(self, input_ids, attention_mask, task_type):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        pooled = self.dropout(outputs.last_hidden_state[:, 0])
        
        # Görev tipine göre çıkış seçimi
        if task_type[0] == 0:  # Severity
            return self.severity_head(pooled)
        else:  # CISA
            return self.cisa_head(pooled)

# Eğitim ve Değerlendirme #####################################################

class MultiTaskTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        # Gerekli parametreleri çıkar
        task_type = inputs.pop('task_type')
        labels = inputs.pop('labels')
        
        # Model çıktıları
        outputs = model(**inputs, task_type=task_type)
        
        # Kayıp hesaplama
        loss_fn = nn.CrossEntropyLoss()
        loss = loss_fn(outputs, labels)
        
        return (loss, outputs) if return_outputs else loss

def compute_metrics(p: EvalPrediction):
    # Görev tipini belirle
    task_type = p.task_type[0]
    
    preds = p.predictions
    labels = p.label_ids
    
    if task_type == 0:  # Severity
        preds = np.argmax(preds, axis=1)
        return {
            'accuracy': accuracy_score(labels, preds),
            'f1': f1_score(labels, preds, average='weighted')
        }
    else:  # CISA
        preds = np.argmax(preds, axis=1)
        return {
            'accuracy': accuracy_score(labels, preds),
            'f1': f1_score(labels, preds)
        }

# Ana Fonksiyon ###############################################################

def main():
    # Veri hazırlama
    data = prepare_all_data()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    def tokenize_fn(batch):
        tokenized = tokenizer(
            batch['text'],
            padding='max_length',
            truncation=True,
            max_length=256
        )
        # Task type ve label'ları ekle
        tokenized['task_type'] = batch['task_type']
        tokenized['labels'] = batch['label']
        return tokenized
    
    # Dataset'leri tokenize et
    train_ds = data['train'].map(tokenize_fn, batched=True)
    test_ds = data['test'].map(tokenize_fn, batched=True)

    # Eğitim parametreleri
    args = TrainingArguments(
        output_dir='./results',
        evaluation_strategy='steps',
        eval_steps=500,
        logging_steps=100,
        save_steps=500,
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        num_train_epochs=5,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model='f1',
        fp16=True,
        remove_unused_columns=False
    )

    # Trainer
    trainer = MultiTaskTrainer(
        model=MultiTaskBERT(),
        args=args,
        train_dataset=train_ds,
        eval_dataset=test_ds,
        compute_metrics=compute_metrics
    )
    
    # Eğitimi başlat
    print("Eğitim başlatılıyor...")
    trainer.train()
    
    # Modeli kaydet
    trainer.save_model("/content/drive/MyDrive/CyberAI/multitask_model")
    print("Eğitim başarıyla tamamlandı!")

if __name__ == "__main__":
    main()
