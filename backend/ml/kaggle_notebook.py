"""
🏥 Drug-Pred AI — Kaggle Training Notebook
===========================================

Chạy trực tiếp trên Kaggle Notebook.

SETUP trên Kaggle:
  1. Tạo notebook mới: kaggle.com/code → New Notebook
  2. Bật GPU: Settings → Accelerator → GPU T4 x2
  3. Add datasets (nút "+ Add Data" bên phải):
     - saisumanthv/medicine-recommendation-system-dataset
     - jessicali9530/kuc-hackathon-winter-2018
     - singhnavjot2062001/11000-medicine-details
     - ddrbcn/openfda-drug-labeling       (optional, ~1.7GB)
  4. Upload file này hoặc copy-paste vào notebook
  5. Run All

Output:
  - /kaggle/working/model/           → model weights
  - /kaggle/working/data/            → processed datasets
  - /kaggle/working/results/         → evaluation metrics
"""

# %%
# ============================================================
# CELL 1: INSTALL DEPENDENCIES
# ============================================================
import subprocess
import sys

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", package])

install("underthesea")
install("datasets")  # HuggingFace datasets

print("✅ Dependencies installed")

# %%
# ============================================================
# CELL 2: IMPORTS
# ============================================================
import os
import json
import warnings
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

warnings.filterwarnings("ignore")

# Paths trên Kaggle
KAGGLE_INPUT = Path("/kaggle/input")
WORKING_DIR = Path("/kaggle/working")
DATA_DIR = WORKING_DIR / "data"
MODEL_DIR = WORKING_DIR / "model"
RESULTS_DIR = WORKING_DIR / "results"

for d in [DATA_DIR, MODEL_DIR, RESULTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# Check GPU
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🖥️ Device: {DEVICE}")
if torch.cuda.is_available():
    print(f"   GPU: {torch.cuda.get_device_name(0)}")
    print(f"   VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")

# %%
# ============================================================
# CELL 3: DRUG → DRUG GROUP MAPPING
# ============================================================
DRUG_TO_GROUP = {
    # Kháng sinh
    "Amoxicillin": "Kháng sinh", "Ampicillin": "Kháng sinh",
    "Penicillin": "Kháng sinh", "Augmentin": "Kháng sinh",
    "Azithromycin": "Kháng sinh", "Erythromycin": "Kháng sinh",
    "Clarithromycin": "Kháng sinh", "Cephalexin": "Kháng sinh",
    "Ceftriaxone": "Kháng sinh", "Cefuroxime": "Kháng sinh",
    "Ciprofloxacin": "Kháng sinh", "Levofloxacin": "Kháng sinh",
    "Moxifloxacin": "Kháng sinh", "Doxycycline": "Kháng sinh",
    "Tetracycline": "Kháng sinh", "Metronidazole": "Kháng sinh",
    "Trimethoprim": "Kháng sinh", "Sulfamethoxazole": "Kháng sinh",
    "Clindamycin": "Kháng sinh", "Vancomycin": "Kháng sinh",
    "Nitrofurantoin": "Kháng sinh",

    # Giảm đau - NSAID
    "Ibuprofen": "Giảm đau NSAID", "Naproxen": "Giảm đau NSAID",
    "Diclofenac": "Giảm đau NSAID", "Celecoxib": "Giảm đau NSAID",
    "Meloxicam": "Giảm đau NSAID", "Aspirin": "Giảm đau NSAID",
    "Indomethacin": "Giảm đau NSAID", "Ketorolac": "Giảm đau NSAID",

    # Giảm đau khác
    "Acetaminophen": "Giảm đau Paracetamol", "Paracetamol": "Giảm đau Paracetamol",
    "Tramadol": "Giảm đau Opioid", "Codeine": "Giảm đau Opioid",
    "Morphine": "Giảm đau Opioid", "Oxycodone": "Giảm đau Opioid",

    # Tim mạch
    "Lisinopril": "Tim mạch", "Enalapril": "Tim mạch",
    "Ramipril": "Tim mạch", "Captopril": "Tim mạch",
    "Losartan": "Tim mạch", "Valsartan": "Tim mạch",
    "Amlodipine": "Tim mạch", "Nifedipine": "Tim mạch",
    "Diltiazem": "Tim mạch", "Metoprolol": "Tim mạch",
    "Atenolol": "Tim mạch", "Propranolol": "Tim mạch",
    "Bisoprolol": "Tim mạch", "Carvedilol": "Tim mạch",
    "Hydrochlorothiazide": "Tim mạch", "Furosemide": "Tim mạch",
    "Spironolactone": "Tim mạch", "Digoxin": "Tim mạch",

    # Tiêu hóa
    "Omeprazole": "Tiêu hóa", "Esomeprazole": "Tiêu hóa",
    "Pantoprazole": "Tiêu hóa", "Lansoprazole": "Tiêu hóa",
    "Ranitidine": "Tiêu hóa", "Famotidine": "Tiêu hóa",
    "Loperamide": "Tiêu hóa", "Domperidone": "Tiêu hóa",
    "Ondansetron": "Tiêu hóa", "Metoclopramide": "Tiêu hóa",

    # Nội tiết - Đái tháo đường
    "Metformin": "Đái tháo đường", "Glipizide": "Đái tháo đường",
    "Gliclazide": "Đái tháo đường", "Insulin": "Đái tháo đường",
    "Sitagliptin": "Đái tháo đường", "Pioglitazone": "Đái tháo đường",

    # Nội tiết - Tuyến giáp
    "Levothyroxine": "Tuyến giáp",

    # Hô hấp
    "Salbutamol": "Hô hấp", "Albuterol": "Hô hấp",
    "Ipratropium": "Hô hấp", "Montelukast": "Hô hấp",
    "Theophylline": "Hô hấp", "Budesonide": "Hô hấp",
    "Fluticasone": "Hô hấp",

    # Thần kinh - Trầm cảm/Lo âu
    "Sertraline": "Thần kinh", "Fluoxetine": "Thần kinh",
    "Escitalopram": "Thần kinh", "Citalopram": "Thần kinh",
    "Paroxetine": "Thần kinh", "Venlafaxine": "Thần kinh",
    "Duloxetine": "Thần kinh", "Amitriptyline": "Thần kinh",
    "Bupropion": "Thần kinh",

    # Thần kinh - Động kinh
    "Gabapentin": "Chống động kinh", "Pregabalin": "Chống động kinh",
    "Carbamazepine": "Chống động kinh", "Valproic Acid": "Chống động kinh",
    "Lamotrigine": "Chống động kinh", "Levetiracetam": "Chống động kinh",
    "Phenytoin": "Chống động kinh", "Topiramate": "Chống động kinh",

    # An thần
    "Diazepam": "An thần", "Lorazepam": "An thần",
    "Alprazolam": "An thần", "Clonazepam": "An thần",
    "Zolpidem": "An thần",

    # Dị ứng
    "Cetirizine": "Dị ứng", "Loratadine": "Dị ứng",
    "Fexofenadine": "Dị ứng", "Diphenhydramine": "Dị ứng",
    "Chlorpheniramine": "Dị ứng", "Hydroxyzine": "Dị ứng",

    # Corticosteroid
    "Prednisolone": "Corticosteroid", "Prednisone": "Corticosteroid",
    "Dexamethasone": "Corticosteroid", "Hydrocortisone": "Corticosteroid",
    "Methylprednisolone": "Corticosteroid",

    # Chuyển hóa - Lipid
    "Atorvastatin": "Hạ mỡ máu", "Simvastatin": "Hạ mỡ máu",
    "Rosuvastatin": "Hạ mỡ máu", "Pravastatin": "Hạ mỡ máu",
    "Fenofibrate": "Hạ mỡ máu",

    # Chống đông
    "Warfarin": "Chống đông", "Heparin": "Chống đông",
    "Clopidogrel": "Chống đông", "Rivaroxaban": "Chống đông",
    "Apixaban": "Chống đông", "Enoxaparin": "Chống đông",

    # Kháng nấm
    "Clotrimazole": "Kháng nấm", "Fluconazole": "Kháng nấm",
    "Ketoconazole": "Kháng nấm", "Itraconazole": "Kháng nấm",
    "Nystatin": "Kháng nấm", "Terbinafine": "Kháng nấm",

    # Kháng virus
    "Acyclovir": "Kháng virus", "Valacyclovir": "Kháng virus",
    "Oseltamivir": "Kháng virus",

    # Cơ xương khớp
    "Allopurinol": "Cơ xương khớp", "Colchicine": "Cơ xương khớp",
    "Methotrexate": "Cơ xương khớp",

    # Migraine
    "Sumatriptan": "Migraine", "Rizatriptan": "Migraine",
}


def get_drug_group(drug_name: str) -> str | None:
    """Map tên thuốc → nhóm thuốc."""
    if not drug_name or not isinstance(drug_name, str):
        return None
    name = drug_name.strip()
    # Exact
    if name in DRUG_TO_GROUP:
        return DRUG_TO_GROUP[name]
    # Case-insensitive
    for drug, group in DRUG_TO_GROUP.items():
        if drug.lower() == name.lower():
            return group
    # Partial
    for drug, group in DRUG_TO_GROUP.items():
        if drug.lower() in name.lower():
            return group
    return None


DRUG_GROUPS = sorted(set(DRUG_TO_GROUP.values()))
print(f"📋 {len(DRUG_TO_GROUP)} drugs → {len(DRUG_GROUPS)} groups:")
for g in DRUG_GROUPS:
    print(f"   • {g}")

# %%
# ============================================================
# CELL 4: LOAD & PROCESS DATASETS
# ============================================================
print("=" * 60)
print("📥 Loading datasets từ /kaggle/input/")
print("=" * 60)

all_data = []

# --- Tự động tìm datasets đã add ---
available_datasets = []
if KAGGLE_INPUT.exists():
    available_datasets = [d.name for d in KAGGLE_INPUT.iterdir() if d.is_dir()]
    print(f"\n📁 Datasets có sẵn: {available_datasets}")

def find_dir(keywords):
    if not KAGGLE_INPUT.exists(): return None
    for d in KAGGLE_INPUT.iterdir():
        if d.is_dir() and any(k in d.name.lower() for k in keywords):
            return d
    return None

# --- Dataset 1: Medicine Recommendation System ---
med_dir = find_dir(["medicine-recommendation", "saisumanthv"])
if med_dir:
    print(f"\n🔧 [1] Medicine Recommendation: {med_dir}")
    for f in med_dir.rglob("*.csv"):
        try:
            df = pd.read_csv(f, on_bad_lines="skip")
            symptom_cols = [c for c in df.columns if "symptom" in c.lower()]
            drug_cols = [c for c in df.columns if any(x in c.lower() for x in ["drug", "medicine", "medication"])]

            if symptom_cols and drug_cols:
                print(f"  → {f.name}: {len(df)} rows")
                for _, row in df.iterrows():
                    symptoms = [
                        str(row[c]).replace("_", " ").strip()
                        for c in symptom_cols
                        if str(row.get(c, "")).strip().lower() not in ["nan", "none", ""]
                    ]
                    if not symptoms:
                        continue
                    text = ", ".join(symptoms)
                    for col in drug_cols:
                        dg = get_drug_group(str(row.get(col, "")))
                        if dg:
                            all_data.append({"text": text, "drug_group": dg, "source": "med_rec"})
                            break
        except Exception as e:
            print(f"  ⚠️ Error: {e}")
else:
    print("\n⚠️ [1] Không tìm thấy Medicine Recommendation Dataset")

# --- Dataset 2: UCI Drug Review ---
uci_dir = find_dir(["kuc-hackathon", "drug-review", "jessicali9530"])
if uci_dir:
    print(f"\n🔧 [2] UCI Drug Review: {uci_dir}")
    for f in list(uci_dir.rglob("*.tsv")) + list(uci_dir.rglob("*.csv")):
        try:
            sep = "\t" if f.suffix == ".tsv" else ","
            df = pd.read_csv(f, sep=sep, on_bad_lines="skip")
            if "drugName" in df.columns and "review" in df.columns:
                print(f"  → {f.name}: {len(df)} rows")
                if "rating" in df.columns:
                    df = df[df["rating"] >= 7]
                for _, row in df.iterrows():
                    dg = get_drug_group(str(row.get("drugName", "")))
                    review = str(row.get("review", "")).strip()
                    if dg and len(review) > 20:
                        all_data.append({
                            "text": review[:500],
                            "drug_group": dg,
                            "source": "uci_review",
                        })
        except Exception as e:
            print(f"  ⚠️ Error: {e}")
else:
    print("\n⚠️ [2] Không tìm thấy UCI Drug Review Dataset")

# --- Dataset 3: 11000 Medicine Details ---
det_dir = find_dir(["11000-medicine", "11000medicine", "singhnavjot2062001"])
if det_dir:
    print(f"\n🔧 [3] 11000 Medicine Details: {det_dir}")
    for f in det_dir.rglob("*.csv"):
        try:
            df = pd.read_csv(f, on_bad_lines="skip")
            name_col = next((c for c in df.columns if c.lower().strip() in
                           ["medicine name", "medicine_name", "name"]), None)
            uses_col = next((c for c in df.columns if c.lower().strip() in
                           ["uses", "use", "indication", "description"]), None)
            comp_col = next((c for c in df.columns if "compos" in c.lower() or
                           "salt" in c.lower()), None)

            if uses_col:
                count = 0
                for _, row in df.iterrows():
                    dg = None
                    if name_col:
                        dg = get_drug_group(str(row.get(name_col, "")))
                    if not dg and comp_col:
                        comp = str(row.get(comp_col, ""))
                        for drug in DRUG_TO_GROUP:
                            if drug.lower() in comp.lower():
                                dg = DRUG_TO_GROUP[drug]
                                break
                    uses = str(row.get(uses_col, "")).strip()
                    if dg and len(uses) > 15:
                        all_data.append({"text": uses[:500], "drug_group": dg, "source": "11k_med"})
                        count += 1
                print(f"  → {f.name}: matched {count} samples")
        except Exception as e:
            print(f"  ⚠️ Error: {e}")
else:
    print("\n⚠️ [3] Không tìm thấy 11000 Medicine Details Dataset")

# --- Dataset 4: OpenFDA (optional, large) ---
fda_dir = find_dir(["openfda", "ddrbcn"])
if fda_dir:
    print(f"\n🔧 [4] OpenFDA Drug Labeling: {fda_dir}")
    json_files = list(fda_dir.rglob("*.json"))

    # Giải nén nếu cần
    if not json_files:
        import zipfile
        for zf in fda_dir.rglob("*.zip"):
            try:
                extract_to = WORKING_DIR / "fda_extracted"
                extract_to.mkdir(exist_ok=True)
                with zipfile.ZipFile(zf) as z:
                    z.extractall(extract_to)
            except Exception as e:
                print(f"  ⚠️ Unzip error: {e}")
        json_files = list((WORKING_DIR / "fda_extracted").rglob("*.json"))

    fda_count = 0
    for f in json_files[:5]:  # Giới hạn 5 files vì data rất lớn
        try:
            with open(f, "r", encoding="utf-8") as fp:
                data = json.load(fp)
            results = data.get("results", [data] if isinstance(data, dict) else data)
            if not isinstance(results, list):
                results = [results]
            for rec in results:
                ind = rec.get("indications_and_usage", [])
                if isinstance(ind, list):
                    ind = " ".join(ind)
                ind = str(ind).strip()
                openfda = rec.get("openfda", {})
                names = openfda.get("generic_name", []) or openfda.get("brand_name", [])
                drug_name = names[0] if isinstance(names, list) and names else str(names)
                dg = get_drug_group(drug_name)
                if dg and len(ind) > 30:
                    all_data.append({"text": ind[:500], "drug_group": dg, "source": "openfda"})
                    fda_count += 1
        except Exception as e:
            print(f"  ⚠️ Error: {e}")
    print(f"  → Matched: {fda_count} samples from OpenFDA")

# --- Dataset 5: HoangHa/medical-data (HuggingFace - Vietnamese) ---
print(f"\n🔧 [5] HoangHa/medical-data (HuggingFace)...")
try:
    from datasets import load_dataset
    ds = load_dataset("HoangHa/medical-data", "vietnamese", split="train",
                      trust_remote_code=True)
    print(f"  → Loaded: {len(ds)} samples")
    vi_count = 0
    for row in ds:
        messages = row.get("messages", [])
        disease = str(row.get("target_disease", "")).strip()
        patient_text = ""
        if isinstance(messages, list):
            for msg in messages:
                if isinstance(msg, dict) and msg.get("role", "").lower() in ["user", "patient", "human"]:
                    patient_text = msg.get("content", "")
                    break
            if not patient_text and messages:
                first = messages[0]
                patient_text = first.get("content", "") if isinstance(first, dict) else str(first)
        if len(patient_text) > 20 and disease:
            all_data.append({
                "text": patient_text[:500],
                "drug_group": f"[DISEASE] {disease}",  # Đánh dấu để xử lý sau
                "source": "meddies_vi",
                "is_vietnamese": True,
            })
            vi_count += 1
    print(f"  → Vietnamese samples: {vi_count}")
except Exception as e:
    print(f"  ⚠️ Lỗi: {e}")
    print("  → Dataset này sẽ được tải từ HuggingFace Hub, cần internet.")

# --- Tổng hợp ---
print("\n" + "=" * 60)
df_all = pd.DataFrame(all_data)
print(f"📊 TỔNG: {len(df_all)} samples")
print(f"\nTheo nguồn:")
print(df_all["source"].value_counts().to_string())

# Tách Vietnamese data (có disease thay vì drug group)
df_vi = df_all[df_all.get("is_vietnamese", False) == True].copy()
df_en = df_all[df_all.get("is_vietnamese", False) != True].copy()

print(f"\n📊 English data (có drug group): {len(df_en)}")
print(f"📊 Vietnamese data (có disease):  {len(df_vi)}")
print(f"\nDrug groups ({df_en['drug_group'].nunique()}):")
print(df_en["drug_group"].value_counts().to_string())

# %%
# ============================================================
# CELL 5: BALANCE & PREPARE DATASET
# ============================================================
# Chỉ dùng English data đã có drug group labels
df = df_en[["text", "drug_group"]].copy()
df = df.dropna()
df = df[df["text"].str.len() > 10]

# Bỏ classes quá nhỏ (< 20 samples)
class_counts = df["drug_group"].value_counts()
valid_classes = class_counts[class_counts >= 20].index
df = df[df["drug_group"].isin(valid_classes)]

# Cân bằng: max 1500 per class
MAX_PER_CLASS = 1500
df = df.groupby("drug_group").apply(
    lambda x: x.sample(min(len(x), MAX_PER_CLASS), random_state=42)
).reset_index(drop=True)

# Label encoding
label_list = sorted(df["drug_group"].unique())
label2id = {label: idx for idx, label in enumerate(label_list)}
id2label = {idx: label for label, idx in label2id.items()}
NUM_CLASSES = len(label_list)

df["label"] = df["drug_group"].map(label2id)

# Train/Val/Test split (70/15/15)
train_df, temp_df = train_test_split(df, test_size=0.3, random_state=42, stratify=df["label"])
val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42, stratify=temp_df["label"])

print(f"✅ Dataset prepared!")
print(f"   Classes: {NUM_CLASSES}")
print(f"   Train:   {len(train_df)}")
print(f"   Val:     {len(val_df)}")
print(f"   Test:    {len(test_df)}")

# Lưu
train_df.to_csv(DATA_DIR / "train.csv", index=False)
val_df.to_csv(DATA_DIR / "val.csv", index=False)
test_df.to_csv(DATA_DIR / "test.csv", index=False)
with open(DATA_DIR / "label_map.json", "w") as f:
    json.dump({"label2id": label2id, "id2label": id2label}, f, ensure_ascii=False, indent=2)

print(f"\n📁 Saved to {DATA_DIR}")

# %%
# ============================================================
# CELL 6: TOKENIZER & DATASET CLASS
# ============================================================
from transformers import AutoTokenizer

# Dùng PhoBERT tokenizer (hoạt động tốt cho cả English text)
MODEL_NAME = "vinai/phobert-base-v2"
MAX_LEN = 256

print(f"📝 Loading tokenizer: {MODEL_NAME}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


class DrugTextDataset(Dataset):
    """Custom Dataset cho drug group classification."""

    def __init__(self, dataframe, tokenizer, max_len):
        self.texts = dataframe["text"].values
        self.labels = dataframe["label"].values
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]

        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt",
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "label": torch.tensor(label, dtype=torch.long),
        }


# Tạo DataLoaders
BATCH_SIZE = 32

train_dataset = DrugTextDataset(train_df, tokenizer, MAX_LEN)
val_dataset = DrugTextDataset(val_df, tokenizer, MAX_LEN)
test_dataset = DrugTextDataset(test_df, tokenizer, MAX_LEN)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

print(f"✅ DataLoaders ready: {len(train_loader)} train batches, {len(val_loader)} val batches")

# %%
# ============================================================
# CELL 7: MODEL DEFINITION
# ============================================================
from transformers import AutoModel


class DrugGroupClassifier(nn.Module):
    """
    PhoBERT + Classification Head
    Fine-tune PhoBERT base cho multi-class drug group classification.
    """

    def __init__(self, model_name, num_classes, dropout=0.3):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        hidden_size = self.bert.config.hidden_size  # 768

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 256),
            nn.GELU(),
            nn.LayerNorm(256),
            nn.Dropout(dropout / 2),
            nn.Linear(256, num_classes),
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        # Lấy [CLS] token embedding
        cls_output = outputs.last_hidden_state[:, 0, :]
        logits = self.classifier(cls_output)
        return logits


model = DrugGroupClassifier(MODEL_NAME, NUM_CLASSES).to(DEVICE)
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"✅ Model loaded: {MODEL_NAME}")
print(f"   Total params:     {total_params:,}")
print(f"   Trainable params: {trainable_params:,}")

# %%
# ============================================================
# CELL 8: TRAINING
# ============================================================
EPOCHS = 5
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01

# Class weights để xử lý imbalance
class_weights = []
for label_name in label_list:
    count = len(train_df[train_df["drug_group"] == label_name])
    class_weights.append(1.0 / max(count, 1))
class_weights = torch.FloatTensor(class_weights)
class_weights = class_weights / class_weights.sum() * NUM_CLASSES
class_weights = class_weights.to(DEVICE)

criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS)

best_val_f1 = 0.0
history = {"train_loss": [], "val_loss": [], "val_f1": [], "val_acc": []}

print(f"\n🚀 Training for {EPOCHS} epochs...")
print(f"   LR: {LEARNING_RATE}, Batch: {BATCH_SIZE}, Max Len: {MAX_LEN}")
print("=" * 70)

for epoch in range(EPOCHS):
    # --- Train ---
    model.train()
    train_loss = 0.0
    train_steps = 0

    for batch in train_loader:
        input_ids = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        labels = batch["label"].to(DEVICE)

        optimizer.zero_grad()
        logits = model(input_ids, attention_mask)
        loss = criterion(logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        train_loss += loss.item()
        train_steps += 1

    scheduler.step()
    avg_train_loss = train_loss / train_steps

    # --- Validate ---
    model.eval()
    val_loss = 0.0
    val_steps = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels = batch["label"].to(DEVICE)

            logits = model(input_ids, attention_mask)
            loss = criterion(logits, labels)
            val_loss += loss.item()
            val_steps += 1

            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_val_loss = val_loss / val_steps
    val_acc = accuracy_score(all_labels, all_preds)
    val_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)

    history["train_loss"].append(avg_train_loss)
    history["val_loss"].append(avg_val_loss)
    history["val_f1"].append(val_f1)
    history["val_acc"].append(val_acc)

    # Save best model
    improved = ""
    if val_f1 > best_val_f1:
        best_val_f1 = val_f1
        torch.save(model.state_dict(), MODEL_DIR / "best_model.pt")
        improved = " ⭐ BEST"

    print(f"Epoch {epoch+1}/{EPOCHS} | "
          f"Train Loss: {avg_train_loss:.4f} | "
          f"Val Loss: {avg_val_loss:.4f} | "
          f"Val Acc: {val_acc:.4f} | "
          f"Val F1: {val_f1:.4f}{improved}")

print("\n" + "=" * 70)
print(f"🏆 Best Val F1: {best_val_f1:.4f}")

# %%
# ============================================================
# CELL 9: EVALUATION ON TEST SET
# ============================================================
print("\n📊 Evaluating on Test Set...")
model.load_state_dict(torch.load(MODEL_DIR / "best_model.pt"))
model.eval()

all_preds = []
all_labels = []

with torch.no_grad():
    for batch in test_loader:
        input_ids = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        labels = batch["label"].to(DEVICE)

        logits = model(input_ids, attention_mask)
        preds = torch.argmax(logits, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# Metrics
test_acc = accuracy_score(all_labels, all_preds)
test_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
test_prec = precision_score(all_labels, all_preds, average="macro", zero_division=0)
test_rec = recall_score(all_labels, all_preds, average="macro", zero_division=0)

print(f"\n{'='*50}")
print(f"  TEST RESULTS")
print(f"{'='*50}")
print(f"  Accuracy:  {test_acc:.4f}")
print(f"  Precision: {test_prec:.4f}")
print(f"  Recall:    {test_rec:.4f}")
print(f"  F1 Macro:  {test_f1:.4f}")
print(f"{'='*50}")

# Classification Report
report = classification_report(all_labels, all_preds,
                               target_names=label_list, zero_division=0)
print(f"\n📋 Classification Report:\n{report}")

# Save report
with open(RESULTS_DIR / "classification_report.txt", "w") as f:
    f.write(f"Test Accuracy:  {test_acc:.4f}\n")
    f.write(f"Test Precision: {test_prec:.4f}\n")
    f.write(f"Test Recall:    {test_rec:.4f}\n")
    f.write(f"Test F1 Macro:  {test_f1:.4f}\n\n")
    f.write(report)

# Confusion Matrix
cm = confusion_matrix(all_labels, all_preds)
cm_df = pd.DataFrame(cm, index=label_list, columns=label_list)
cm_df.to_csv(RESULTS_DIR / "confusion_matrix.csv")

# Training history
pd.DataFrame(history).to_csv(RESULTS_DIR / "training_history.csv", index=False)

# Save label map
with open(MODEL_DIR / "label_map.json", "w") as f:
    json.dump({"label2id": label2id, "id2label": id2label}, f, ensure_ascii=False, indent=2)

# Save tokenizer config
tokenizer.save_pretrained(MODEL_DIR / "tokenizer")

print(f"\n✅ Kết quả lưu tại {RESULTS_DIR}")
print(f"✅ Model lưu tại {MODEL_DIR}")

# %%
# ============================================================
# CELL 10: INFERENCE FUNCTION (Copy vào backend)
# ============================================================
def predict_drug_groups(text: str, top_k: int = 3) -> list[dict]:
    """
    Dự đoán nhóm thuốc từ text.
    Copy hàm này vào backend/ml/inference.py
    """
    model.eval()
    encoding = tokenizer(
        text,
        truncation=True,
        padding="max_length",
        max_length=MAX_LEN,
        return_tensors="pt",
    )
    input_ids = encoding["input_ids"].to(DEVICE)
    attention_mask = encoding["attention_mask"].to(DEVICE)

    with torch.no_grad():
        logits = model(input_ids, attention_mask)
        probs = torch.softmax(logits, dim=1).squeeze()

    top_probs, top_indices = torch.topk(probs, k=min(top_k, NUM_CLASSES))

    results = []
    for prob, idx in zip(top_probs.cpu().numpy(), top_indices.cpu().numpy()):
        results.append({
            "drug_group": id2label[int(idx)],
            "confidence": float(prob),
            "rank": len(results) + 1,
        })
    return results


# --- Test thử ---
test_cases = [
    "Patient has high fever for 3 days, productive cough with yellow sputum, chest pain",
    "Severe headache, nausea, sensitivity to light, pulsating pain on one side",
    "Blood pressure 160/95 mmHg, dizziness, mild headache",
    "Burning sensation during urination, frequent urination, lower abdominal pain",
    "Joint pain, swelling in both knees, morning stiffness lasting 1 hour",
]

print("\n🧪 Demo predictions:")
print("=" * 70)
for text in test_cases:
    results = predict_drug_groups(text, top_k=3)
    print(f"\n📝 Input: \"{text[:80]}...\"")
    for r in results:
        bar = "█" * int(r["confidence"] * 30)
        print(f"   [{r['rank']}] {r['drug_group']:<25} {r['confidence']:.2%}  {bar}")

# %%
# ============================================================
# CELL 11: PACKAGE MODEL FOR DOWNLOAD
# ============================================================
import shutil

# Tạo archive để download
archive_name = "drugpred-model"
shutil.make_archive(
    str(WORKING_DIR / archive_name),
    "zip",
    root_dir=WORKING_DIR,
    base_dir="model",
)
print(f"\n📦 Model packaged: {WORKING_DIR / archive_name}.zip")
print("   Download từ Output tab bên phải →")
print("\n🎉 DONE! Copy file model vào backend/ml/models/weights/")
