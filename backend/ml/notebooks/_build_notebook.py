"""
Builder cho Kaggle training notebook của Drug-Pred AI.

Chạy:  python backend/ml/notebooks/_build_notebook.py
Sinh ra: backend/ml/notebooks/drugpred_kaggle_train.ipynb

File này là "source of truth" để regen notebook (tránh sửa JSON tay).
"""
import nbformat as nbf
from pathlib import Path

nb = nbf.v4.new_notebook()
cells = []


def md(src):
    cells.append(nbf.v4.new_markdown_cell(src.strip("\n")))


def code(src):
    cells.append(nbf.v4.new_code_cell(src.strip("\n")))


# ────────────────────────────────────────────────────────────────────
md(r'''
# 🏥 Drug-Pred AI — Kaggle Training Notebook

Dự đoán **nhóm thuốc** từ mô tả bệnh án (đa ngữ Anh + Việt).

Notebook này **kiểm tra (validate) và xử lý dữ liệu** từ nhiều dataset thuốc có
cấu trúc khác nhau, gộp về một schema chung, rồi train một bộ phân loại transformer.

## ⚙️ Setup trên Kaggle
1. **New Notebook** → Settings:
   - **Accelerator**: GPU T4 x2 (hoặc P100)
   - **Internet**: **ON**  ← bắt buộc để tải dữ liệu tiếng Việt (HuggingFace) + tải model
2. **+ Add Data** (panel phải), thêm các dataset:
   - `saisumanthv/medicine-recommendation-system-dataset`
   - `jessicali9530/kuc-hackathon-winter-2018`  (UCI Drug Review)
   - `singhnavjot2062001/11000-medicine-details`
   - `ddrbcn/openfda-drug-labeling`  *(tùy chọn, ~1.7GB)*
3. **Run All**.

## 📤 Output (`/kaggle/working/`)
- `data/` → train/val/test.csv + `label_map.json` + `data_card.md`
- `model/` → `best_model.pt` + tokenizer + label map
- `results/` → classification report, confusion matrix, history, plots

## 🧱 Triết lý xử lý dữ liệu
Mỗi nguồn có schema khác nhau → mỗi loader **tự dò cột**, **làm sạch text**,
**ánh xạ thuốc→nhóm** bằng một bảng taxonomy **duy nhất**, và **in báo cáo chất
lượng** (đã match bao nhiêu, rớt bao nhiêu). Cuối cùng gộp → dedup → cân bằng → split.
''')

# ── CONFIG / INSTALL ────────────────────────────────────────────────
md(r'''
## 0. Cài đặt & cấu hình

> `transformers`/`torch` đã có sẵn trên Kaggle; chỉ cài thêm khi thiếu.
''')

code(r'''
import subprocess, sys

def pip(*pkgs):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *pkgs])

for mod, pkg in [("transformers", "transformers"), ("datasets", "datasets"),
                 ("sklearn", "scikit-learn")]:
    try:
        __import__(mod)
    except Exception:
        print("installing", pkg); pip(pkg)

# peft chỉ cần khi USE_LORA=True (xem config)
print("✅ deps ready")
''')

code(r'''
import os, re, json, html, random, warnings
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
import torch

warnings.filterwarnings("ignore")

# ===================== CONFIG =====================
SEED          = 42
LABEL_LEVEL   = "category"        # "category" = ít lớp, dễ học (khớp drug_groups.category)
                                  # "subgroup" = chi tiết, khớp drug_groups.name
MODEL_NAME    = "xlm-roberta-base"  # ĐA NGỮ: train data English + infer tiếng Việt KHÔNG cần dịch.
                                    # Đổi "vinai/phobert-base-v2" nếu chỉ dùng dữ liệu tiếng Việt
MAX_LEN       = 256
BATCH_SIZE    = 32
EPOCHS        = 10
LR            = 5e-4
WEIGHT_DECAY  = 0.01
WARMUP_RATIO  = 0.1
PATIENCE      = 3                 # early stopping theo macro-F1
MAX_PER_CLASS = 1500             # cân bằng: tối đa N mẫu/lớp
MIN_PER_CLASS = 20               # bỏ lớp quá hiếm
USE_LORA      = True            # True = fine-tune nhẹ bằng peft/LoRA
USE_AMP       = True             # mixed precision (nhanh hơn trên T4)
# ==================================================

random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

KAGGLE_INPUT = Path("/kaggle/input")
WORKING      = Path("/kaggle/working")
DATA_DIR     = WORKING / "data"
MODEL_DIR    = WORKING / "model"
RESULTS_DIR  = WORKING / "results"
for d in (DATA_DIR, MODEL_DIR, RESULTS_DIR):
    d.mkdir(parents=True, exist_ok=True)

print("🖥️ Device:", DEVICE)
if torch.cuda.is_available():
    print("   GPU:", torch.cuda.get_device_name(0))
print("🎯 LABEL_LEVEL:", LABEL_LEVEL, "| MODEL:", MODEL_NAME)
''')

# ── TAXONOMY ────────────────────────────────────────────────────────
md(r'''
## 1. Taxonomy: thuốc → nhóm thuốc (nguồn chân lý duy nhất)

Bảng dưới dùng định dạng **`"Category - Subgroup"`** để khớp với `drug_groups` trong
`schema.sql` (`category` = "Kháng sinh", `name` = "Kháng sinh - Penicillin").
`LABEL_LEVEL` quyết định lấy mức nào làm nhãn train.

> ⚠️ Hai file cũ (`data_pipeline.py` mức chi tiết, `kaggle_notebook.py` mức thô) **lệch nhau** —
> notebook này hợp nhất lại để nhãn nhất quán với DB.
''')

code(r'''
DRUG_TO_GROUP = {
    # ===== KHÁNG SINH =====
    "Amoxicillin": "Kháng sinh - Penicillin", "Ampicillin": "Kháng sinh - Penicillin",
    "Penicillin": "Kháng sinh - Penicillin", "Augmentin": "Kháng sinh - Penicillin",
    "Amoxiclav": "Kháng sinh - Penicillin", "Cloxacillin": "Kháng sinh - Penicillin",
    "Azithromycin": "Kháng sinh - Macrolide", "Erythromycin": "Kháng sinh - Macrolide",
    "Clarithromycin": "Kháng sinh - Macrolide", "Roxithromycin": "Kháng sinh - Macrolide",
    "Cephalexin": "Kháng sinh - Cephalosporin", "Ceftriaxone": "Kháng sinh - Cephalosporin",
    "Cefuroxime": "Kháng sinh - Cephalosporin", "Cefixime": "Kháng sinh - Cephalosporin",
    "Cefpodoxime": "Kháng sinh - Cephalosporin", "Cefadroxil": "Kháng sinh - Cephalosporin",
    "Ciprofloxacin": "Kháng sinh - Fluoroquinolone", "Levofloxacin": "Kháng sinh - Fluoroquinolone",
    "Moxifloxacin": "Kháng sinh - Fluoroquinolone", "Ofloxacin": "Kháng sinh - Fluoroquinolone",
    "Norfloxacin": "Kháng sinh - Fluoroquinolone",
    "Doxycycline": "Kháng sinh - Tetracycline", "Tetracycline": "Kháng sinh - Tetracycline",
    "Minocycline": "Kháng sinh - Tetracycline",
    "Metronidazole": "Kháng sinh - Nitroimidazole", "Tinidazole": "Kháng sinh - Nitroimidazole",
    "Trimethoprim": "Kháng sinh - Sulfonamide", "Sulfamethoxazole": "Kháng sinh - Sulfonamide",
    "Cotrimoxazole": "Kháng sinh - Sulfonamide",
    "Clindamycin": "Kháng sinh - Lincosamide",
    "Vancomycin": "Kháng sinh - Glycopeptide",
    "Nitrofurantoin": "Kháng sinh - Nitrofuran",
    "Gentamicin": "Kháng sinh - Aminoglycoside", "Amikacin": "Kháng sinh - Aminoglycoside",

    # ===== GIẢM ĐAU / HẠ SỐT =====
    "Ibuprofen": "Giảm đau - NSAID", "Naproxen": "Giảm đau - NSAID",
    "Diclofenac": "Giảm đau - NSAID", "Celecoxib": "Giảm đau - NSAID",
    "Meloxicam": "Giảm đau - NSAID", "Aspirin": "Giảm đau - NSAID",
    "Indomethacin": "Giảm đau - NSAID", "Ketorolac": "Giảm đau - NSAID",
    "Etoricoxib": "Giảm đau - NSAID", "Piroxicam": "Giảm đau - NSAID",
    "Acetaminophen": "Giảm đau - Paracetamol", "Paracetamol": "Giảm đau - Paracetamol",
    "Tramadol": "Giảm đau - Opioid", "Codeine": "Giảm đau - Opioid",
    "Morphine": "Giảm đau - Opioid", "Oxycodone": "Giảm đau - Opioid",
    "Fentanyl": "Giảm đau - Opioid",

    # ===== TIM MẠCH =====
    "Lisinopril": "Tim mạch - ACE inhibitor", "Enalapril": "Tim mạch - ACE inhibitor",
    "Ramipril": "Tim mạch - ACE inhibitor", "Captopril": "Tim mạch - ACE inhibitor",
    "Perindopril": "Tim mạch - ACE inhibitor",
    "Losartan": "Tim mạch - ARB", "Valsartan": "Tim mạch - ARB",
    "Irbesartan": "Tim mạch - ARB", "Telmisartan": "Tim mạch - ARB",
    "Amlodipine": "Tim mạch - Chẹn kênh Canxi", "Nifedipine": "Tim mạch - Chẹn kênh Canxi",
    "Diltiazem": "Tim mạch - Chẹn kênh Canxi", "Felodipine": "Tim mạch - Chẹn kênh Canxi",
    "Metoprolol": "Tim mạch - Beta blocker", "Atenolol": "Tim mạch - Beta blocker",
    "Propranolol": "Tim mạch - Beta blocker", "Bisoprolol": "Tim mạch - Beta blocker",
    "Carvedilol": "Tim mạch - Beta blocker", "Nebivolol": "Tim mạch - Beta blocker",
    "Hydrochlorothiazide": "Tim mạch - Lợi tiểu", "Furosemide": "Tim mạch - Lợi tiểu",
    "Spironolactone": "Tim mạch - Lợi tiểu", "Indapamide": "Tim mạch - Lợi tiểu",
    "Digoxin": "Tim mạch - Glycoside",

    # ===== TIÊU HÓA =====
    "Omeprazole": "Tiêu hóa - PPI", "Esomeprazole": "Tiêu hóa - PPI",
    "Pantoprazole": "Tiêu hóa - PPI", "Lansoprazole": "Tiêu hóa - PPI",
    "Rabeprazole": "Tiêu hóa - PPI",
    "Ranitidine": "Tiêu hóa - H2 blocker", "Famotidine": "Tiêu hóa - H2 blocker",
    "Loperamide": "Tiêu hóa - Chống tiêu chảy",
    "Domperidone": "Tiêu hóa - Chống nôn", "Ondansetron": "Tiêu hóa - Chống nôn",
    "Metoclopramide": "Tiêu hóa - Chống nôn",

    # ===== NỘI TIẾT =====
    "Metformin": "Nội tiết - Đái tháo đường", "Glipizide": "Nội tiết - Đái tháo đường",
    "Gliclazide": "Nội tiết - Đái tháo đường", "Glimepiride": "Nội tiết - Đái tháo đường",
    "Insulin": "Nội tiết - Đái tháo đường", "Sitagliptin": "Nội tiết - Đái tháo đường",
    "Pioglitazone": "Nội tiết - Đái tháo đường", "Empagliflozin": "Nội tiết - Đái tháo đường",
    "Levothyroxine": "Nội tiết - Tuyến giáp",

    # ===== HÔ HẤP =====
    "Salbutamol": "Hô hấp - Giãn phế quản", "Albuterol": "Hô hấp - Giãn phế quản",
    "Ipratropium": "Hô hấp - Giãn phế quản", "Theophylline": "Hô hấp - Giãn phế quản",
    "Montelukast": "Hô hấp - Kháng leukotriene",
    "Budesonide": "Hô hấp - Corticoid hít", "Fluticasone": "Hô hấp - Corticoid hít",
    "Dextromethorphan": "Hô hấp - Giảm ho", "Guaifenesin": "Hô hấp - Long đờm",
    "Bromhexine": "Hô hấp - Long đờm", "Acetylcysteine": "Hô hấp - Long đờm",

    # ===== THẦN KINH / TÂM THẦN =====
    "Sertraline": "Thần kinh - SSRI", "Fluoxetine": "Thần kinh - SSRI",
    "Escitalopram": "Thần kinh - SSRI", "Citalopram": "Thần kinh - SSRI",
    "Paroxetine": "Thần kinh - SSRI",
    "Venlafaxine": "Thần kinh - SNRI", "Duloxetine": "Thần kinh - SNRI",
    "Amitriptyline": "Thần kinh - TCA", "Bupropion": "Thần kinh - Khác",
    "Gabapentin": "Thần kinh - Chống động kinh", "Pregabalin": "Thần kinh - Chống động kinh",
    "Carbamazepine": "Thần kinh - Chống động kinh", "Valproic Acid": "Thần kinh - Chống động kinh",
    "Lamotrigine": "Thần kinh - Chống động kinh", "Levetiracetam": "Thần kinh - Chống động kinh",
    "Phenytoin": "Thần kinh - Chống động kinh", "Topiramate": "Thần kinh - Chống động kinh",
    "Diazepam": "Thần kinh - Benzodiazepine", "Lorazepam": "Thần kinh - Benzodiazepine",
    "Alprazolam": "Thần kinh - Benzodiazepine", "Clonazepam": "Thần kinh - Benzodiazepine",
    "Zolpidem": "Thần kinh - An thần",
    "Sumatriptan": "Thần kinh - Triptan", "Rizatriptan": "Thần kinh - Triptan",

    # ===== DỊ ỨNG =====
    "Cetirizine": "Dị ứng - Kháng histamine", "Loratadine": "Dị ứng - Kháng histamine",
    "Fexofenadine": "Dị ứng - Kháng histamine", "Desloratadine": "Dị ứng - Kháng histamine",
    "Diphenhydramine": "Dị ứng - Kháng histamine", "Chlorpheniramine": "Dị ứng - Kháng histamine",
    "Hydroxyzine": "Dị ứng - Kháng histamine",

    # ===== CORTICOSTEROID =====
    "Prednisolone": "Chống viêm - Corticosteroid", "Prednisone": "Chống viêm - Corticosteroid",
    "Dexamethasone": "Chống viêm - Corticosteroid", "Hydrocortisone": "Chống viêm - Corticosteroid",
    "Methylprednisolone": "Chống viêm - Corticosteroid", "Betamethasone": "Chống viêm - Corticosteroid",

    # ===== CHUYỂN HÓA / MỠ MÁU =====
    "Atorvastatin": "Chuyển hóa - Statin", "Simvastatin": "Chuyển hóa - Statin",
    "Rosuvastatin": "Chuyển hóa - Statin", "Pravastatin": "Chuyển hóa - Statin",
    "Fenofibrate": "Chuyển hóa - Fibrate",

    # ===== HUYẾT HỌC / CHỐNG ĐÔNG =====
    "Warfarin": "Huyết học - Chống đông", "Heparin": "Huyết học - Chống đông",
    "Clopidogrel": "Huyết học - Chống kết tập", "Rivaroxaban": "Huyết học - Chống đông",
    "Apixaban": "Huyết học - Chống đông", "Enoxaparin": "Huyết học - Chống đông",

    # ===== KHÁNG NẤM / VIRUS (DA LIỄU) =====
    "Clotrimazole": "Da liễu - Kháng nấm", "Fluconazole": "Da liễu - Kháng nấm",
    "Ketoconazole": "Da liễu - Kháng nấm", "Itraconazole": "Da liễu - Kháng nấm",
    "Nystatin": "Da liễu - Kháng nấm", "Terbinafine": "Da liễu - Kháng nấm",
    "Acyclovir": "Da liễu - Kháng virus", "Valacyclovir": "Da liễu - Kháng virus",
    "Oseltamivir": "Da liễu - Kháng virus",

    # ===== CƠ XƯƠNG KHỚP =====
    "Allopurinol": "Cơ xương khớp - Chống gout", "Colchicine": "Cơ xương khớp - Chống gout",
    "Febuxostat": "Cơ xương khớp - Chống gout",
    "Methotrexate": "Cơ xương khớp - DMARD",
}

# ─── Helpers: chuẩn hóa & ánh xạ ───────────────────────────────────
import re as _re
_WS = _re.compile(r"\s+")

def norm_key(s):
    return _WS.sub(" ", str(s)).strip().lower()

# Regex alternation: khớp THEO RANH GIỚI TỪ, ưu tiên tên dài trước
#   → tránh lỗi "Penicillin" lọt vào "Penicillinase", và bắt được dạng kết hợp.
_KEYS = sorted(DRUG_TO_GROUP, key=len, reverse=True)
DRUG_RE = _re.compile(r"\b(" + "|".join(_re.escape(k) for k in _KEYS) + r")\b", _re.I)
_LOWER = {k.lower(): v for k, v in DRUG_TO_GROUP.items()}

def get_group(name):
    """Tên thuốc đơn -> nhóm. Exact (lower) rồi regex theo ranh giới từ."""
    if not isinstance(name, str) or not name.strip():
        return None
    n = name.strip().lower()
    if n in _LOWER:
        return _LOWER[n]
    m = DRUG_RE.search(name)
    return _LOWER[m.group(1).lower()] if m else None

def find_groups(text):
    """Tìm MỌI thuốc xuất hiện trong free-text (composition, indications...)."""
    if not isinstance(text, str):
        return []
    return [_LOWER[m.lower()] for m in DRUG_RE.findall(text)]

def to_category(group):
    """'Kháng sinh - Penicillin' -> 'Kháng sinh'. Chuỗi đã là category -> giữ nguyên."""
    return group.split(" - ")[0] if isinstance(group, str) else None

CATEGORIES = sorted({to_category(v) for v in DRUG_TO_GROUP.values()})
SUBGROUPS  = sorted(set(DRUG_TO_GROUP.values()))
print(f"📋 {len(DRUG_TO_GROUP)} thuốc → {len(SUBGROUPS)} subgroups → {len(CATEGORIES)} categories")
for c in CATEGORIES:
    print("   •", c)
''')

# ── VIETNAMESE MAPPING ─────────────────────────────────────────────
md(r'''
### 1b. Ánh xạ tiếng Việt: triệu chứng/bệnh → nhóm thuốc

Dữ liệu tiếng Việt (HoangHa/medical-data, ViMedAQA) gắn nhãn theo **bệnh**, không phải
tên thuốc → không match bảng trên. Ta dùng **từ khóa tiếng Việt** để suy ra *category*.
Mẫu tiếng Việt vì thế chỉ dùng được ở mức `category` (không tới subgroup).

> Token unmatched sẽ được in ra để bạn mở rộng `VI_KEYWORD_TO_CATEGORY`.
''')

code(r'''
# category -> danh sách từ khóa tiếng Việt (không dấu hoa/thường đều ok vì norm_key đã lower)
VI_KEYWORD_TO_CATEGORY = {
    "Kháng sinh":   ["nhiễm khuẩn", "nhiễm trùng", "viêm họng", "viêm amidan", "viêm phổi",
                      "viêm xoang", "viêm tai giữa", "viêm bàng quang", "nhiễm trùng tiểu",
                      "nhiễm trùng đường tiểu", "áp xe", "mưng mủ", "lậu"],
    "Hô hấp":       ["hen suyễn", "hen phế quản", "khó thở", "khò khè", "ho khan", "ho có đờm",
                      "viêm phế quản", "copd", "tắc nghẽn phổi"],
    "Giảm đau":     ["đau đầu", "nhức đầu", "đau lưng", "đau cơ", "đau bụng kinh", "sốt cao",
                      "hạ sốt", "đau răng", "nhức mỏi"],
    "Tim mạch":     ["tăng huyết áp", "cao huyết áp", "huyết áp cao", "đau thắt ngực",
                      "suy tim", "rối loạn nhịp tim", "nhồi máu"],
    "Tiêu hóa":     ["đau dạ dày", "viêm dạ dày", "trào ngược", "ợ chua", "ợ nóng",
                      "loét dạ dày", "tiêu chảy", "buồn nôn", "nôn ói", "khó tiêu", "đầy hơi"],
    "Nội tiết":     ["tiểu đường", "đái tháo đường", "đường huyết cao", "tuyến giáp",
                      "suy giáp", "cường giáp"],
    "Dị ứng":       ["dị ứng", "mề đay", "mẩn ngứa", "nổi mẩn", "viêm mũi dị ứng", "sổ mũi",
                      "hắt hơi", "ngứa"],
    "Thần kinh":    ["trầm cảm", "lo âu", "mất ngủ", "căng thẳng", "stress", "động kinh",
                      "co giật", "đau nửa đầu", "migraine"],
    "Cơ xương khớp": ["gout", "gút", "viêm khớp", "thoái hóa khớp", "đau khớp", "sưng khớp",
                      "cứng khớp"],
    "Da liễu":      ["nấm da", "hắc lào", "lang ben", "nấm móng", "zona", "mụn rộp", "nấm candida"],
    "Chuyển hóa":   ["mỡ máu", "rối loạn lipid", "cholesterol cao", "tăng mỡ máu"],
}

def vi_category(text):
    """Suy category từ text tiếng Việt; chọn category có nhiều từ khóa khớp nhất."""
    t = norm_key(text)
    best, best_n = None, 0
    for cat, kws in VI_KEYWORD_TO_CATEGORY.items():
        n = sum(1 for k in kws if k in t)
        if n > best_n:
            best, best_n = cat, n
    return best if best_n > 0 else None

print("🇻🇳 VI keyword map:", len(VI_KEYWORD_TO_CATEGORY), "categories")
''')

# ── CLEANING + EXTRACTORS ──────────────────────────────────────────
md(r'''
## 2. Làm sạch text + bộ trích xuất tự dò cột

`extract_from_df` thử lần lượt 4 mẫu schema phổ biến và lấy mẫu nào ra dòng:
1. **review + tên thuốc** (UCI Drug Review) — lọc `rating ≥ 7`
2. **uses + tên/thành phần** (11000 Medicine Details)
3. **symptom\* + thuốc** (Medicine Recommendation)
4. **disease/condition + medication** (dạng bảng bệnh→thuốc)
''')

code(r'''
_TAG = _re.compile(r"<[^>]+>")
_URL = _re.compile(r"http\S+|www\.\S+")

def clean_text(s):
    if not isinstance(s, str):
        return ""
    s = html.unescape(s)          # &#039; -> '  &amp; -> &
    s = _TAG.sub(" ", s)          # bỏ thẻ HTML
    s = _URL.sub(" ", s)          # bỏ URL
    s = s.replace("\r", " ").replace("\n", " ")
    s = _WS.sub(" ", s).strip()
    return s

def _find(df, *subs):
    return [c for c in df.columns if any(s in c.lower() for s in subs)]

def _exact(df, names):
    return [c for c in df.columns if c.lower().strip() in names]

def _row(text, drug_name, group, source, lang="en"):
    return {"text": text[:500], "drug_name": str(drug_name)[:80],
            "drug_group": group, "source": source, "lang": lang}

def extract_from_df(df, source):
    """Trả về list[dict] đã chuẩn hóa từ 1 DataFrame bất kỳ."""
    rows = []
    sym     = _find(df, "symptom")
    drug    = _find(df, "drug", "medication", "medicine")
    name    = _exact(df, {"medicine name", "medicine_name", "name", "drug_name", "drugname"})
    uses    = _exact(df, {"uses", "use", "indication", "indications", "description"})
    comp    = _find(df, "composition", "salt", "ingredient")
    review  = _exact(df, {"review", "reviews", "benefits_review"})
    disease = _find(df, "disease", "condition", "prognosis")
    rating  = _exact(df, {"rating"})

    # (1) UCI review + tên thuốc
    if review and (name or drug):
        ncol = (name or drug)[0]
        sub = df
        if rating:
            sub = sub[pd.to_numeric(sub[rating[0]], errors="coerce").fillna(0) >= 7]
        for _, r in sub.iterrows():
            g = get_group(str(r.get(ncol, "")))
            t = clean_text(str(r.get(review[0], "")))
            if g and len(t) > 20:
                rows.append(_row(t, r.get(ncol, ""), g, source))
        return rows

    # (2) 11k details: uses + name/composition
    if uses and (name or comp):
        for _, r in df.iterrows():
            g = get_group(str(r.get(name[0], ""))) if name else None
            if not g and comp:
                gg = find_groups(str(r.get(comp[0], "")))
                g = gg[0] if gg else None
            t = clean_text(str(r.get(uses[0], "")))
            if g and len(t) > 15:
                rows.append(_row(t, (r.get(name[0], "") if name else ""), g, source))
        return rows

    # (3) symptom* -> thuốc
    if sym and drug:
        for _, r in df.iterrows():
            syl = [clean_text(str(r[c])).replace("_", " ") for c in sym
                   if norm_key(r.get(c, "")) not in ("", "nan", "none")]
            if not syl:
                continue
            t = ", ".join(syl)
            g = None
            for c in drug:
                g = get_group(str(r.get(c, "")))
                if not g:
                    gg = find_groups(str(r.get(c, "")))
                    g = gg[0] if gg else None
                if g:
                    break
            if g:
                rows.append(_row(t, "", g, source))
        return rows

    # (4) disease/condition + medication
    if disease and drug:
        for _, r in df.iterrows():
            t = clean_text(str(r.get(disease[0], "")))
            gg = find_groups(str(r.get(drug[0], "")))
            if gg and len(t) > 3:
                rows.append(_row(t, r.get(drug[0], ""), gg[0], source))
        return rows

    return rows
''')

# ── OPENFDA + VI LOADERS ───────────────────────────────────────────
md(r'''
## 3. Loader riêng: OpenFDA (JSON lớn) & dữ liệu tiếng Việt (HuggingFace)
''')

code(r'''
def load_openfda(base, cap_per_file=20000):
    """Stream toàn bộ JSON (không giới hạn 5 file như pipeline cũ); auto-giải nén zip."""
    rows = []
    files = list(base.rglob("*.json"))
    if not files:
        import zipfile
        ex = WORKING / "fda_extracted"; ex.mkdir(exist_ok=True)
        for zf in base.rglob("*.zip"):
            try:
                with zipfile.ZipFile(zf) as z:
                    z.extractall(ex)
            except Exception as e:
                print("   ⚠️ unzip:", e)
        files = list(ex.rglob("*.json"))
    for f in files:
        try:
            data = json.load(open(f, encoding="utf-8"))
        except Exception:
            continue
        res = data.get("results", data) if isinstance(data, dict) else data
        if isinstance(res, dict):
            res = [res]
        for rec in res[:cap_per_file]:
            if not isinstance(rec, dict):
                continue
            ind = rec.get("indications_and_usage", [])
            if isinstance(ind, list):
                ind = " ".join(map(str, ind))
            t = clean_text(str(ind))
            of = rec.get("openfda", {}) or {}
            names = of.get("generic_name") or of.get("brand_name") or []
            nm = names[0] if isinstance(names, list) and names else str(names)
            g = get_group(nm)
            if not g:
                gg = find_groups(t)
                g = gg[0] if gg else None
            if g and len(t) > 30:
                rows.append(_row(t, nm, g, "openfda"))
    return rows


def load_vietnamese():
    """HoangHa/medical-data (vietnamese). Cần Internet=ON. Nhãn -> category bằng từ khóa."""
    rows, unmatched = [], Counter()
    try:
        from datasets import load_dataset
        ds = load_dataset("HoangHa/medical-data", "vietnamese", split="train",
                          trust_remote_code=True)
        for r in ds:
            msgs = r.get("messages", [])
            txt = ""
            if isinstance(msgs, list):
                for m in msgs:
                    if isinstance(m, dict) and m.get("role", "").lower() in ("user", "patient", "human"):
                        txt = m.get("content", ""); break
                if not txt and msgs:
                    txt = msgs[0].get("content", "") if isinstance(msgs[0], dict) else str(msgs[0])
            txt = clean_text(txt)
            disease = str(r.get("target_disease", ""))
            cat = vi_category(disease + " " + txt)
            if cat and len(txt) > 20:
                rows.append(_row(txt, "", cat, "meddies_vi", lang="vi"))
            elif disease:
                unmatched[disease] += 1
        print(f"   🇻🇳 HoangHa: +{len(rows)} mẫu. Top bệnh CHƯA map:",
              [d for d, _ in unmatched.most_common(8)])
    except Exception as e:
        print("   ⚠️ Bỏ qua dữ liệu VI (cần Internet=ON):", e)
    return rows
''')

# ── DISCOVER / MERGE ───────────────────────────────────────────────
md(r'''
## 4. Quét `/kaggle/input`, chạy loader, gộp + báo cáo chất lượng
''')

code(r'''
def discover():
    rows = []
    dirs = [d for d in sorted(KAGGLE_INPUT.iterdir()) if d.is_dir()] if KAGGLE_INPUT.exists() else []
    print("📁 Datasets có sẵn:", [d.name for d in dirs] or "— (chưa Add Data)")
    for d in dirs:
        before = len(rows)
        # JSON/ZIP -> OpenFDA
        if list(d.rglob("*.json")) or list(d.rglob("*.zip")):
            r = load_openfda(d); rows += r
            print(f"  [{d.name}] OpenFDA JSON → {len(r)} mẫu")
        # CSV/TSV -> extractor tự dò
        for f in list(d.rglob("*.csv")) + list(d.rglob("*.tsv")):
            try:
                sep = "\t" if f.suffix == ".tsv" else ","
                df = pd.read_csv(f, sep=sep, on_bad_lines="skip", low_memory=False)
            except Exception as e:
                print(f"  ⚠️ đọc {f.name}: {e}"); continue
            r = extract_from_df(df, d.name); rows += r
            print(f"  [{d.name}/{f.name}] {len(df)}×{len(df.columns)} → "
                  f"{len(r) if r else 'không khớp schema'}")
        print(f"  Σ {d.name}: +{len(rows) - before}\n")
    return rows

raw = discover()
raw += load_vietnamese()

df = pd.DataFrame(raw)
assert len(df) > 0, "❌ Không có dữ liệu! Kiểm tra: đã Add Data? Internet=ON cho phần VI?"

df["text"]       = df["text"].map(clean_text)
df["category"]   = df["drug_group"].map(to_category)
df["label_name"] = df["category"] if LABEL_LEVEL == "category" else df["drug_group"]

print("=" * 60)
print(f"📊 Tổng mẫu thô: {len(df)}")
print("\n— Theo nguồn —\n", df["source"].value_counts().to_string())
print("\n— Theo ngôn ngữ —\n", df["lang"].value_counts().to_string())
''')

code(r'''
# ─── Kiểm tra chất lượng: dedup, null, độ dài, phân bố lớp ───
print("🔍 KIỂM TRA CHẤT LƯỢNG\n" + "-" * 40)

n0 = len(df)
df["_key"] = df["text"].str.lower().map(lambda s: _WS.sub(" ", s).strip())
df = df[df["_key"].str.len() >= 12]                       # bỏ text quá ngắn
df = df.drop_duplicates("_key").drop(columns="_key")      # bỏ trùng (review hay lặp)
df = df.dropna(subset=["text", "label_name"]).reset_index(drop=True)
print(f"Sau dedup + lọc ngắn/null: {n0} → {len(df)} (-{n0 - len(df)})")

df["text_len"] = df["text"].str.len()
print(f"Độ dài text — min {df.text_len.min()}, mean {df.text_len.mean():.0f}, "
      f"median {df.text_len.median():.0f}, max {df.text_len.max()}")
print(f"\nSố lớp ({LABEL_LEVEL}): {df['label_name'].nunique()}")
print(df["label_name"].value_counts().to_string())
''')

# ── EDA PLOTS ──────────────────────────────────────────────────────
md(r'''
## 5. EDA — trực quan hóa (lưu vào `results/`)
''')

code(r'''
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

fig, ax = plt.subplots(2, 2, figsize=(15, 11))

vc = df["label_name"].value_counts()
ax[0, 0].barh(vc.index[::-1], vc.values[::-1], color="#2563eb")
ax[0, 0].set_title(f"Phân bố lớp ({LABEL_LEVEL}) — {df['label_name'].nunique()} lớp")

ax[0, 1].hist(df["text_len"].clip(upper=600), bins=40, color="#16a34a")
ax[0, 1].set_title("Độ dài text (ký tự)"); ax[0, 1].set_xlabel("len")

sc = df["source"].value_counts()
ax[1, 0].bar(sc.index, sc.values, color="#f59e0b")
ax[1, 0].set_title("Mẫu theo nguồn"); ax[1, 0].tick_params(axis="x", rotation=30)

lc = df["lang"].value_counts()
ax[1, 1].pie(lc.values, labels=lc.index, autopct="%1.1f%%",
             colors=["#3b82f6", "#ef4444"])
ax[1, 1].set_title("Ngôn ngữ (EN train / VI infer-domain)")

plt.tight_layout()
plt.savefig(RESULTS_DIR / "eda.png", dpi=110, bbox_inches="tight")
plt.show()
print("🖼️ Lưu", RESULTS_DIR / "eda.png")
''')

# ── BALANCE / SPLIT ────────────────────────────────────────────────
md(r'''
## 6. Lọc lớp hiếm → cân bằng → encode → split (70/15/15, stratified)
''')

code(r'''
from sklearn.model_selection import train_test_split

vc = df["label_name"].value_counts()
keep = vc[vc >= MIN_PER_CLASS].index
dropped = vc[vc < MIN_PER_CLASS]
if len(dropped):
    print(f"⚠️ Bỏ {len(dropped)} lớp < {MIN_PER_CLASS} mẫu:", list(dropped.index))
df = df[df["label_name"].isin(keep)].copy()

# Cân bằng: tối đa MAX_PER_CLASS mẫu/lớp
df = (df.groupby("label_name", group_keys=False)
        .apply(lambda x: x.sample(min(len(x), MAX_PER_CLASS), random_state=SEED))
        .reset_index(drop=True))

labels   = sorted(df["label_name"].unique())
label2id = {l: i for i, l in enumerate(labels)}
id2label = {i: l for l, i in label2id.items()}
NUM_CLASSES = len(labels)
df["label"] = df["label_name"].map(label2id)

train_df, tmp_df = train_test_split(df, test_size=0.30, random_state=SEED, stratify=df["label"])
val_df,  test_df = train_test_split(tmp_df, test_size=0.50, random_state=SEED, stratify=tmp_df["label"])
for s in (train_df, val_df, test_df):
    s.reset_index(drop=True, inplace=True)

print(f"✅ {NUM_CLASSES} lớp | train {len(train_df)} | val {len(val_df)} | test {len(test_df)}")

# Lưu artifacts
train_df.to_csv(DATA_DIR / "train.csv", index=False)
val_df.to_csv(DATA_DIR / "val.csv", index=False)
test_df.to_csv(DATA_DIR / "test.csv", index=False)
json.dump({"label2id": label2id, "id2label": id2label, "level": LABEL_LEVEL},
          open(DATA_DIR / "label_map.json", "w"), ensure_ascii=False, indent=2)

# Data card (báo cáo dữ liệu)
card = [f"# Data Card — Drug-Pred AI ({LABEL_LEVEL})", "",
        f"- Model: {MODEL_NAME}", f"- Tổng: {len(df)} | Lớp: {NUM_CLASSES}",
        f"- Split: {len(train_df)}/{len(val_df)}/{len(test_df)}", "",
        "## Nguồn", df["source"].value_counts().to_string(),
        "", "## Ngôn ngữ", df["lang"].value_counts().to_string(),
        "", "## Phân bố lớp", df["label_name"].value_counts().to_string()]
(DATA_DIR / "data_card.md").write_text("\n".join(card), encoding="utf-8")
print("📁 Lưu artifacts →", DATA_DIR)
''')

# ── TOKENIZER / DATASET ────────────────────────────────────────────
md(r'''
## 7. Tokenizer + Dataset + DataLoader
''')

code(r'''
from transformers import AutoTokenizer
from torch.utils.data import Dataset, DataLoader

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
print("📝 Tokenizer:", MODEL_NAME)

class DrugDataset(Dataset):
    def __init__(self, frame):
        self.texts  = frame["text"].astype(str).values
        self.labels = frame["label"].values
    def __len__(self):
        return len(self.texts)
    def __getitem__(self, i):
        enc = tokenizer(self.texts[i], truncation=True, padding="max_length",
                        max_length=MAX_LEN, return_tensors="pt")
        return {"input_ids": enc["input_ids"].squeeze(0),
                "attention_mask": enc["attention_mask"].squeeze(0),
                "label": torch.tensor(self.labels[i], dtype=torch.long)}

train_loader = DataLoader(DrugDataset(train_df), batch_size=BATCH_SIZE, shuffle=True,  num_workers=2)
val_loader   = DataLoader(DrugDataset(val_df),   batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
test_loader  = DataLoader(DrugDataset(test_df),  batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
print(f"✅ {len(train_loader)} train / {len(val_loader)} val / {len(test_loader)} test batches")
''')

# ── MODEL ──────────────────────────────────────────────────────────
md(r'''
## 8. Model — Transformer encoder + classification head (LoRA tùy chọn)

`forward` chấp nhận cả `input_ids` lẫn `inputs_embeds` để phục vụ **XAI** (gradient ×
embedding) ở mục 11.
''')

code(r'''
import torch.nn as nn
from transformers import AutoModel

class DrugClassifier(nn.Module):
    def __init__(self, name, num_classes, dropout=0.3, use_lora=True):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(name)
        if use_lora:
            from peft import LoraConfig, get_peft_model
            cfg = LoraConfig(r=32, lora_alpha=32, lora_dropout=0.05, bias="none",
                             target_modules=["query", "key", "value"],
                             task_type="FEATURE_EXTRACTION")
            self.encoder = get_peft_model(self.encoder, cfg)
            self.encoder.print_trainable_parameters()
        h = self.encoder.config.hidden_size
        self.head = nn.Sequential(
            nn.Dropout(dropout), nn.Linear(h, 256), nn.GELU(),
            nn.LayerNorm(256), nn.Dropout(dropout / 2), nn.Linear(256, num_classes))

    def forward(self, input_ids=None, attention_mask=None, inputs_embeds=None):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask,
                           inputs_embeds=inputs_embeds)
        cls = out.last_hidden_state[:, 0]          # <s> / [CLS]
        return self.head(cls)

if USE_LORA:
    try:
        import peft  # noqa
    except Exception:
        pip("peft")

model = DrugClassifier(MODEL_NAME, NUM_CLASSES, use_lora=USE_LORA).to(DEVICE)
tot = sum(p.numel() for p in model.parameters())
tr  = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"✅ Model {MODEL_NAME} | params {tot:,} | trainable {tr:,}")
''')

# ── TRAIN ──────────────────────────────────────────────────────────
md(r'''
## 9. Train — class weights + warmup + AMP + early stopping (macro-F1)
''')

code(r'''
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from sklearn.metrics import accuracy_score, f1_score

# Class weights (chống mất cân bằng) — tính trên TRAIN
counts = train_df["label"].value_counts().sort_index()
w = torch.tensor([1.0 / max(counts.get(i, 1), 1) for i in range(NUM_CLASSES)], dtype=torch.float)
w = (w / w.sum() * NUM_CLASSES).to(DEVICE)

criterion = nn.CrossEntropyLoss(weight=w)
optimizer = AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
total_steps = len(train_loader) * EPOCHS
scheduler = get_linear_schedule_with_warmup(
    optimizer, int(total_steps * WARMUP_RATIO), total_steps)
scaler = torch.cuda.amp.GradScaler(enabled=USE_AMP and torch.cuda.is_available())

@torch.no_grad()
def evaluate(loader):
    model.eval(); preds, gts, loss = [], [], 0.0
    for b in loader:
        ids, mask, y = b["input_ids"].to(DEVICE), b["attention_mask"].to(DEVICE), b["label"].to(DEVICE)
        logits = model(ids, mask)
        loss += criterion(logits, y).item()
        preds += logits.argmax(1).cpu().tolist(); gts += y.cpu().tolist()
    return (loss / len(loader), accuracy_score(gts, preds),
            f1_score(gts, preds, average="macro", zero_division=0), preds, gts)

history = {"train_loss": [], "val_loss": [], "val_acc": [], "val_f1": []}
best_f1, no_improve = 0.0, 0
print(f"🚀 Train {EPOCHS} epochs | LR {LR} | batch {BATCH_SIZE} | AMP {USE_AMP}\n" + "=" * 70)

for ep in range(1, EPOCHS + 1):
    model.train(); run = 0.0
    for b in train_loader:
        ids, mask, y = b["input_ids"].to(DEVICE), b["attention_mask"].to(DEVICE), b["label"].to(DEVICE)
        optimizer.zero_grad()
        with torch.cuda.amp.autocast(enabled=USE_AMP and torch.cuda.is_available()):
            loss = criterion(model(ids, mask), y)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer); scaler.update(); scheduler.step()
        run += loss.item()
    tr_loss = run / len(train_loader)
    v_loss, v_acc, v_f1, _, _ = evaluate(val_loader)
    for k, v in zip(history, [tr_loss, v_loss, v_acc, v_f1]):
        history[k].append(v)
    flag = ""
    if v_f1 > best_f1:
        best_f1, no_improve = v_f1, 0
        torch.save(model.state_dict(), MODEL_DIR / "best_model.pt"); flag = " ⭐"
    else:
        no_improve += 1
    print(f"Epoch {ep}/{EPOCHS} | train {tr_loss:.4f} | val {v_loss:.4f} | "
          f"acc {v_acc:.4f} | macroF1 {v_f1:.4f}{flag}")
    if no_improve >= PATIENCE:
        print(f"⏹️ Early stop (không cải thiện {PATIENCE} epoch)"); break

print("=" * 70 + f"\n🏆 Best val macro-F1: {best_f1:.4f}")

# Nếu dùng LoRA: merge adapter vào base → state_dict "phẳng" để inference.py
# (AutoModel thường) load được mà không cần peft.
if USE_LORA:
    model.load_state_dict(torch.load(MODEL_DIR / "best_model.pt"))
    model.encoder = model.encoder.merge_and_unload()
    torch.save(model.state_dict(), MODEL_DIR / "best_model.pt")
    print("🔁 Đã merge LoRA → lưu lại best_model.pt dạng phẳng (inference dùng AutoModel thường).")
''')

# ── EVAL ───────────────────────────────────────────────────────────
md(r'''
## 10. Đánh giá trên test — report, confusion matrix, accuracy theo nguồn/ngôn ngữ
''')

code(r'''
from sklearn.metrics import (classification_report, confusion_matrix,
                             precision_score, recall_score)

model.load_state_dict(torch.load(MODEL_DIR / "best_model.pt"))
_, t_acc, t_f1, preds, gts = evaluate(test_loader)
t_prec = precision_score(gts, preds, average="macro", zero_division=0)
t_rec  = recall_score(gts, preds, average="macro", zero_division=0)

print("=" * 50, f"\n  TEST: acc {t_acc:.4f} | P {t_prec:.4f} | R {t_rec:.4f} | macroF1 {t_f1:.4f}\n", "=" * 50)
report = classification_report(gts, preds, target_names=labels, zero_division=0)
print(report)

# Accuracy theo nguồn & ngôn ngữ (test_df cùng thứ tự với loader vì shuffle=False)
test_df = test_df.copy(); test_df["pred"] = preds
test_df["correct"] = (test_df["pred"] == test_df["label"]).astype(int)
print("\n— Accuracy theo nguồn —\n", test_df.groupby("source")["correct"].mean().round(3).to_string())
print("\n— Accuracy theo ngôn ngữ —\n", test_df.groupby("lang")["correct"].mean().round(3).to_string())

# Lưu kết quả
(RESULTS_DIR / "classification_report.txt").write_text(
    f"acc {t_acc:.4f}\nP {t_prec:.4f}\nR {t_rec:.4f}\nmacroF1 {t_f1:.4f}\n\n{report}", encoding="utf-8")
pd.DataFrame(confusion_matrix(gts, preds), index=labels, columns=labels)\
  .to_csv(RESULTS_DIR / "confusion_matrix.csv")
pd.DataFrame(history).to_csv(RESULTS_DIR / "training_history.csv", index=False)
json.dump({"label2id": label2id, "id2label": id2label, "level": LABEL_LEVEL},
          open(MODEL_DIR / "label_map.json", "w"), ensure_ascii=False, indent=2)
tokenizer.save_pretrained(MODEL_DIR / "tokenizer")

# Vẽ history + confusion matrix
fig, ax = plt.subplots(1, 2, figsize=(15, 5))
ax[0].plot(history["train_loss"], label="train"); ax[0].plot(history["val_loss"], label="val")
ax[0].plot(history["val_f1"], label="val macroF1"); ax[0].legend(); ax[0].set_title("Training history")
cm = confusion_matrix(gts, preds, normalize="true")
im = ax[1].imshow(cm, cmap="Blues"); ax[1].set_title("Confusion matrix (norm)")
fig.colorbar(im, ax=ax[1]); plt.tight_layout()
plt.savefig(RESULTS_DIR / "training_curves.png", dpi=110, bbox_inches="tight"); plt.show()
print("✅ Lưu kết quả →", RESULTS_DIR)
''')

# ── XAI ────────────────────────────────────────────────────────────
md(r'''
## 11. XAI — token attribution (gradient × embedding)

Giải thích **token nào đẩy mô hình về nhóm thuốc dự đoán** — phục vụ "XAI demo" của dự án.
Không cần thư viện ngoài; chạy được cho cả tiếng Việt lẫn tiếng Anh nhờ XLM-R.
''')

code(r'''
def explain(text, top_k=3, n_tokens=12):
    model.eval()
    enc = tokenizer(text, truncation=True, max_length=MAX_LEN, return_tensors="pt").to(DEVICE)
    emb_layer = model.encoder.get_input_embeddings()
    emb = emb_layer(enc["input_ids"]); emb.requires_grad_(True); emb.retain_grad()
    logits = model(attention_mask=enc["attention_mask"], inputs_embeds=emb)
    probs = torch.softmax(logits, -1)[0]
    top_p, top_i = probs.topk(min(top_k, NUM_CLASSES))
    model.zero_grad()
    logits[0, top_i[0]].backward()                       # giải thích cho lớp top-1
    attr = (emb.grad * emb).sum(-1).abs()[0]             # |grad·input| theo token
    attr = (attr / (attr.max() + 1e-9)).cpu().tolist()
    toks = tokenizer.convert_ids_to_tokens(enc["input_ids"][0].cpu().tolist())

    print(f"\n📝 \"{text[:90]}\"")
    for p, i in zip(top_p.tolist(), top_i.tolist()):
        print(f"   → {id2label[i]:<32} {p:.1%}")
    pairs = [(t, a) for t, a in zip(toks, attr) if t not in tokenizer.all_special_tokens]
    print("   🔑 Token ảnh hưởng nhất:",
          ", ".join(f"{t}({a:.2f})" for t, a in sorted(pairs, key=lambda x: -x[1])[:n_tokens]))

for s in [
    "Patient has high fever, productive cough with yellow sputum and chest pain",
    "Bệnh nhân tăng huyết áp 160/95, chóng mặt, nhức đầu",
    "Đau dạ dày, ợ chua, nóng rát thượng vị sau khi ăn",
    "Joint pain and swelling in both knees with morning stiffness",
]:
    explain(s)
''')

# ── INFERENCE / PACKAGE ────────────────────────────────────────────
md(r'''
## 12. Hàm inference + đóng gói model để tải về

Sau khi chạy xong: tải `drugpred-model.zip` ở tab **Output** → giải nén vào
`backend/ml/models/weights/` và dùng đoạn `inference.py` in ra dưới đây.
''')

code(r'''
@torch.no_grad()
def predict_drug_groups(text, top_k=3):
    model.eval()
    enc = tokenizer(text, truncation=True, padding="max_length", max_length=MAX_LEN,
                    return_tensors="pt").to(DEVICE)
    probs = torch.softmax(model(enc["input_ids"], enc["attention_mask"]), -1)[0]
    p, idx = probs.topk(min(top_k, NUM_CLASSES))
    return [{"drug_group": id2label[int(i)], "confidence": float(c), "rank": r + 1}
            for r, (c, i) in enumerate(zip(p.cpu(), idx.cpu()))]

print("🧪 Demo:")
for r in predict_drug_groups("Bệnh nhân sốt cao, ho có đờm vàng, đau ngực"):
    print("  ", r)

import shutil
shutil.make_archive(str(WORKING / "drugpred-model"), "zip", root_dir=WORKING, base_dir="model")
print("\n📦 Đã đóng gói:", WORKING / "drugpred-model.zip")
print("   Tải ở tab Output → giải nén vào backend/ml/models/weights/")
''')

code(r'''
# ── Sinh sẵn backend/ml/inference.py (in ra + lưu inference_snippet.py để tải về) ──
INFERENCE_SNIPPET = r"""
import json, torch, torch.nn as nn
from pathlib import Path
from transformers import AutoTokenizer, AutoModel

WEIGHTS = Path(__file__).parent / "models" / "weights"
MODEL_NAME, MAX_LEN = "xlm-roberta-base", 256
_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_state = json.load(open(WEIGHTS / "label_map.json", encoding="utf-8"))
_id2label = {int(k): v for k, v in _state["id2label"].items()}

class DrugClassifier(nn.Module):
    def __init__(self, name, n):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(name)
        h = self.encoder.config.hidden_size
        self.head = nn.Sequential(nn.Dropout(0.3), nn.Linear(h, 256), nn.GELU(),
                                  nn.LayerNorm(256), nn.Dropout(0.15), nn.Linear(256, n))
    def forward(self, ids, mask):
        return self.head(self.encoder(input_ids=ids, attention_mask=mask).last_hidden_state[:, 0])

_tok = AutoTokenizer.from_pretrained(str(WEIGHTS / "tokenizer"))
_model = DrugClassifier(MODEL_NAME, len(_id2label)).to(_DEVICE)
_model.load_state_dict(torch.load(WEIGHTS / "best_model.pt", map_location=_DEVICE))
_model.eval()

@torch.no_grad()
def predict_drug_groups(text, top_k=3):
    enc = _tok(text, truncation=True, padding="max_length", max_length=MAX_LEN,
               return_tensors="pt").to(_DEVICE)
    probs = torch.softmax(_model(enc["input_ids"], enc["attention_mask"]), -1)[0]
    p, idx = probs.topk(min(top_k, len(_id2label)))
    return [{"drug_group_name": _id2label[int(i)], "confidence": float(c), "rank": r + 1}
            for r, (c, i) in enumerate(zip(p.cpu(), idx.cpu()))]
"""

(WORKING / "inference_snippet.py").write_text(INFERENCE_SNIPPET, encoding="utf-8")
print(INFERENCE_SNIPPET)
print("📄 Đã lưu:", WORKING / "inference_snippet.py")
''')

# ────────────────────────────────────────────────────────────────────
nb["cells"] = cells
nb["metadata"] = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python"},
    "accelerator": "GPU",
}

out = Path(__file__).parent / "drugpred_kaggle_train.ipynb"
nbf.validate(nb)
nbf.write(nb, str(out))
print("✅ wrote", out, "—", len(cells), "cells")
