"""
Drug-Pred AI — Data Pipeline
Script tải, xử lý, và chuẩn bị dataset cho training.

Datasets (6 nguồn):
  Kaggle:
    1. Medicine Recommendation System  — symptoms → drug mapping
    2. UCI Drug Review                  — 215K text reviews → condition → drug
    3. OpenFDA Drug Labeling            — FDA labels (indications, interactions)
    4. 11000 Medicine Details           — uses, composition, side effects
  HuggingFace:
    5. ViMedAQA                         — Vietnamese Medical QA
    6. HoangHa/medical-data             — Vietnamese clinical dialogues

Usage:
    python data_pipeline.py download    # Tải 6 datasets
    python data_pipeline.py process     # Xử lý & merge
    python data_pipeline.py translate   # Dịch sang tiếng Việt
    python data_pipeline.py split       # Chia train/val/test
    python data_pipeline.py eda         # Thống kê
    python data_pipeline.py all         # Chạy tất cả

Yêu cầu:
    pip install -r requirements-ml.txt
"""

import sys
import os
from pathlib import Path

# Thư mục gốc
DATA_DIR = Path(__file__).parent / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
FINAL_DIR = DATA_DIR / "final"


def setup_dirs():
    """Tạo thư mục cần thiết."""
    for d in [RAW_DIR, PROCESSED_DIR, FINAL_DIR]:
        d.mkdir(parents=True, exist_ok=True)
    print("✅ Thư mục đã sẵn sàng")


# ============================================================
# BƯỚC 1: TẢI DATASETS
# ============================================================
def _download_kaggle(slug: str, dest_name: str, index: int, total: int):
    """Helper: tải 1 dataset từ Kaggle."""
    print(f"\n[{index}/{total}] {slug}...")
    try:
        import kagglehub
        import shutil
        path = kagglehub.dataset_download(slug)
        print(f"  ✅ Đã tải về: {path}")
        dest = RAW_DIR / dest_name
        if dest.exists():
            shutil.rmtree(dest)
        shutil.copytree(path, dest)
    except Exception as e:
        print(f"  ⚠️ Lỗi tải: {e}")
        print(f"  → Tải thủ công tại: kaggle.com/datasets/{slug}")


def _download_huggingface(repo: str, dest_name: str, index: int, total: int,
                          config: str | None = None):
    """Helper: tải 1 dataset từ HuggingFace."""
    print(f"\n[{index}/{total}] {repo}" + (f" (config={config})" if config else "") + "...")
    try:
        from datasets import load_dataset
        ds = load_dataset(repo, config, trust_remote_code=True)
        dest = RAW_DIR / dest_name
        dest.mkdir(exist_ok=True)
        for split_name, split_data in ds.items():
            split_data.to_csv(dest / f"{split_name}.csv", index=False)
        print(f"  ✅ Đã tải về: {dest}")
    except Exception as e:
        print(f"  ⚠️ Lỗi tải: {e}")
        print(f"  → Tải thủ công tại: huggingface.co/datasets/{repo}")


def download_datasets():
    """Tải tất cả datasets từ Kaggle và HuggingFace."""
    setup_dirs()

    TOTAL = 6
    print(f"\n📥 Đang tải {TOTAL} datasets...")
    print("=" * 60)

    # 1. Medicine Recommendation System (Kaggle)
    _download_kaggle(
        "saisumanthv/medicine-recommendation-system-dataset",
        "medicine_recommendation", 1, TOTAL,
    )

    # 2. UCI Drug Review (Kaggle)
    _download_kaggle(
        "jessicali9530/kuc-hackathon-winter-2018",
        "drug_review", 2, TOTAL,
    )

    # 3. OpenFDA Drug Labeling (Kaggle) — FDA labels với indications
    _download_kaggle(
        "ddrbcn/openfda-drug-labeling",
        "openfda_labeling", 3, TOTAL,
    )

    # 4. 11000 Medicine Details (Kaggle) — uses, composition, side effects
    _download_kaggle(
        "singhnavjot2062001/11000-medicine-details",
        "medicine_details", 4, TOTAL,
    )

    # 5. ViMedAQA (HuggingFace) — Vietnamese Medical QA
    _download_huggingface("tmnam20/ViMedAQA", "vimedaqa", 5, TOTAL)

    # 6. HoangHa/medical-data (HuggingFace) — Vietnamese clinical dialogues
    #    Có 4 configs: vietnamese, english, RandomQA, RandomQuestion
    #    Chỉ cần "vietnamese" cho dự án
    _download_huggingface(
        "HoangHa/medical-data", "meddies_vi", 6, TOTAL,
        config="vietnamese",
    )

    print("\n" + "=" * 60)
    print("📦 Tải xong! Kiểm tra thư mục:", RAW_DIR)


# ============================================================
# DRUG → DRUG GROUP MAPPING
# ============================================================
DRUG_TO_GROUP = {
    # === KHÁNG SINH ===
    "Amoxicillin": "Kháng sinh - Penicillin",
    "Ampicillin": "Kháng sinh - Penicillin",
    "Penicillin": "Kháng sinh - Penicillin",
    "Augmentin": "Kháng sinh - Penicillin",
    "Azithromycin": "Kháng sinh - Macrolide",
    "Erythromycin": "Kháng sinh - Macrolide",
    "Clarithromycin": "Kháng sinh - Macrolide",
    "Cephalexin": "Kháng sinh - Cephalosporin",
    "Ceftriaxone": "Kháng sinh - Cephalosporin",
    "Cefuroxime": "Kháng sinh - Cephalosporin",
    "Ciprofloxacin": "Kháng sinh - Fluoroquinolone",
    "Levofloxacin": "Kháng sinh - Fluoroquinolone",
    "Moxifloxacin": "Kháng sinh - Fluoroquinolone",
    "Doxycycline": "Kháng sinh - Tetracycline",
    "Tetracycline": "Kháng sinh - Tetracycline",
    "Metronidazole": "Kháng sinh - Nitroimidazole",
    "Trimethoprim": "Kháng sinh - Sulfonamide",

    # === GIẢM ĐAU ===
    "Ibuprofen": "Giảm đau - NSAID",
    "Naproxen": "Giảm đau - NSAID",
    "Diclofenac": "Giảm đau - NSAID",
    "Celecoxib": "Giảm đau - NSAID",
    "Meloxicam": "Giảm đau - NSAID",
    "Aspirin": "Giảm đau - NSAID",
    "Acetaminophen": "Giảm đau - Paracetamol",
    "Paracetamol": "Giảm đau - Paracetamol",
    "Tramadol": "Giảm đau - Opioid nhẹ",
    "Codeine": "Giảm đau - Opioid nhẹ",

    # === TIM MẠCH ===
    "Lisinopril": "Tim mạch - ACE inhibitor",
    "Enalapril": "Tim mạch - ACE inhibitor",
    "Ramipril": "Tim mạch - ACE inhibitor",
    "Captopril": "Tim mạch - ACE inhibitor",
    "Losartan": "Tim mạch - ARB",
    "Valsartan": "Tim mạch - ARB",
    "Irbesartan": "Tim mạch - ARB",
    "Amlodipine": "Tim mạch - Chẹn kênh Canxi",
    "Nifedipine": "Tim mạch - Chẹn kênh Canxi",
    "Diltiazem": "Tim mạch - Chẹn kênh Canxi",
    "Metoprolol": "Tim mạch - Beta blocker",
    "Atenolol": "Tim mạch - Beta blocker",
    "Propranolol": "Tim mạch - Beta blocker",
    "Bisoprolol": "Tim mạch - Beta blocker",
    "Hydrochlorothiazide": "Tim mạch - Lợi tiểu",
    "Furosemide": "Tim mạch - Lợi tiểu",
    "Spironolactone": "Tim mạch - Lợi tiểu",

    # === TIÊU HÓA ===
    "Omeprazole": "Tiêu hóa - PPI",
    "Esomeprazole": "Tiêu hóa - PPI",
    "Pantoprazole": "Tiêu hóa - PPI",
    "Lansoprazole": "Tiêu hóa - PPI",
    "Ranitidine": "Tiêu hóa - H2 blocker",
    "Famotidine": "Tiêu hóa - H2 blocker",
    "Loperamide": "Tiêu hóa - Chống tiêu chảy",
    "Domperidone": "Tiêu hóa - Chống nôn",
    "Ondansetron": "Tiêu hóa - Chống nôn",

    # === NỘI TIẾT ===
    "Metformin": "Nội tiết - Biguanide",
    "Glipizide": "Nội tiết - Sulfonylurea",
    "Gliclazide": "Nội tiết - Sulfonylurea",
    "Insulin": "Nội tiết - Insulin",
    "Levothyroxine": "Nội tiết - Hormone tuyến giáp",

    # === HÔ HẤP ===
    "Salbutamol": "Hô hấp - Giãn phế quản",
    "Albuterol": "Hô hấp - Giãn phế quản",
    "Ipratropium": "Hô hấp - Kháng cholinergic",
    "Montelukast": "Hô hấp - Kháng leukotriene",
    "Dextromethorphan": "Hô hấp - Giảm ho",
    "Guaifenesin": "Hô hấp - Long đờm",
    "Bromhexine": "Hô hấp - Long đờm",

    # === THẦN KINH ===
    "Sertraline": "Thần kinh - SSRI",
    "Fluoxetine": "Thần kinh - SSRI",
    "Escitalopram": "Thần kinh - SSRI",
    "Amitriptyline": "Thần kinh - TCA",
    "Gabapentin": "Thần kinh - Chống động kinh",
    "Pregabalin": "Thần kinh - Chống động kinh",
    "Carbamazepine": "Thần kinh - Chống động kinh",
    "Diazepam": "Thần kinh - Benzodiazepine",
    "Lorazepam": "Thần kinh - Benzodiazepine",
    "Alprazolam": "Thần kinh - Benzodiazepine",
    "Sumatriptan": "Thần kinh - Triptan (Migraine)",

    # === DỊ ỨNG ===
    "Cetirizine": "Dị ứng - Kháng histamine",
    "Loratadine": "Dị ứng - Kháng histamine",
    "Fexofenadine": "Dị ứng - Kháng histamine",
    "Diphenhydramine": "Dị ứng - Kháng histamine",
    "Chlorpheniramine": "Dị ứng - Kháng histamine",

    # === CORTICOSTEROID ===
    "Prednisolone": "Chống viêm - Corticosteroid",
    "Prednisone": "Chống viêm - Corticosteroid",
    "Dexamethasone": "Chống viêm - Corticosteroid",
    "Hydrocortisone": "Chống viêm - Corticosteroid",
    "Methylprednisolone": "Chống viêm - Corticosteroid",

    # === DA LIỄU ===
    "Clotrimazole": "Da liễu - Kháng nấm",
    "Fluconazole": "Da liễu - Kháng nấm",
    "Ketoconazole": "Da liễu - Kháng nấm",
    "Acyclovir": "Da liễu - Kháng virus",

    # === CƠ XƯƠNG KHỚP ===
    "Allopurinol": "Cơ xương khớp - Chống gout",
    "Colchicine": "Cơ xương khớp - Chống gout",
    "Methotrexate": "Cơ xương khớp - DMARD",

    # === KHÁC ===
    "Warfarin": "Huyết học - Chống đông",
    "Heparin": "Huyết học - Chống đông",
    "Atorvastatin": "Chuyển hóa - Statin",
    "Simvastatin": "Chuyển hóa - Statin",
    "Rosuvastatin": "Chuyển hóa - Statin",
}

# Danh sách drug groups
DRUG_GROUPS = sorted(set(DRUG_TO_GROUP.values()))


def get_drug_group(drug_name: str) -> str | None:
    """Map tên thuốc → nhóm thuốc. Fuzzy match."""
    if not drug_name:
        return None

    name = drug_name.strip()

    # Exact match
    if name in DRUG_TO_GROUP:
        return DRUG_TO_GROUP[name]

    # Case-insensitive match
    name_lower = name.lower()
    for drug, group in DRUG_TO_GROUP.items():
        if drug.lower() == name_lower:
            return group

    # Partial match (drug name contains known drug)
    for drug, group in DRUG_TO_GROUP.items():
        if drug.lower() in name_lower or name_lower in drug.lower():
            return group

    return None


# ============================================================
# BƯỚC 2: XỬ LÝ & MERGE DATASETS
# ============================================================
def process_datasets():
    """Xử lý và merge các datasets thành 1 file."""
    import pandas as pd

    setup_dirs()
    all_data = []

    # --- Process Medicine Recommendation Dataset ---
    print("\n🔧 Xử lý Medicine Recommendation Dataset...")
    med_dir = RAW_DIR / "medicine_recommendation"
    csv_files = list(med_dir.rglob("*.csv"))

    if csv_files:
        for f in csv_files:
            print(f"  Đang đọc: {f.name}")
            try:
                df = pd.read_csv(f, on_bad_lines="skip")
                print(f"  → Columns: {list(df.columns)}")
                print(f"  → Rows: {len(df)}")

                # Tìm cột symptoms và drug
                symptom_cols = [c for c in df.columns if "symptom" in c.lower()]
                drug_cols = [c for c in df.columns if any(x in c.lower() for x in ["drug", "medicine", "medication"])]
                disease_cols = [c for c in df.columns if "disease" in c.lower()]

                if symptom_cols and (drug_cols or disease_cols):
                    for _, row in df.iterrows():
                        # Ghép symptoms thành text
                        symptoms = []
                        for col in symptom_cols:
                            val = str(row.get(col, "")).strip()
                            if val and val.lower() not in ["nan", "none", ""]:
                                symptoms.append(val.replace("_", " "))

                        if not symptoms:
                            continue

                        text = ", ".join(symptoms)

                        # Lấy drug group
                        drug_group = None
                        for col in drug_cols:
                            drug_name = str(row.get(col, "")).strip()
                            drug_group = get_drug_group(drug_name)
                            if drug_group:
                                break

                        if drug_group:
                            all_data.append({
                                "text_en": text,
                                "drug_group": drug_group,
                                "source": "medicine_recommendation",
                            })
            except Exception as e:
                print(f"  ⚠️ Lỗi đọc {f.name}: {e}")
    else:
        print("  ⚠️ Không tìm thấy files. Tải dataset trước (python data_pipeline.py download)")

    # --- Process UCI Drug Review Dataset ---
    print("\n🔧 Xử lý UCI Drug Review Dataset...")
    review_dir = RAW_DIR / "drug_review"
    csv_files = list(review_dir.rglob("*.tsv")) + list(review_dir.rglob("*.csv"))

    if csv_files:
        for f in csv_files:
            print(f"  Đang đọc: {f.name}")
            try:
                sep = "\t" if f.suffix == ".tsv" else ","
                df = pd.read_csv(f, sep=sep, on_bad_lines="skip")
                print(f"  → Columns: {list(df.columns)}")
                print(f"  → Rows: {len(df)}")

                if "drugName" in df.columns and "review" in df.columns:
                    # Lọc reviews tốt (rating >= 7)
                    if "rating" in df.columns:
                        df = df[df["rating"] >= 7]
                        print(f"  → Sau lọc rating >= 7: {len(df)} rows")

                    for _, row in df.iterrows():
                        drug_name = str(row.get("drugName", "")).strip()
                        review = str(row.get("review", "")).strip()
                        condition = str(row.get("condition", "")).strip()

                        drug_group = get_drug_group(drug_name)
                        if drug_group and len(review) > 20:
                            # Truncate review nếu quá dài
                            text = review[:500] if len(review) > 500 else review
                            all_data.append({
                                "text_en": text,
                                "drug_group": drug_group,
                                "source": "uci_drug_review",
                                "condition": condition,
                            })
            except Exception as e:
                print(f"  ⚠️ Lỗi đọc {f.name}: {e}")
    else:
        print("  ⚠️ Không tìm thấy files.")

    # --- Process OpenFDA Drug Labeling ---
    print("\n🔧 Xử lý OpenFDA Drug Labeling...")
    fda_dir = RAW_DIR / "openfda_labeling"
    json_files = list(fda_dir.rglob("*.json"))
    zip_files = list(fda_dir.rglob("*.zip"))

    if zip_files and not json_files:
        print("  📦 Giải nén ZIP files...")
        import zipfile
        for zf in zip_files:
            try:
                with zipfile.ZipFile(zf, "r") as z:
                    z.extractall(fda_dir / "extracted")
            except Exception as e:
                print(f"  ⚠️ Lỗi giải nén {zf.name}: {e}")
        json_files = list(fda_dir.rglob("*.json"))

    if json_files:
        import json as json_module
        fda_count = 0
        for f in json_files[:5]:  # Giới hạn 5 files (dữ liệu rất lớn)
            print(f"  Đang đọc: {f.name}")
            try:
                with open(f, "r", encoding="utf-8") as fp:
                    data = json_module.load(fp)

                results = data.get("results", [data] if isinstance(data, dict) else data)
                if not isinstance(results, list):
                    results = [results]

                for record in results:
                    # Lấy indications_and_usage làm text input
                    indications = record.get("indications_and_usage", [])
                    if isinstance(indications, list):
                        indications = " ".join(indications)
                    indications = str(indications).strip()

                    # Lấy tên thuốc
                    openfda = record.get("openfda", {})
                    drug_names = openfda.get("generic_name", []) or openfda.get("brand_name", [])
                    if isinstance(drug_names, list) and drug_names:
                        drug_name = drug_names[0]
                    elif isinstance(drug_names, str):
                        drug_name = drug_names
                    else:
                        continue

                    drug_group = get_drug_group(drug_name)
                    if drug_group and len(indications) > 30:
                        text = indications[:500]
                        all_data.append({
                            "text_en": text,
                            "drug_group": drug_group,
                            "source": "openfda_labeling",
                        })
                        fda_count += 1
            except Exception as e:
                print(f"  ⚠️ Lỗi đọc {f.name}: {e}")
        print(f"  → Trích xuất được {fda_count} samples từ OpenFDA")
    else:
        print("  ⚠️ Không tìm thấy JSON files. Dataset ~1.7GB, có thể cần tải thủ công.")

    # --- Process 11000 Medicine Details ---
    print("\n🔧 Xử lý 11000 Medicine Details...")
    med_details_dir = RAW_DIR / "medicine_details"
    csv_files = list(med_details_dir.rglob("*.csv"))

    if csv_files:
        for f in csv_files:
            print(f"  Đang đọc: {f.name}")
            try:
                df = pd.read_csv(f, on_bad_lines="skip")
                print(f"  → Columns: {list(df.columns)}")
                print(f"  → Rows: {len(df)}")

                # Tìm cột "uses" và "Medicine Name" hoặc tương tự
                name_col = None
                uses_col = None
                for c in df.columns:
                    cl = c.lower().strip()
                    if cl in ["medicine name", "medicine_name", "name", "drug_name"]:
                        name_col = c
                    if cl in ["uses", "use", "indication", "description"]:
                        uses_col = c

                if name_col and uses_col:
                    med_count = 0
                    for _, row in df.iterrows():
                        drug_name = str(row.get(name_col, "")).strip()
                        uses_text = str(row.get(uses_col, "")).strip()

                        # Thử match tên thuốc
                        drug_group = get_drug_group(drug_name)

                        # Nếu không match tên, thử tìm tên thuốc trong composition
                        if not drug_group:
                            comp_col = None
                            for c in df.columns:
                                if "compos" in c.lower() or "salt" in c.lower():
                                    comp_col = c
                                    break
                            if comp_col:
                                composition = str(row.get(comp_col, ""))
                                for known_drug in DRUG_TO_GROUP:
                                    if known_drug.lower() in composition.lower():
                                        drug_group = DRUG_TO_GROUP[known_drug]
                                        break

                        if drug_group and len(uses_text) > 15:
                            all_data.append({
                                "text_en": uses_text[:500],
                                "drug_group": drug_group,
                                "source": "11k_medicine_details",
                            })
                            med_count += 1
                    print(f"  → Trích xuất được {med_count} samples")
                else:
                    print(f"  ⚠️ Không tìm thấy cột phù hợp. Columns: {list(df.columns)}")
            except Exception as e:
                print(f"  ⚠️ Lỗi đọc {f.name}: {e}")
    else:
        print("  ⚠️ Không tìm thấy files.")

    # --- Process HoangHa/medical-data (Vietnamese) ---
    print("\n🔧 Xử lý HoangHa/medical-data (Vietnamese dialogues)...")
    meddies_dir = RAW_DIR / "meddies_vi"
    csv_files = list(meddies_dir.rglob("*.csv"))

    vi_data = []  # Lưu riêng vì đã là tiếng Việt
    if csv_files:
        for f in csv_files:
            print(f"  Đang đọc: {f.name}")
            try:
                df = pd.read_csv(f, on_bad_lines="skip")
                print(f"  → Columns: {list(df.columns)}")
                print(f"  → Rows: {len(df)}")

                # Dataset có: messages, target_disease, patient_persona
                if "messages" in df.columns and "target_disease" in df.columns:
                    meddies_count = 0
                    for _, row in df.iterrows():
                        messages = str(row.get("messages", ""))
                        disease = str(row.get("target_disease", "")).strip()

                        # Extract patient message (triệu chứng)
                        # messages là JSON list: [{role, content}, ...]
                        try:
                            import json as json_module
                            msg_list = json_module.loads(messages.replace("'", '"'))
                            # Lấy message đầu tiên của patient
                            patient_text = ""
                            for msg in msg_list:
                                if isinstance(msg, dict):
                                    role = msg.get("role", "").lower()
                                    if role in ["user", "patient", "human"]:
                                        patient_text = msg.get("content", "")
                                        break
                            if not patient_text and msg_list:
                                # Fallback: lấy message đầu tiên
                                patient_text = msg_list[0].get("content", "") if isinstance(msg_list[0], dict) else str(msg_list[0])
                        except Exception:
                            patient_text = messages[:500]

                        if len(patient_text) > 20 and disease:
                            vi_data.append({
                                "text_vi": patient_text[:500],
                                "target_disease": disease,
                                "source": "meddies_hoangha",
                            })
                            meddies_count += 1
                    print(f"  → Trích xuất được {meddies_count} Vietnamese samples")
            except Exception as e:
                print(f"  ⚠️ Lỗi đọc {f.name}: {e}")
    else:
        print("  ⚠️ Không tìm thấy files.")

    # Lưu Vietnamese data riêng (không cần dịch)
    if vi_data:
        df_vi = pd.DataFrame(vi_data)
        vi_output = PROCESSED_DIR / "meddies_vi.csv"
        df_vi.to_csv(vi_output, index=False)
        print(f"\n✅ Vietnamese data (HoangHa): {vi_output} ({len(df_vi)} samples)")
        print(f"   Lưu ý: Cần thêm bước map target_disease → drug_group thủ công")

    # --- Tổng hợp English data ---
    if all_data:
        df_all = pd.DataFrame(all_data)

        # Cân bằng classes (lấy max N samples per class)
        MAX_PER_CLASS = 1000
        df_balanced = df_all.groupby("drug_group").apply(
            lambda x: x.sample(min(len(x), MAX_PER_CLASS), random_state=42)
        ).reset_index(drop=True)

        output_path = PROCESSED_DIR / "merged_en.csv"
        df_balanced.to_csv(output_path, index=False)

        print("\n" + "=" * 60)
        print(f"✅ English dataset merged: {output_path}")
        print(f"   Tổng samples: {len(df_balanced)}")
        print(f"   Số drug groups: {df_balanced['drug_group'].nunique()}")
        print(f"\n📊 Phân bố theo nguồn:")
        print(df_balanced["source"].value_counts().to_string())
        print(f"\n📊 Phân bố classes:")
        print(df_balanced["drug_group"].value_counts().to_string())
    else:
        print("\n❌ Không có data. Hãy chạy `python data_pipeline.py download` trước.")


# ============================================================
# BƯỚC 3: DỊCH SANG TIẾNG VIỆT
# ============================================================
def translate_dataset():
    """Dịch dataset từ English sang Vietnamese."""
    import pandas as pd

    input_path = PROCESSED_DIR / "merged_en.csv"
    if not input_path.exists():
        print(f"❌ Chưa có file {input_path}. Chạy `python data_pipeline.py process` trước.")
        return

    df = pd.read_csv(input_path)
    print(f"📝 Đang dịch {len(df)} samples sang tiếng Việt...")
    print("   (Có thể mất 10-30 phút tùy kích thước dataset)")

    try:
        from googletrans import Translator
        translator = Translator()

        translated_texts = []
        errors = 0
        batch_size = 50

        for i, row in df.iterrows():
            try:
                result = translator.translate(row["text_en"], src="en", dest="vi")
                translated_texts.append(result.text)
            except Exception:
                # Fallback: giữ nguyên text tiếng Anh
                translated_texts.append(row["text_en"])
                errors += 1

            if (i + 1) % batch_size == 0:
                print(f"   Đã dịch {i + 1}/{len(df)} ({errors} lỗi)")

        df["text_vi"] = translated_texts

        # Lưu
        output_path = FINAL_DIR / "dataset_vi.csv"
        df.to_csv(output_path, index=False)

        print(f"\n✅ Đã dịch xong!")
        print(f"   Output: {output_path}")
        print(f"   Samples: {len(df)}")
        print(f"   Lỗi dịch: {errors}")

    except ImportError:
        print("⚠️ Cần install: pip install googletrans==4.0.0-rc.1")
        print("   Hoặc dùng MarianMT (xem hướng dẫn trong dataset_guide.md)")


# ============================================================
# BƯỚC 4: TẠO TRAIN/VAL/TEST SPLIT
# ============================================================
def create_splits():
    """Chia dataset thành train/val/test."""
    import pandas as pd
    from sklearn.model_selection import train_test_split

    input_path = FINAL_DIR / "dataset_vi.csv"
    if not input_path.exists():
        # Fallback: dùng English version
        input_path = PROCESSED_DIR / "merged_en.csv"
        if not input_path.exists():
            print("❌ Chưa có dataset. Chạy các bước trước.")
            return

    df = pd.read_csv(input_path)
    text_col = "text_vi" if "text_vi" in df.columns else "text_en"

    # Lọc bỏ rows không hợp lệ
    df = df.dropna(subset=[text_col, "drug_group"])
    df = df[df[text_col].str.len() > 10]

    # Lọc classes quá nhỏ (< 10 samples)
    class_counts = df["drug_group"].value_counts()
    valid_classes = class_counts[class_counts >= 10].index
    df = df[df["drug_group"].isin(valid_classes)]

    print(f"📊 Dataset sau khi lọc: {len(df)} samples, {df['drug_group'].nunique()} classes")

    # Split: 70% train, 15% val, 15% test (stratified)
    train_df, temp_df = train_test_split(
        df, test_size=0.3, random_state=42, stratify=df["drug_group"]
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=0.5, random_state=42, stratify=temp_df["drug_group"]
    )

    # Lưu
    train_df.to_csv(FINAL_DIR / "train.csv", index=False)
    val_df.to_csv(FINAL_DIR / "val.csv", index=False)
    test_df.to_csv(FINAL_DIR / "test.csv", index=False)

    # Label mapping
    labels = sorted(df["drug_group"].unique())
    label_map = {label: idx for idx, label in enumerate(labels)}
    import json
    with open(FINAL_DIR / "label_map.json", "w", encoding="utf-8") as f:
        json.dump(label_map, f, ensure_ascii=False, indent=2)

    print(f"\n✅ Đã chia dataset!")
    print(f"   Train: {len(train_df)} samples")
    print(f"   Val:   {len(val_df)} samples")
    print(f"   Test:  {len(test_df)} samples")
    print(f"   Labels: {FINAL_DIR / 'label_map.json'}")
    print(f"\n📁 Files:")
    print(f"   {FINAL_DIR / 'train.csv'}")
    print(f"   {FINAL_DIR / 'val.csv'}")
    print(f"   {FINAL_DIR / 'test.csv'}")
    print(f"   {FINAL_DIR / 'label_map.json'}")


# ============================================================
# BƯỚC 5: EDA — THỐNG KÊ NHANH
# ============================================================
def eda():
    """Exploratory Data Analysis."""
    import pandas as pd

    # Tìm dataset file
    for path in [FINAL_DIR / "train.csv", PROCESSED_DIR / "merged_en.csv"]:
        if path.exists():
            df = pd.read_csv(path)
            text_col = "text_vi" if "text_vi" in df.columns else "text_en"

            print("=" * 60)
            print(f"📊 EDA — {path.name}")
            print("=" * 60)
            print(f"\nTổng samples: {len(df)}")
            print(f"Số drug groups: {df['drug_group'].nunique()}")

            print(f"\n--- Phân bố classes ---")
            print(df["drug_group"].value_counts().to_string())

            print(f"\n--- Độ dài text ---")
            df["text_len"] = df[text_col].astype(str).str.len()
            print(f"  Min: {df['text_len'].min()}")
            print(f"  Max: {df['text_len'].max()}")
            print(f"  Mean: {df['text_len'].mean():.0f}")
            print(f"  Median: {df['text_len'].median():.0f}")

            if "source" in df.columns:
                print(f"\n--- Nguồn data ---")
                print(df["source"].value_counts().to_string())

            return

    print("❌ Chưa có dataset. Chạy download → process trước.")


# ============================================================
# MAIN
# ============================================================
def main():
    if len(sys.argv) < 2:
        print("""
🏥 Drug-Pred AI — Data Pipeline

Usage:
    python data_pipeline.py <command>

Commands:
    download    Tải datasets từ Kaggle + HuggingFace
    process     Xử lý, merge, và tạo drug group labels
    translate   Dịch sang tiếng Việt (Google Translate)
    split       Chia train/val/test (70/15/15)
    eda         Thống kê nhanh về dataset
    all         Chạy tất cả các bước

Ví dụ:
    python data_pipeline.py download
    python data_pipeline.py all
        """)
        return

    command = sys.argv[1].lower()

    if command == "download":
        download_datasets()
    elif command == "process":
        process_datasets()
    elif command == "translate":
        translate_dataset()
    elif command == "split":
        create_splits()
    elif command == "eda":
        eda()
    elif command == "all":
        download_datasets()
        process_datasets()
        translate_dataset()
        create_splits()
        eda()
    else:
        print(f"❌ Lệnh không hợp lệ: {command}")
        print("   Dùng: download | process | translate | split | eda | all")


if __name__ == "__main__":
    main()
