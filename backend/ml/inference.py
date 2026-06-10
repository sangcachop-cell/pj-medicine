"""
ML Inference Module — Drug Group Prediction

Đây là interface chính để Backend gọi ML model.
ML Engineer sẽ implement các hàm bên dưới.

Usage:
    from ml.inference import predict_drug_groups
    results = predict_drug_groups("Bệnh nhân sốt cao 3 ngày, ho có đờm...")
"""

from dataclasses import dataclass


@dataclass
class PredictionResult:
    """Kết quả dự đoán cho 1 nhóm thuốc."""
    drug_group_id: str
    drug_group_name: str
    confidence: float
    rank: int


# TODO: ML Engineer implement
# - Load model từ file weights
# - Preprocessing text (tokenize, clean)
# - Run inference
# - Return top-K results

_model = None  # Global model instance


def load_model(model_path: str) -> None:
    """
    Load model weights vào memory.
    Được gọi 1 lần khi server khởi động (trong lifespan).
    """
    global _model
    # TODO: Implement
    # _model = torch.load(model_path)
    # _model.eval()
    print(f"[ML] Model loaded from {model_path}")


def predict_drug_groups(
    text: str,
    top_k: int = 3,
) -> list[PredictionResult]:
    """
    Dự đoán nhóm thuốc từ text mô tả bệnh án.

    Args:
        text: Mô tả bệnh án tiếng Việt
        top_k: Số nhóm thuốc trả về (default: 3)

    Returns:
        List[PredictionResult] sorted by confidence (descending)
    """
    # TODO: ML Engineer implement real inference

    # --- MOCK: Trả về kết quả giả để Backend + Frontend test ---
    mock_results = [
        PredictionResult(
            drug_group_id="mock-uuid-1",
            drug_group_name="Kháng sinh - Beta-lactam",
            confidence=0.87,
            rank=1,
        ),
        PredictionResult(
            drug_group_id="mock-uuid-2",
            drug_group_name="Thuốc giảm đau - NSAID",
            confidence=0.52,
            rank=2,
        ),
        PredictionResult(
            drug_group_id="mock-uuid-3",
            drug_group_name="Thuốc long đờm",
            confidence=0.23,
            rank=3,
        ),
    ]
    return mock_results[:top_k]
