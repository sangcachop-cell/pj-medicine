# 🏥 Drug-Pred AI

> Hệ thống dự đoán nhóm thuốc từ mô tả bệnh án tiếng Việt — sử dụng kiến trúc Mamba SSM + Multi-LoRA

## 📋 Tổng quan

Drug-Pred AI là hệ thống hỗ trợ quyết định lâm sàng (CDSS) giúp dự đoán nhóm thuốc phù hợp dựa trên mô tả triệu chứng và bệnh án của bệnh nhân, sử dụng mô hình học sâu Mamba (State Space Model).

### Kiến trúc

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│   React +    │     │   FastAPI    │     │  Mamba SSM   │
│  TypeScript  │────▶│   Python     │────▶│  + LoRA      │
│   Frontend   │     │   Backend    │     │  ML Engine   │
└──────────────┘     └──────┬───────┘     └──────────────┘
                            │
                   ┌────────┴────────┐
                   │   PostgreSQL    │
                   │   + Redis      │
                   └─────────────────┘
```

## 🚀 Quick Start

### Yêu cầu

- [Docker](https://docs.docker.com/get-docker/) & Docker Compose
- Node.js 20+ (cho frontend dev)
- Python 3.11+ (cho backend dev)

### 1. Clone & Setup

```bash
git clone https://github.com/your-repo/pj-medicine.git
cd pj-medicine
cp .env.example .env
```

### 2. Chạy với Docker (Khuyến nghị)

```bash
# Khởi động toàn bộ stack
make up

# Hoặc không dùng make:
docker compose up -d
```

Truy cập:
- 🎨 Frontend: http://localhost:5173
- ⚙️ Backend API: http://localhost:8000
- 📚 API Docs (Swagger): http://localhost:8000/docs
- 🗄️ PostgreSQL: localhost:5432

### 3. Chạy không cần Docker (Dev riêng)

**Backend:**
```bash
cd backend
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

**Frontend:**
```bash
cd frontend
npm install
npm run dev
```

## 📁 Cấu trúc Dự án

```
pj-medicine/
├── backend/
│   ├── app/
│   │   ├── api/              # FastAPI route handlers
│   │   ├── models/           # SQLAlchemy ORM models
│   │   ├── schemas/          # Pydantic request/response schemas
│   │   ├── services/         # Business logic layer
│   │   ├── db/               # Database session & config
│   │   ├── config.py         # App settings
│   │   └── main.py           # FastAPI entry point
│   ├── ml/
│   │   ├── inference.py      # Prediction interface
│   │   └── models/weights/   # Model weights (.pt files)
│   ├── requirements.txt
│   └── Dockerfile
├── frontend/
│   ├── src/
│   │   ├── types/            # TypeScript interfaces
│   │   ├── services/         # API client (Axios)
│   │   ├── components/       # React components
│   │   ├── pages/            # Page components
│   │   └── hooks/            # Custom React hooks
│   ├── package.json
│   └── Dockerfile
├── schema.sql                # Database schema
├── docker-compose.yml        # Full stack Docker setup
├── Makefile                  # Dev commands
├── .env.example              # Environment variables template
└── techstack.md              # Chi tiết tech stack
```

## 🛠️ Tech Stack

| Layer | Công nghệ |
|-------|-----------|
| **Frontend** | React 19 · TypeScript · Vite · Tailwind CSS |
| **Backend** | FastAPI · Python 3.11 · SQLAlchemy · Pydantic |
| **ML Engine** | PyTorch · Mamba SSM · HuggingFace · Underthesea |
| **Database** | PostgreSQL 16 · Redis 7 |
| **DevOps** | Docker · GitHub Actions |

## 👥 Nhóm phát triển

| Role | Nhiệm vụ |
|------|----------|
| 🧠 ML Engineer | Data preprocessing, model training, inference |
| ⚙️ Backend Dev | FastAPI, database, API endpoints |
| 🎨 Frontend Dev 1 | UI/UX, components, pages |
| 🎨 Frontend Dev 2 | API integration, state management |
| 🐳 DevOps | Docker, CI/CD, deployment |
| 📝 Doc & Report | Documentation, testing, báo cáo |

## 📝 License

Dự án phục vụ mục đích nghiên cứu và học tập.

© 2026 Drug-Pred AI Team — HUTECH