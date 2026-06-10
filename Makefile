.PHONY: up down logs backend frontend db migrate seed clean

# ============================================
# Drug-Pred AI — Development Commands
# ============================================

# --- Docker ---
up:
	docker compose up -d
	@echo "✅ Stack is running!"
	@echo "   Frontend: http://localhost:5173"
	@echo "   Backend:  http://localhost:8000"
	@echo "   API Docs: http://localhost:8000/docs"

down:
	docker compose down

logs:
	docker compose logs -f

restart:
	docker compose restart

# --- Individual Services ---
backend:
	docker compose up -d backend

frontend:
	docker compose up -d frontend

db:
	docker compose up -d db redis

# --- Database ---
migrate:
	docker compose exec backend alembic upgrade head

seed:
	docker compose exec backend python -m app.seed

psql:
	docker compose exec db psql -U admin -d pj_medicine

# --- Development (without Docker) ---
dev-backend:
	cd backend && uvicorn app.main:app --reload --port 8000

dev-frontend:
	cd frontend && npm run dev

# --- Cleanup ---
clean:
	docker compose down -v --remove-orphans
	@echo "🧹 Cleaned up all containers and volumes"
