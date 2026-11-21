# Backend API + RAG pipeline (auto-starts Ollama if down)
Set-Location "$PSScriptRoot\backend"
Write-Host "Starting backend on http://localhost:8000 ..." -ForegroundColor Green
& ".\.venv\Scripts\python.exe" -m uvicorn app.main:app --port 8000
