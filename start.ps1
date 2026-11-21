# Start the Agri-RAG Advisor stack (backend + frontend) in separate windows.
# Usage:  .\start.ps1          (double-click won't work; run from a terminal)
$root = $PSScriptRoot

# Backend: auto-starts Ollama if it isn't running
Start-Process powershell -ArgumentList "-NoExit", "-ExecutionPolicy", "Bypass", `
  -File "$root\start-backend.ps1"

# Frontend
Start-Process powershell -ArgumentList "-NoExit", "-ExecutionPolicy", "Bypass", `
  -File "$root\start-frontend.ps1"

Write-Host ""
Write-Host "Starting:" -ForegroundColor Green
Write-Host "  Backend  -> http://localhost:8000  (opens in its own window)"
Write-Host "  Frontend -> http://localhost:5173  (opens in its own window)"
Write-Host "Open http://localhost:5173 in your browser once both are up."
