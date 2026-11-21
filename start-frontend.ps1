# React frontend (Vite dev server)
Set-Location "$PSScriptRoot\frontend"
if (-not (Test-Path "node_modules")) {
  Write-Host "Installing frontend dependencies (first run) ..." -ForegroundColor Yellow
  npm install
}
Write-Host "Starting frontend on http://localhost:5173 ..." -ForegroundColor Green
npm run dev
