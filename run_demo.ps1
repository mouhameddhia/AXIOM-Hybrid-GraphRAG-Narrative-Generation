<#
.SYNOPSIS
  Starts the AXIOM demo front end in your browser.

.DESCRIPTION
  First run: creates the virtual environment, installs the packages and copies
  .env.example to .env. Later runs just start the server.

  The language model is needed for "exact answer" and "explain a price". Start it
  separately with:  ollama serve   (and once:  ollama pull qwen2.5:7b)

.EXAMPLE
  .\run_demo.ps1
  .\run_demo.ps1 -Port 9000
  .\run_demo.ps1 -Dense      # adds the dense retrieval stream (about 50 s longer to start)
#>
param(
  [int]$Port = 8000,
  [switch]$Dense
)

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

if (-not (Test-Path ".venv\Scripts\python.exe")) {
  Write-Host "Creating the virtual environment (first run only)..."
  python -m venv .venv
  if ($LASTEXITCODE -ne 0) { throw "Python 3.13 (or later) is required: https://www.python.org/downloads/" }
}
$py = ".\.venv\Scripts\python.exe"

& $py -c "import rank_bm25, fastapi, uvicorn" 2>$null
if ($LASTEXITCODE -ne 0) {
  Write-Host "Installing packages (first run only, a few minutes)..."
  & $py -m pip install --quiet -r requirements.txt -r requirements-demo.txt
  if ($LASTEXITCODE -ne 0) { throw "Package installation failed. Check your internet connection." }
}

if (-not (Test-Path ".env")) {
  Copy-Item ".env.example" ".env"
  Write-Host "Created .env from .env.example (the configuration used for the reported results)."
}

if ($Dense) { $env:DEMO_DENSE = "1" } else { Remove-Item Env:DEMO_DENSE -ErrorAction SilentlyContinue }

try {
  $null = Invoke-WebRequest -UseBasicParsing "http://localhost:11434/api/tags" -TimeoutSec 2
  Write-Host "Language model server: reachable."
} catch {
  Write-Host "WARNING: the language model server is not running. Evidence search works, but 'Exact answer' and 'Explain a price' need it: run 'ollama serve' in another window." -ForegroundColor Yellow
}

$url = "http://127.0.0.1:$Port"
Start-Job -ScriptBlock { param($u) Start-Sleep -Seconds 5; Start-Process $u } -ArgumentList $url | Out-Null
Write-Host "Starting the demo at $url  (press Ctrl+C to stop)"
& $py -m uvicorn demo_app.server:app --host 127.0.0.1 --port $Port --log-level warning
