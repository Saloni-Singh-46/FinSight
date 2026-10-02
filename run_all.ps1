# FinSight One-Click PowerShell Launcher
Write-Host "======================================================================" -ForegroundColor Cyan
Write-Host "  🚀 FINSIGHT: MULTI-AGENT AI FINANCIAL RESEARCH ANALYST" -ForegroundColor Cyan
Write-Host "======================================================================" -ForegroundColor Cyan

$env:PYTHONPATH = "src"
$PythonExe = ".\.venv\Scripts\python.exe"

if (-Not (Test-Path $PythonExe)) {
    $PythonExe = "python"
}

Write-Host "[*] Launching Unified Services via run_all.py..." -ForegroundColor Green
& $PythonExe run_all.py
