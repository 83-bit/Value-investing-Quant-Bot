# 打包 Invest Bot 視窗版 InvestBot.exe（含圖示）
# 用法: .\build_exe.ps1

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

Write-Host "Installing build dependencies..."
python -m pip install -q -r requirements.txt pyinstaller pillow

Write-Host "Generating icon..."
python assets\generate_icon.py

Write-Host "Building InvestBot.exe (GUI, no console)..."
python -m PyInstaller --noconfirm invest_bot.spec

$exe = Join-Path $PSScriptRoot "dist\InvestBot.exe"
if (Test-Path $exe) {
    Write-Host ""
    Write-Host "Done: $exe"
    Write-Host ""
    Write-Host "雙擊開啟圖形介面。請在 dist 或 exe 同目錄放置 .env："
    Write-Host "  DEEPSEEK_API_KEY=..."
    Write-Host "  FMP_API_KEY=..."
} else {
    Write-Error "Build failed — InvestBot.exe not found"
}
