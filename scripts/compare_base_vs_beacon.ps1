param(
    [string]$Question = "Floodwater touch hue baby bottle nipples/pacifiers ko bleach se sanitize karke use kar sakte hain?",
    [string]$BaseModel = "gemma4:e2b",
    [string]$BeaconModel = "beacon-gemma4-current-best"
)

$ErrorActionPreference = "Stop"
$RepoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $RepoRoot

function Resolve-Ollama {
    if ($env:OLLAMA_EXE -and (Test-Path $env:OLLAMA_EXE)) {
        return (Resolve-Path $env:OLLAMA_EXE).Path
    }
    $command = Get-Command ollama -ErrorAction SilentlyContinue
    if ($command) {
        return $command.Source
    }
    $portable = Join-Path $env:LOCALAPPDATA "Programs\OllamaPortable\ollama.exe"
    if (Test-Path $portable) {
        return $portable
    }
    throw "Ollama was not found. Install Ollama or set OLLAMA_EXE."
}

$ollama = Resolve-Ollama

Write-Host ""
Write-Host "=== Base Gemma 4 E2B, no Beacon tools ===" -ForegroundColor Yellow
$payload = @{
    model = $BaseModel
    prompt = $Question
    stream = $false
    think = $false
    options = @{
        temperature = 0.2
        num_predict = 220
    }
} | ConvertTo-Json -Depth 5
$base = Invoke-RestMethod -Method Post -Uri "http://localhost:11434/api/generate" -Body $payload -ContentType "application/json" -TimeoutSec 360
Write-Host $base.response

Write-Host ""
Write-Host "=== Beacon DPO + offline official-doc tool ===" -ForegroundColor Green
python scripts\beacon_ollama_agent.py --model $BeaconModel --force-docs --num-predict 220 --timeout-seconds 360 $Question
