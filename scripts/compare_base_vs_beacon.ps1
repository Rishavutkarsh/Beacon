param(
    [string]$Question = "Floodwater touch hue baby bottle nipples/pacifiers ko bleach se sanitize karke use kar sakte hain?",
    [string]$BaseModel = "gemma4:e2b",
    [string]$BeaconModel = "beacon-gemma4-current-best",
    [switch]$ToolLoop
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
Write-Host "Question:" -ForegroundColor Cyan
Write-Host $Question
Write-Host ""
Write-Host "=== Base Gemma 4 E2B, no Beacon prompt, no offline docs ===" -ForegroundColor Yellow
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
if ($ToolLoop) {
    Write-Host "=== Beacon DPO + canonical offline official-doc tool loop ===" -ForegroundColor Green
    python scripts\beacon_ollama_tool_loop_agent.py --model $BeaconModel --num-predict 220 --timeout-seconds 360 $Question
} else {
    Write-Host "=== Beacon DPO + preloaded offline official-doc evidence ===" -ForegroundColor Green
    python scripts\beacon_ollama_tool_loop_agent.py --model $BeaconModel --preload-docs --num-predict 220 --timeout-seconds 360 $Question
}
