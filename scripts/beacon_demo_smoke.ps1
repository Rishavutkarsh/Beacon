param(
    [string]$Model = "beacon-gemma4-current-best",
    [string]$Question = "Can I run a generator in my garage if the door is open?",
    [int]$TimeoutSeconds = 360,
    [int]$NumPredict = 260
)

$ErrorActionPreference = "Stop"

$RepoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $RepoRoot

function Write-Step {
    param([string]$Message)
    Write-Host ""
    Write-Host "==> $Message" -ForegroundColor Cyan
}

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

    throw "Ollama was not found. Install Ollama or set OLLAMA_EXE to ollama.exe."
}

function Test-OllamaApi {
    try {
        Invoke-RestMethod -Uri "http://localhost:11434/api/tags" -TimeoutSec 5 | Out-Null
        return $true
    } catch {
        return $false
    }
}

Write-Step "Resolving Ollama"
$Ollama = Resolve-Ollama
Write-Host "Ollama: $Ollama"

if (-not (Test-OllamaApi)) {
    Write-Step "Starting Ollama server"
    Start-Process -FilePath $Ollama -ArgumentList "serve" -WindowStyle Hidden
    Start-Sleep -Seconds 3
}

if (-not (Test-OllamaApi)) {
    throw "Ollama API did not become reachable at http://localhost:11434."
}

Write-Step "Checking local GGUF and Ollama model"
$Gguf = Join-Path $RepoRoot "ollama\beacon-gemma4-e2b-current-best-q4_k_m.gguf"
if (-not (Test-Path $Gguf)) {
    throw "Missing GGUF at $Gguf. Download/export it before demo."
}

$modelList = & $Ollama list
if (($modelList -join "`n") -notmatch [regex]::Escape($Model)) {
    Write-Host "Model $Model not found in Ollama. Creating it from ollama\Modelfile..."
    & $Ollama create $Model -f "ollama\Modelfile"
} else {
    Write-Host "Model $Model is already imported."
}

Write-Step "Checking offline official-doc retrieval"
python scripts\beacon_docs_cli.py list-docs --query "generator carbon monoxide placement" --json

Write-Step "Checking Ollama API final-answer channel"
$payload = @{
    model = $Model
    prompt = "Say READY only."
    stream = $false
    think = $false
    options = @{
        temperature = 0
        num_predict = 32
    }
} | ConvertTo-Json -Depth 5

$ready = Invoke-RestMethod -Method Post -Uri "http://localhost:11434/api/generate" -Body $payload -ContentType "application/json" -TimeoutSec 120
if (($ready.response).Trim() -ne "READY") {
    throw "Ollama API smoke expected READY, got: '$($ready.response)'"
}
Write-Host "Ollama API returned READY."

Write-Step "Running Beacon grounded controller smoke"
python scripts\beacon_ollama_agent.py --model $Model --force-docs --num-predict $NumPredict --timeout-seconds $TimeoutSeconds $Question

Write-Step "Smoke complete"
Write-Host "Beacon demo path is ready: offline docs + local Ollama + grounded answer." -ForegroundColor Green
