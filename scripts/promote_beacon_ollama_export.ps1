param(
    [string]$Kernel = "rishavutkarsh/beacon-ollama-export-current-best",
    [string]$Model = "beacon-gemma4-current-best",
    [string]$ExpectedGgufName = "beacon-gemma4-e2b-current-best-q4_k_m.gguf",
    [string]$DownloadDir = "kaggle_outputs\beacon_ollama_export_current_best_latest"
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
    throw "Ollama was not found. Install Ollama or set OLLAMA_EXE."
}

Write-Step "Downloading latest Kaggle export output"
$targetDir = Join-Path $RepoRoot $DownloadDir
New-Item -ItemType Directory -Force -Path $targetDir | Out-Null
kaggle kernels output $Kernel -p $targetDir

Write-Step "Finding GGUF artifact"
$gguf = Get-ChildItem -Path $targetDir -Recurse -Filter $ExpectedGgufName | Select-Object -First 1
if (-not $gguf) {
    throw "Could not find $ExpectedGgufName under $targetDir."
}
Write-Host "Found: $($gguf.FullName)"

Write-Step "Installing GGUF into ollama package directory"
$ollamaDir = Join-Path $RepoRoot "ollama"
New-Item -ItemType Directory -Force -Path $ollamaDir | Out-Null
$dest = Join-Path $ollamaDir $ExpectedGgufName
Copy-Item -Path $gguf.FullName -Destination $dest -Force
Get-FileHash $dest -Algorithm SHA256

Write-Step "Recreating Ollama model"
$ollama = Resolve-Ollama
if (-not (Invoke-RestMethod -Uri "http://localhost:11434/api/tags" -TimeoutSec 5 -ErrorAction SilentlyContinue)) {
    Start-Process -FilePath $ollama -ArgumentList "serve" -WindowStyle Hidden
    Start-Sleep -Seconds 3
}
& $ollama create $Model -f "ollama\Modelfile"

Write-Step "Running grounded demo question"
python scripts\beacon_ollama_agent.py --model $Model --force-docs --num-predict 260 --timeout-seconds 360 "Can I run a generator in my garage if the door is open?"

Write-Step "Promotion complete"
Write-Host "Latest Beacon Ollama export is installed and demo-tested." -ForegroundColor Green
