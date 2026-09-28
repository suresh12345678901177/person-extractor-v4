<#
One-time setup for the optional LLM reviewer (src/review/llm_reviewer.py).
Run by hand, with internet: the tool itself never downloads or installs
anything, so it can run on an offline evidence workstation.

  - With a GPU (NVIDIA via nvidia-smi, or AMD Radeon): installs Ollama with
    winget if it is missing, and downloads the pinned model.
  - Without one: does nothing. Scans then run without the reviewer - on the
    CPU alone it would be about 10x slower (4.25 s vs 0.41 s per name) - and
    each scan's summary records why it was off.

    powershell -ExecutionPolicy Bypass -File scripts\setup_llm_reviewer.ps1
#>
$ErrorActionPreference = "Stop"
$model = "llama3.1:latest"
$pinnedDigest = "46e0c10c039e019119339687c3c1757cc81b9da49709a3b3924863ba87ca666e"  # config.py llm_reviewer.model_digest

$gpu = $null
if (Get-Command nvidia-smi -ErrorAction SilentlyContinue) {
    $gpu = (& nvidia-smi --query-gpu=name --format=csv,noheader | Select-Object -First 1)
}
if (-not $gpu) {
    $gpu = (Get-CimInstance Win32_VideoController | Where-Object { $_.Name -match "Radeon" } | Select-Object -First 1).Name
}
if (-not $gpu) {
    Write-Host "No supported GPU found - nothing to install. Scans run without the LLM reviewer."
    exit 0
}
Write-Host "GPU: $gpu"

$ollama = (Get-Command ollama -ErrorAction SilentlyContinue).Source
if (-not $ollama) {
    $ollama = Join-Path $env:LOCALAPPDATA "Programs\Ollama\ollama.exe"
}
if (-not (Test-Path $ollama)) {
    Write-Host "Installing Ollama (winget) ..."
    winget install --id Ollama.Ollama -e --silent --accept-package-agreements --accept-source-agreements
    $ollama = Join-Path $env:LOCALAPPDATA "Programs\Ollama\ollama.exe"
}
Write-Host "Ollama: $(& $ollama --version)"

try { Invoke-RestMethod -Uri http://127.0.0.1:11434/api/version -TimeoutSec 3 | Out-Null }
catch { Start-Process -WindowStyle Hidden $ollama -ArgumentList "serve"; Start-Sleep -Seconds 5 }

Write-Host "Downloading $model (about 4.9 GB) ..."
& $ollama pull $model

$installed = (Invoke-RestMethod -Uri http://127.0.0.1:11434/api/tags).models | Where-Object { $_.name -eq $model }
if ($installed.digest -ne $pinnedDigest) {
    Write-Warning ("$model is $($installed.digest.Substring(0,12)), not the pinned $($pinnedDigest.Substring(0,12)) - " +
                   "the reviewer stays off until the new model is measured and config.py's model_digest updated.")
    exit 1
}
Write-Host "Done: the LLM reviewer will be used on this machine (pinned model, GPU)."
