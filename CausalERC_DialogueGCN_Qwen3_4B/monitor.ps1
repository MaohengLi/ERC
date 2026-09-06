param([string]$OutDir = "outputs/qwen3_4b")
while ($true) {
    Clear-Host
    Get-Date
    $progress = Join-Path $OutDir "progress.json"
    if (Test-Path -LiteralPath $progress) { Get-Content -LiteralPath $progress }
    else { Write-Host "progress.json not created yet" }
    nvidia-smi
    Start-Sleep -Seconds 10
}
