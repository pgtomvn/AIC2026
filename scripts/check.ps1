[CmdletBinding()]
param(
    [switch]$IncludeCompose,
    [switch]$IncludeServices,
    [switch]$IncludeStreamlit,
    [switch]$IncludeArchives
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$IsWindowsHost = $env:OS -eq "Windows_NT"
$UvBinaryName = if ($IsWindowsHost) { "uv.exe" } else { "uv" }
$UvExe = Join-Path $ProjectRoot ".tools/uv/0.11.16/$UvBinaryName"
if (-not (Test-Path -LiteralPath $UvExe -PathType Leaf)) {
    throw "Project-local uv is missing. Run scripts/bootstrap.ps1 first."
}

$env:UV_CACHE_DIR = Join-Path $ProjectRoot ".cache/uv"
$env:UV_PYTHON_INSTALL_DIR = Join-Path $ProjectRoot ".tools/python"
$env:UV_PROJECT_ENVIRONMENT = Join-Path $ProjectRoot ".venv"
$env:UV_PYTHON_NO_REGISTRY = "1"
$env:PYTHONUTF8 = "1"

function Invoke-Checked {
    param(
        [Parameter(Mandatory = $true)][string]$FilePath,
        [Parameter(ValueFromRemainingArguments = $true)][string[]]$Arguments
    )

    & $FilePath @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "$FilePath failed with exit code $LASTEXITCODE."
    }
}

function Get-DotEnvValue {
    param([Parameter(Mandatory = $true)][string]$Name)

    $envPath = Join-Path $ProjectRoot ".env"
    foreach ($line in Get-Content -LiteralPath $envPath -Encoding UTF8) {
        if ($line -match "^$([regex]::Escape($Name))=(.*)$") {
            return $Matches[1].Trim().Trim('"').Trim("'")
        }
    }
    return $null
}

Push-Location $ProjectRoot
try {
    Invoke-Checked $UvExe run --frozen ruff check .
    Invoke-Checked $UvExe run --frozen pytest -q
    if ($IncludeArchives) {
        Invoke-Checked $UvExe run --frozen python scripts/extract_data.py --verify
    }

    if ($IncludeCompose -or $IncludeServices) {
        if (-not (Get-Command docker -ErrorAction SilentlyContinue)) {
            throw "Docker is not available on PATH."
        }
        if (-not (Test-Path -LiteralPath (Join-Path $ProjectRoot ".env"))) {
            throw ".env is missing. Run scripts/bootstrap.ps1 first."
        }
        Invoke-Checked "docker" compose --env-file .env config --quiet
    }

    if ($IncludeServices) {
        $qdrantUrl = Get-DotEnvValue "QDRANT_URL"
        $meiliUrl = Get-DotEnvValue "MEILISEARCH_URL"
        $qdrantKey = Get-DotEnvValue "QDRANT_API_KEY"
        if (-not $qdrantUrl) { $qdrantUrl = "http://127.0.0.1:6333" }
        if (-not $meiliUrl) { $meiliUrl = "http://127.0.0.1:7700" }
        $headers = @{}
        if ($qdrantKey) { $headers["api-key"] = $qdrantKey }
        $qdrant = Invoke-WebRequest -UseBasicParsing -Headers $headers -Uri "$($qdrantUrl.TrimEnd('/'))/healthz"
        $meili = Invoke-WebRequest -UseBasicParsing -Uri "$($meiliUrl.TrimEnd('/'))/health"
        if ($qdrant.StatusCode -ne 200 -or $meili.StatusCode -ne 200) {
            throw "One or more local search services are unhealthy."
        }
        Write-Host "Qdrant and Meilisearch health checks passed."
    }

    if ($IncludeStreamlit) {
        $python = if ($IsWindowsHost) {
            Join-Path $ProjectRoot ".venv/Scripts/python.exe"
        }
        else {
            Join-Path $ProjectRoot ".venv/bin/python"
        }
        if (-not (Test-Path -LiteralPath $python -PathType Leaf)) {
            throw "The project virtual environment is missing. Run scripts/bootstrap.ps1 first."
        }
        $checkCache = Join-Path $ProjectRoot ".cache/check"
        New-Item -ItemType Directory -Force -Path $checkCache | Out-Null
        $stdoutPath = Join-Path $checkCache "streamlit.out.log"
        $stderrPath = Join-Path $checkCache "streamlit.err.log"
        $startParameters = @{
            FilePath = $python
            ArgumentList = @(
                "-m", "streamlit", "run", "app/streamlit_app.py",
                "--server.headless", "true", "--server.address", "127.0.0.1",
                "--server.port", "8501", "--browser.gatherUsageStats", "false"
            )
            WorkingDirectory = $ProjectRoot
            PassThru = $true
            RedirectStandardOutput = $stdoutPath
            RedirectStandardError = $stderrPath
        }
        if ($IsWindowsHost) { $startParameters["WindowStyle"] = "Hidden" }
        $streamlit = Start-Process @startParameters
        try {
            $healthy = $false
            foreach ($attempt in 1..20) {
                if ($streamlit.HasExited) { break }
                try {
                    $response = Invoke-WebRequest -UseBasicParsing `
                        -Uri "http://127.0.0.1:8501/_stcore/health" -TimeoutSec 2
                    if ($response.StatusCode -eq 200) { $healthy = $true; break }
                }
                catch { }
                Start-Sleep -Seconds 1
            }
            if (-not $healthy) { throw "Streamlit did not become healthy within 20 seconds." }
            Write-Host "Streamlit headless smoke check passed."
        }
        finally {
            if (-not $streamlit.HasExited) { Stop-Process -Id $streamlit.Id -Force }
        }
    }
}
finally {
    Pop-Location
}

Write-Host "All requested checks passed."
