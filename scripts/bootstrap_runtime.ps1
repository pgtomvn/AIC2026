[CmdletBinding()]
param(
    [Alias("Profile")]
    [ValidateSet("laptop", "accelerated")]
    [string]$RuntimeProfile = "laptop"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$IsWindowsHost = $env:OS -eq "Windows_NT"
$UvBinaryName = if ($IsWindowsHost) { "uv.exe" } else { "uv" }
$UvExe = Join-Path $ProjectRoot ".tools/uv/0.11.16/$UvBinaryName"
$RuntimeEnvironment = Join-Path $ProjectRoot ".venv-runtime"
$Python = if ($IsWindowsHost) {
    Join-Path $RuntimeEnvironment "Scripts/python.exe"
}
else {
    Join-Path $RuntimeEnvironment "bin/python"
}

if (-not (Test-Path -LiteralPath $UvExe -PathType Leaf)) {
    throw "Project-local uv is missing. Run scripts/bootstrap.ps1 first."
}

$env:UV_CACHE_DIR = Join-Path $ProjectRoot ".cache/uv"
$env:UV_PYTHON_INSTALL_DIR = Join-Path $ProjectRoot ".tools/python"
$env:UV_PROJECT_ENVIRONMENT = $RuntimeEnvironment
$env:UV_PYTHON_NO_REGISTRY = "1"
$env:PYTHONUTF8 = "1"
$env:AIC_PROFILE = $RuntimeProfile
# PaddleX creates its cache directory while ``paddleocr`` is imported. Keep
# that import-time state inside the ignored project cache rather than the
# operator's home directory.
$env:PADDLE_PDX_CACHE_HOME = Join-Path $ProjectRoot ".cache/paddlex"
$env:PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK = "True"

Push-Location $ProjectRoot
try {
    & $UvExe sync --frozen --no-dev --group competition
    if ($LASTEXITCODE -ne 0) {
        throw "Competition runtime dependency synchronization failed with exit code $LASTEXITCODE."
    }
    if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) {
        throw "uv completed without creating the dedicated runtime interpreter."
    }

    $imports = @(
        "av", "faster_whisper", "google.genai", "open_clip", "paddleocr",
        "streamlit", "torch", "torchvision", "transformers"
    ) -join ","
    & $Python -c "import importlib; [importlib.import_module(name) for name in '$imports'.split(',')]"
    if ($LASTEXITCODE -ne 0) {
        throw "The competition runtime import verification failed with exit code $LASTEXITCODE."
    }

    $PythonVersion = (& $Python -c "import platform; print(platform.python_version())").Trim()
    if ($LASTEXITCODE -ne 0 -or -not $PythonVersion) {
        throw "Unable to determine the competition runtime Python version."
    }
    $LockHash = (Get-FileHash -LiteralPath (Join-Path $ProjectRoot "uv.lock") -Algorithm SHA256).Hash.ToLowerInvariant()
    $Stamp = [ordered]@{
        schema_version = 1
        profile = $RuntimeProfile
        python_version = $PythonVersion
        uv_lock_sha256 = $LockHash
        created_at_utc = [DateTime]::UtcNow.ToString("o")
    }
    $StampPath = Join-Path $RuntimeEnvironment ".aic-runtime-stamp.json"
    $Stamp | ConvertTo-Json | Set-Content -LiteralPath $StampPath -Encoding UTF8
    Write-Host "Competition runtime is ready: $RuntimeEnvironment ($RuntimeProfile, Python $PythonVersion)"
}
finally {
    Pop-Location
}
