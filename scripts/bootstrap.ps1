[CmdletBinding()]
param(
    [switch]$GenerateLock,
    [switch]$NoEnvFile
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"
if (Get-Variable PSNativeCommandUseErrorActionPreference -ErrorAction SilentlyContinue) {
    $PSNativeCommandUseErrorActionPreference = $true
}
[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12

$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$UvVersion = "0.11.16"
$PythonVersion = (Get-Content -LiteralPath (Join-Path $ProjectRoot ".python-version") -Raw).Trim()
$IsWindowsHost = $env:OS -eq "Windows_NT"
$RuntimeArchitecture = [Runtime.InteropServices.RuntimeInformation]::OSArchitecture.ToString()

switch ($RuntimeArchitecture) {
    "X64" { $AssetArchitecture = "x86_64" }
    "Arm64" { $AssetArchitecture = "aarch64" }
    default { throw "Unsupported architecture for project-local uv: $RuntimeArchitecture" }
}

if ($IsWindowsHost) {
    $UvAsset = "uv-$AssetArchitecture-pc-windows-msvc.zip"
    $UvBinaryName = "uv.exe"
    $ArchiveKind = "zip"
}
elseif ($IsLinux) {
    $UvAsset = "uv-$AssetArchitecture-unknown-linux-gnu.tar.gz"
    $UvBinaryName = "uv"
    $ArchiveKind = "tar"
}
else {
    throw "bootstrap.ps1 currently supports Windows and Linux."
}

$UvRoot = Join-Path $ProjectRoot ".tools/uv/$UvVersion"
$UvExe = Join-Path $UvRoot $UvBinaryName
$DownloadRoot = Join-Path $ProjectRoot ".tools/downloads"
$UvArchive = Join-Path $DownloadRoot $UvAsset
$UvChecksum = "$UvArchive.sha256"
$ReleaseBase = "https://github.com/astral-sh/uv/releases/download/$UvVersion"

function New-RandomHex {
    param([int]$ByteCount = 32)

    $bytes = New-Object byte[] $ByteCount
    $generator = [Security.Cryptography.RandomNumberGenerator]::Create()
    try { $generator.GetBytes($bytes) }
    finally { $generator.Dispose() }
    return -join ($bytes | ForEach-Object { $_.ToString("x2") })
}

function Invoke-NativeChecked {
    param(
        [Parameter(Mandatory = $true)][string]$FilePath,
        [Parameter(ValueFromRemainingArguments = $true)][string[]]$Arguments
    )

    & $FilePath @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "$FilePath failed with exit code $LASTEXITCODE."
    }
}

if (-not (Test-Path -LiteralPath $UvExe -PathType Leaf)) {
    New-Item -ItemType Directory -Force -Path $UvRoot, $DownloadRoot | Out-Null
    Write-Host "Downloading project-local uv $UvVersion ($UvAsset)..."
    Invoke-WebRequest -UseBasicParsing -Uri "$ReleaseBase/$UvAsset" -OutFile $UvArchive
    Invoke-WebRequest -UseBasicParsing -Uri "$ReleaseBase/$UvAsset.sha256" -OutFile $UvChecksum

    $checksumText = Get-Content -LiteralPath $UvChecksum -Raw
    $checksumMatch = [regex]::Match($checksumText, "(?i)[a-f0-9]{64}")
    if (-not $checksumMatch.Success) {
        throw "The published uv checksum file did not contain a SHA-256 digest."
    }
    $expectedHash = $checksumMatch.Value.ToLowerInvariant()
    $actualHash = (Get-FileHash -LiteralPath $UvArchive -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($actualHash -ne $expectedHash) {
        throw "uv archive checksum mismatch; refusing to extract the download."
    }

    $extractRoot = Join-Path $DownloadRoot "uv-extract-$PID"
    if (Test-Path -LiteralPath $extractRoot) {
        Remove-Item -LiteralPath $extractRoot -Recurse -Force
    }
    New-Item -ItemType Directory -Force -Path $extractRoot | Out-Null
    try {
        if ($ArchiveKind -eq "zip") {
            Expand-Archive -LiteralPath $UvArchive -DestinationPath $extractRoot -Force
        }
        else {
            Invoke-NativeChecked "tar" -xzf $UvArchive -C $extractRoot
        }
        $extractedUv = Get-ChildItem -LiteralPath $extractRoot -File -Recurse |
            Where-Object { $_.Name -eq $UvBinaryName } |
            Select-Object -First 1
        if (-not $extractedUv) {
            throw "The verified uv archive did not contain $UvBinaryName."
        }
        Copy-Item -LiteralPath $extractedUv.FullName -Destination $UvExe -Force
        if (-not $IsWindowsHost) {
            Invoke-NativeChecked "chmod" "+x" $UvExe
        }
    }
    finally {
        if (Test-Path -LiteralPath $extractRoot) {
            Remove-Item -LiteralPath $extractRoot -Recurse -Force
        }
    }
}

$uvVersionOutput = & $UvExe --version
if ($LASTEXITCODE -ne 0 -or $uvVersionOutput -notmatch "^uv $([regex]::Escape($UvVersion))\b") {
    throw "Unexpected uv binary at $UvExe. Expected uv $UvVersion, got '$uvVersionOutput'."
}

$env:UV_CACHE_DIR = Join-Path $ProjectRoot ".cache/uv"
$env:UV_PYTHON_INSTALL_DIR = Join-Path $ProjectRoot ".tools/python"
$env:UV_PROJECT_ENVIRONMENT = Join-Path $ProjectRoot ".venv"
$env:UV_PYTHON_NO_REGISTRY = "1"
$env:PYTHONUTF8 = "1"

Push-Location $ProjectRoot
try {
    Invoke-NativeChecked $UvExe python install $PythonVersion

    $lockPath = Join-Path $ProjectRoot "uv.lock"
    if ($GenerateLock) {
        Invoke-NativeChecked $UvExe lock
    }
    elseif (-not (Test-Path -LiteralPath $lockPath -PathType Leaf)) {
        throw "uv.lock is missing. Maintainers must run bootstrap.ps1 -GenerateLock once."
    }

    # PaddleOCR's optional competition stack installs opencv-contrib-python,
    # which shares the cv2 package directory with the locked base headless
    # wheel. Removing that group can remove shared files, so always reinstall
    # the base wheel atomically during a normal-environment sync.
    Invoke-NativeChecked $UvExe sync --frozen --reinstall-package opencv-python-headless
}
finally {
    Pop-Location
}

if (-not $NoEnvFile) {
    $envPath = Join-Path $ProjectRoot ".env"
    if (-not (Test-Path -LiteralPath $envPath)) {
        $qdrantKey = New-RandomHex
        $meilisearchKey = New-RandomHex
        $environmentText = @"
# Generated by scripts/bootstrap.ps1. Never commit this file.
QDRANT_API_KEY=$qdrantKey
MEILISEARCH_MASTER_KEY=$meilisearchKey
QDRANT_URL=http://127.0.0.1:6333
MEILISEARCH_URL=http://127.0.0.1:7700
  # Single-frame Visual Q&A: consent-gated Gemini with local Ollama fallback.
  OLLAMA_URL=http://127.0.0.1:11434
  OLLAMA_LLM_LIBRARY=
  GEMINI_API_KEY=
AIC_API_URL=http://127.0.0.1:8080
CUDA_VISIBLE_DEVICES=0
"@
        $utf8NoBom = New-Object Text.UTF8Encoding($false)
        [IO.File]::WriteAllText($envPath, $environmentText, $utf8NoBom)
        Write-Host "Created .env with new local service credentials."
    }
    else {
        Write-Host "Preserved existing .env."
    }
}

Write-Host "Bootstrap complete. Run ./scripts/check.ps1 next."
