[CmdletBinding()]
param(
    [switch]$IncludeCompose,
    [switch]$IncludeServices,
    [switch]$IncludeStreamlit,
    [switch]$IncludeStreamlitRuntime,
    [switch]$IncludeArchives,
    [switch]$IncludeGemini,
    [switch]$IncludeQwen
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
    # A unique base avoids collisions when a cancelled verification process is
    # still releasing Windows file handles from a prior run.
    $pytestRunId = [guid]::NewGuid().ToString("N")
    $pytestBaseTemp = Join-Path $ProjectRoot ".cache/pytest-run-$pytestRunId"
    $pytestCache = Join-Path $ProjectRoot ".cache/pytest-cache-$pytestRunId"
    New-Item -ItemType Directory -Force -Path (Split-Path $pytestBaseTemp -Parent) | Out-Null
    Invoke-Checked $UvExe run --frozen ruff check .
    Invoke-Checked $UvExe run --frozen pytest -q `
        --basetemp $pytestBaseTemp "--override-ini=cache_dir=$pytestCache"
    if ($IncludeArchives) {
        Invoke-Checked $UvExe run --frozen python scripts/extract_data.py --verify
    }

    if ($IncludeGemini) {
        # This is the sole normal verification path that intentionally uses an
        # external provider. It requires a rotated key in untracked .env.
        Invoke-Checked $UvExe run --frozen python scripts/check_gemini.py
    }

    if ($IncludeQwen) {
        # Explicit accelerated-profile GPU integration; never part of normal checks.
        Invoke-Checked $UvExe run --frozen --group competition python scripts/check_qwen.py
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
        # AppTest executes the application script itself and catches import/render
        # exceptions that Streamlit's liveness endpoint cannot detect.
        Invoke-Checked $UvExe run --frozen python scripts/check_streamlit.py
    }

    if ($IncludeStreamlitRuntime) {
        if ($IsWindowsHost) {
            # Some managed shells supply both Path and PATH in the process
            # block. Windows PowerShell 5.1 cannot copy that block in
            # Start-Process, so normalize it locally before spawning the smoke
            # server. This does not alter the user or machine environment.
            $processPath = [System.Environment]::GetEnvironmentVariable("Path", "Process")
            [System.Environment]::SetEnvironmentVariable("PATH", $null, "Process")
            [System.Environment]::SetEnvironmentVariable("Path", $processPath, "Process")
        }
        $runtimeEnvironment = Join-Path $ProjectRoot ".venv-runtime"
        $python = if ($IsWindowsHost) {
            Join-Path $runtimeEnvironment "Scripts/python.exe"
        }
        else {
            Join-Path $runtimeEnvironment "bin/python"
        }
        if (-not (Test-Path -LiteralPath $python -PathType Leaf)) {
            throw "The competition runtime is missing. Run scripts/bootstrap_runtime.ps1 first."
        }
        $stampPath = Join-Path $runtimeEnvironment ".aic-runtime-stamp.json"
        if (-not (Test-Path -LiteralPath $stampPath -PathType Leaf)) {
            throw "The competition runtime stamp is missing. Run scripts/bootstrap_runtime.ps1."
        }
        $stamp = Get-Content -Raw -LiteralPath $stampPath -Encoding UTF8 | ConvertFrom-Json
        $lockHash = (Get-FileHash -LiteralPath (Join-Path $ProjectRoot "uv.lock") -Algorithm SHA256).Hash.ToLowerInvariant()
        if ($stamp.schema_version -ne 1 -or $stamp.uv_lock_sha256 -ne $lockHash) {
            throw "The competition runtime stamp is stale. Run scripts/bootstrap_runtime.ps1."
        }
        $env:AIC_PROFILE = $stamp.profile
        $env:PADDLE_PDX_CACHE_HOME = Join-Path $ProjectRoot ".cache/paddlex"
        $env:PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK = "True"
        Invoke-Checked $python scripts/check_streamlit.py

        # Exercise the canonical launcher from outside the repository while an
        # exclusive listener owns the requested port. It must fail before the
        # readiness doctor or Streamlit starts, and it must never stop the owner.
        $occupiedListener = [System.Net.Sockets.TcpListener]::new(
            [System.Net.IPAddress]::Loopback,
            0
        )
        $occupiedListener.Server.ExclusiveAddressUse = $true
        $occupiedListener.Start()
        $occupiedPort = ([System.Net.IPEndPoint]$occupiedListener.LocalEndpoint).Port
        $launcherGuarded = $false
        try {
            Push-Location ([IO.Path]::GetTempPath())
            try {
                & (Join-Path $ProjectRoot "scripts/run_app.ps1") `
                    -Profile $stamp.profile -BatchId "aic25-b1-reference" -Port $occupiedPort
            }
            catch {
                if ($_.Exception.Message -match "already owned|already in use") {
                    $launcherGuarded = $true
                }
                else { throw }
            }
            finally { Pop-Location }
        }
        finally { $occupiedListener.Stop() }
        if (-not $launcherGuarded) {
            throw "The canonical launcher did not refuse an occupied port."
        }
        Write-Host "Competition launcher occupied-port and root-directory checks passed."

        $portListener = [System.Net.Sockets.TcpListener]::new(
            [System.Net.IPAddress]::Loopback,
            0
        )
        $portListener.Start()
        $streamlitPort = ([System.Net.IPEndPoint]$portListener.LocalEndpoint).Port
        $portListener.Stop()
        $checkCache = Join-Path $ProjectRoot ".cache/check"
        New-Item -ItemType Directory -Force -Path $checkCache | Out-Null
        $stdoutPath = Join-Path $checkCache "streamlit.out.log"
        $stderrPath = Join-Path $checkCache "streamlit.err.log"
        $startParameters = @{
            FilePath = $python
            ArgumentList = @(
                "-m", "streamlit", "run", "app/streamlit_app.py",
                "--server.headless", "true", "--server.address", "127.0.0.1",
                "--server.port", "$streamlitPort", "--browser.gatherUsageStats", "false"
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
                        -Uri "http://127.0.0.1:$streamlitPort/_stcore/health" -TimeoutSec 2
                    if ($response.StatusCode -eq 200) { $healthy = $true; break }
                }
                catch { }
                Start-Sleep -Seconds 1
            }
            if (-not $healthy) { throw "Streamlit did not become healthy within 20 seconds." }
            Write-Host "Competition Streamlit runtime smoke check passed on ephemeral port $streamlitPort."
        }
        finally {
            if (-not $streamlit.HasExited) { Stop-Process -Id $streamlit.Id -Force }
            $streamlit.WaitForExit(10000) | Out-Null
        }
        $runtimeLog = @(
            Get-Content -Raw -LiteralPath $stdoutPath -ErrorAction SilentlyContinue
            Get-Content -Raw -LiteralPath $stderrPath -ErrorAction SilentlyContinue
        ) -join "`n"
        $forbiddenRuntimeLogPatterns = @(
            "(?im)^Traceback \(most recent call last\):",
            "(?i)Uncaught app execution",
            "(?i)use_container_width",
            "(?i)Api key is used with an insecure connection\."
        )
        foreach ($pattern in $forbiddenRuntimeLogPatterns) {
            if ($runtimeLog -match $pattern) {
                throw "Competition Streamlit runtime log contains a forbidden warning or traceback."
            }
        }
        Write-Host "Competition Streamlit runtime logs contain no forbidden warnings or tracebacks."
    }
}
finally {
    Pop-Location
}

Write-Host "All requested checks passed."
