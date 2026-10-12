[CmdletBinding()]
param(
    [Alias("Profile")]
    [ValidateSet("laptop", "accelerated")]
    [string]$RuntimeProfile,
    [string]$BatchId,
    [ValidateRange(1, 65535)]
    [int]$Port = 8504,
    [ValidateRange(1, 65535)]
    [int]$ApiPort = 8080,
    [ValidateRange(1, 65535)]
    [int]$OllamaPort = 11434,
    [switch]$AllowDegraded,
    [switch]$StartServices
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$IsWindowsHost = $env:OS -eq "Windows_NT"
$RuntimeEnvironment = Join-Path $ProjectRoot ".venv-runtime"
$Python = if ($IsWindowsHost) {
    Join-Path $RuntimeEnvironment "Scripts/python.exe"
}
else {
    Join-Path $RuntimeEnvironment "bin/python"
}
$StampPath = Join-Path $RuntimeEnvironment ".aic-runtime-stamp.json"

function Get-LocalEnvValue {
    param([Parameter(Mandatory = $true)][string]$Name)

    $envPath = Join-Path $ProjectRoot ".env"
    if (-not (Test-Path -LiteralPath $envPath -PathType Leaf)) { return $null }
    foreach ($line in Get-Content -LiteralPath $envPath -Encoding UTF8) {
        if ($line -match "^$([regex]::Escape($Name))=(.*)$") {
            return $Matches[1].Trim().Trim('"').Trim("'")
        }
    }
    return $null
}

function Get-YamlRuntimeDefault {
    param([Parameter(Mandatory = $true)][ValidateSet("profile", "batch")][string]$Name)

    $settingsPath = Join-Path $ProjectRoot "config/settings.yaml"
    if (-not (Test-Path -LiteralPath $settingsPath -PathType Leaf)) { return $null }
    $insideRuntime = $false
    foreach ($line in Get-Content -LiteralPath $settingsPath -Encoding UTF8) {
        if ($Name -eq "profile" -and $line -match "^profile:\s*([^#]+?)\s*$") {
            return $Matches[1].Trim().Trim('"').Trim("'")
        }
        if ($line -match "^runtime:\s*$") {
            $insideRuntime = $true
            continue
        }
        if ($insideRuntime -and $line -match "^[^\s#]") { $insideRuntime = $false }
        if ($Name -eq "batch" -and $insideRuntime -and
            $line -match "^\s+default_batch_id:\s*([^#]+?)\s*$") {
            return $Matches[1].Trim().Trim('"').Trim("'")
        }
    }
    return $null
}

$ProfileWasExplicit = $PSBoundParameters.ContainsKey("RuntimeProfile")
$BatchWasExplicit = $PSBoundParameters.ContainsKey("BatchId")
if (-not $ProfileWasExplicit) {
    $RuntimeProfile = if ($env:AIC_PROFILE) {
        $env:AIC_PROFILE
    }
    else {
        $localValue = Get-LocalEnvValue -Name "AIC_PROFILE"
        if ($localValue) { $localValue } else { Get-YamlRuntimeDefault -Name "profile" }
    }
}
if (-not $BatchWasExplicit) {
    $BatchId = if ($env:AIC_BATCH_ID) {
        $env:AIC_BATCH_ID
    }
    else {
        $localValue = Get-LocalEnvValue -Name "AIC_BATCH_ID"
        if ($localValue) { $localValue } else { Get-YamlRuntimeDefault -Name "batch" }
    }
}
if ($RuntimeProfile -notin @("laptop", "accelerated")) {
    throw "Runtime profile must be laptop or accelerated."
}
if (-not $BatchId -or $BatchId -notmatch "^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$") {
    throw "BatchId must contain only letters, digits, '.', '_' or '-'."
}

function Assert-PortAvailable {
    param(
        [Parameter(Mandatory = $true)][int]$RequestedPort,
        [Parameter(Mandatory = $true)][string]$Phase
    )

    $probe = [System.Net.Sockets.TcpListener]::new(
        [System.Net.IPAddress]::Loopback,
        $RequestedPort
    )
    $probe.Server.ExclusiveAddressUse = $true
    try {
        $probe.Start()
        $probe.Stop()
        return
    }
    catch {
        try { $probe.Stop() } catch { }
    }

    if ($IsWindowsHost) {
        $listeners = @(Get-NetTCPConnection -State Listen -LocalPort $RequestedPort -ErrorAction SilentlyContinue)
        if ($listeners.Count -gt 0) {
            $processId = $listeners[0].OwningProcess
            $owner = Get-Process -Id $processId -ErrorAction SilentlyContinue
            $ownerName = if ($null -ne $owner) { $owner.ProcessName } else { "unknown" }
            $started = if ($null -ne $owner) { $owner.StartTime.ToUniversalTime().ToString("o") } else { "unknown" }
            throw "Port $RequestedPort is already owned by PID $processId ($ownerName, started=$started) during $Phase. Choose another -Port; no process was stopped."
        }
    }
    throw "Port $RequestedPort is already in use during $Phase. Choose another -Port; no process was stopped."
}

function Test-OllamaEndpoint {
    param([Parameter(Mandatory = $true)][int]$RequestedPort)

    try {
        $response = Invoke-WebRequest -UseBasicParsing `
            -Uri "http://127.0.0.1:$RequestedPort/api/tags" -TimeoutSec 2
        if ($response.StatusCode -ne 200) { return $false }
        $payload = $response.Content | ConvertFrom-Json
        if ($null -eq $payload.models) { return $false }
        $version = Invoke-WebRequest -UseBasicParsing `
            -Uri "http://127.0.0.1:$RequestedPort/api/version" -TimeoutSec 2
        if ($version.StatusCode -ne 200) { return $false }
        $versionPayload = $version.Content | ConvertFrom-Json
        return [string]$versionPayload.version -match "^[0-9]+\.[0-9]+\.[0-9]+"
    }
    catch { return $false }
}

function Resolve-OllamaExecutable {
    $command = Get-Command ollama -ErrorAction SilentlyContinue
    if ($null -ne $command) { return $command.Source }
    $projectCandidate = Join-Path $ProjectRoot ".tools/Ollama/ollama.exe"
    if (Test-Path -LiteralPath $projectCandidate -PathType Leaf) {
        return $projectCandidate
    }
    if ($IsWindowsHost -and $env:LOCALAPPDATA) {
        $candidate = Join-Path $env:LOCALAPPDATA "Programs/Ollama/ollama.exe"
        if (Test-Path -LiteralPath $candidate -PathType Leaf) { return $candidate }
    }
    return $null
}

function Write-OtherAicListeners {
    if (-not $IsWindowsHost) { return }
    try {
        $seen = @{}
        foreach ($listener in @(Get-NetTCPConnection -State Listen -ErrorAction Stop)) {
            $processId = $listener.OwningProcess
            if ($seen.ContainsKey($processId)) { continue }
            $process = Get-CimInstance Win32_Process -Filter "ProcessId = $processId" -ErrorAction Stop
            $commandLine = [string]$process.CommandLine
            if ($commandLine -notmatch "(?i)streamlit" -or
                $commandLine -notmatch "(?i)app[\\/]streamlit_app\.py") {
                continue
            }
            $seen[$processId] = $true
            $ports = @(
                Get-NetTCPConnection -State Listen -OwningProcess $processId -ErrorAction SilentlyContinue |
                    Select-Object -ExpandProperty LocalPort -Unique |
                    Sort-Object
            ) -join ","
            $runtimeKind = if ($commandLine -match "(?i)\.venv-runtime[\\/]Scripts[\\/]python\.exe") {
                "canonical-runtime"
            }
            else { "noncanonical-runtime" }
            $started = if ($null -ne $process.CreationDate) {
                ([DateTime]$process.CreationDate).ToUniversalTime().ToString("o")
            }
            else { "unknown" }
            Write-Warning "Another AIC Streamlit listener is active: ports=$ports pid=$processId started=$started kind=$runtimeKind. It was not stopped."
        }
    }
    catch {
        Write-Verbose "Existing Streamlit listeners could not be enumerated; launch continues."
    }
}

if (-not (Test-Path -LiteralPath $Python -PathType Leaf) -or
    -not (Test-Path -LiteralPath $StampPath -PathType Leaf)) {
    throw "Competition runtime is missing. Run .\scripts\bootstrap_runtime.ps1 -Profile $RuntimeProfile"
}
try {
    $Stamp = Get-Content -Raw -LiteralPath $StampPath -Encoding UTF8 | ConvertFrom-Json
}
catch {
    throw "Competition runtime stamp is invalid. Re-run .\scripts\bootstrap_runtime.ps1 -Profile $RuntimeProfile"
}
$LockHash = (Get-FileHash -LiteralPath (Join-Path $ProjectRoot "uv.lock") -Algorithm SHA256).Hash.ToLowerInvariant()
if ($Stamp.schema_version -ne 1 -or $Stamp.profile -ne $RuntimeProfile -or
    $Stamp.uv_lock_sha256 -ne $LockHash) {
    throw "Competition runtime is stale or has the wrong profile. Re-run .\scripts\bootstrap_runtime.ps1 -Profile $RuntimeProfile"
}
$ActualPythonVersion = (& $Python -c "import platform; print(platform.python_version())").Trim()
if ($LASTEXITCODE -ne 0 -or $ActualPythonVersion -ne $Stamp.python_version) {
    throw "Competition runtime interpreter does not match its stamp. Re-run bootstrap_runtime.ps1."
}
$SourceContractHash = (& $Python -c @'
import sys
sys.path.insert(0, sys.argv[1])
from src.runtime_contract import source_contract_fingerprint
print(source_contract_fingerprint(sys.argv[1]))
'@ $ProjectRoot).Trim()
if ($LASTEXITCODE -ne 0 -or $SourceContractHash -notmatch "^[0-9a-f]{64}$") {
    throw "Runtime source-contract fingerprint could not be computed."
}
Assert-PortAvailable -RequestedPort $Port -Phase "initial preflight"
if ($ApiPort -eq $Port -or $OllamaPort -in @($Port, $ApiPort)) {
    throw "Streamlit, FastAPI, and Ollama ports must be different."
}
Assert-PortAvailable -RequestedPort $ApiPort -Phase "API preflight"

$env:PYTHONUTF8 = "1"
$env:AIC_PROFILE = $RuntimeProfile
$env:AIC_BATCH_ID = $BatchId
$env:UV_PROJECT_ENVIRONMENT = $RuntimeEnvironment
$env:PADDLE_PDX_CACHE_HOME = Join-Path $ProjectRoot ".cache/paddlex"
$env:PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK = "True"
$env:AIC_LAUNCH_CONTRACT = "run_app-v2"
$env:AIC_LAUNCH_ID = [guid]::NewGuid().ToString()
$env:AIC_LAUNCH_PORT = [string]$Port
$env:AIC_LAUNCH_BATCH = $BatchId
$env:AIC_LAUNCH_PROFILE = $RuntimeProfile
$env:AIC_LAUNCHED_AT_UTC = [DateTime]::UtcNow.ToString("o")
$env:AIC_LAUNCH_LOCK_SHA256 = $LockHash
$env:AIC_RUNTIME_STAMP_SHA256 = (Get-FileHash -LiteralPath $StampPath -Algorithm SHA256).Hash.ToLowerInvariant()
$env:AIC_SOURCE_CONTRACT_SHA256 = $SourceContractHash
$env:AIC_API_URL = "http://127.0.0.1:$ApiPort"
$env:OLLAMA_URL = "http://127.0.0.1:$OllamaPort"
$env:OLLAMA_HOST = "127.0.0.1:$OllamaPort"
$env:OLLAMA_MODELS = Join-Path $ProjectRoot "models/ollama"
$env:OLLAMA_NO_CLOUD = "1"
if (-not $env:OLLAMA_LLM_LIBRARY) {
    $configuredLibrary = Get-LocalEnvValue -Name "OLLAMA_LLM_LIBRARY"
    # Let Ollama discover CUDA/Vulkan unless the operator explicitly selects one.
    # Forcing Vulkan can silently disable a usable CUDA device on hybrid laptops.
    if ($configuredLibrary) { $env:OLLAMA_LLM_LIBRARY = $configuredLibrary }
}

if ($IsWindowsHost) {
    # Windows PowerShell may expose both Path and PATH, which Start-Process
    # cannot copy into a child process. Normalize only this launcher process.
    $processPath = [System.Environment]::GetEnvironmentVariable("Path", "Process")
    [System.Environment]::SetEnvironmentVariable("PATH", $null, "Process")
    [System.Environment]::SetEnvironmentVariable("Path", $processPath, "Process")
}

# Enumerating every process/listener through CIM is diagnostic-only and can take
# several seconds on Windows.  The requested port is already checked twice with
# an exclusive socket, so keep the broader scan behind PowerShell's verbose flag.
if ($VerbosePreference -eq "Continue") {
    Write-OtherAicListeners
}
if ($AllowDegraded) {
    Write-Warning "-AllowDegraded is retained for compatibility; degraded search availability is now decided by the gated warm-up policy."
}

Push-Location $ProjectRoot
$ApiProcess = $null
$OllamaProcess = $null
try {
    if ($StartServices) {
        if (-not (Get-Command docker -ErrorAction SilentlyContinue)) {
            throw "Docker is unavailable; services were not started."
        }
        & docker compose --env-file .env up -d qdrant meilisearch
        if ($LASTEXITCODE -ne 0) {
            throw "docker compose failed with exit code $LASTEXITCODE."
        }
    }

    if (-not (Test-OllamaEndpoint -RequestedPort $OllamaPort)) {
        try {
            Assert-PortAvailable -RequestedPort $OllamaPort -Phase "Ollama preflight"
        }
        catch {
            throw "Ollama port $OllamaPort is occupied by a non-Ollama process."
        }
        $OllamaExecutable = Resolve-OllamaExecutable
        if (-not $OllamaExecutable) {
            throw "Ollama is not installed. Install the official Windows build before launch."
        }
        New-Item -ItemType Directory -Force -Path $env:OLLAMA_MODELS | Out-Null
        $OllamaStart = @{
            FilePath = $OllamaExecutable
            ArgumentList = @("serve")
            WorkingDirectory = $ProjectRoot
            PassThru = $true
        }
        if ($IsWindowsHost) { $OllamaStart["WindowStyle"] = "Hidden" }
        $OllamaProcess = Start-Process @OllamaStart
        foreach ($attempt in 1..30) {
            if ($OllamaProcess.HasExited) { break }
            if (Test-OllamaEndpoint -RequestedPort $OllamaPort) { break }
            Start-Sleep -Seconds 1
        }
        if (-not (Test-OllamaEndpoint -RequestedPort $OllamaPort)) {
            throw "Ollama did not become healthy within 30 seconds."
        }
    }

    # Keep launcher validation lightweight so liveness is available immediately.
    # Exact service/index/canonical-image readiness runs in the application's
    # single-flight warm-up and gates every search workflow until it completes.
    $DoctorArguments = if ($StartServices) {
        @(
            "scripts/runtime_doctor.py", "--batch-id", $BatchId,
            "--mode", "app", "--wait-seconds", "60", "--poll-seconds", "1",
            "--compact"
        )
    }
    else {
        @(
            "scripts/runtime_doctor.py", "--batch-id", $BatchId,
            "--mode", "app", "--startup-only", "--compact"
        )
    }
    if ($AllowDegraded -and $StartServices) { $DoctorArguments += "--allow-degraded" }
    & $Python @DoctorArguments
    $DoctorExit = $LASTEXITCODE
    if ($DoctorExit -ne 0) {
        throw "Runtime launch preflight is blocked. Verify the runtime and selected batch."
    }

    Assert-PortAvailable -RequestedPort $ApiPort -Phase "final API launch"
    $RunCache = Join-Path $ProjectRoot ".cache/run"
    New-Item -ItemType Directory -Force -Path $RunCache | Out-Null
    $ApiStdout = Join-Path $RunCache "fastapi-$ApiPort.out.log"
    $ApiStderr = Join-Path $RunCache "fastapi-$ApiPort.err.log"
    $ApiStart = @{
        FilePath = $Python
        ArgumentList = @("scripts/serve.py", "--host", "127.0.0.1", "--port", "$ApiPort")
        WorkingDirectory = $ProjectRoot
        PassThru = $true
        RedirectStandardOutput = $ApiStdout
        RedirectStandardError = $ApiStderr
    }
    if ($IsWindowsHost) { $ApiStart["WindowStyle"] = "Hidden" }
    $ApiProcess = Start-Process @ApiStart
    $ApiHealthy = $false
    foreach ($attempt in 1..90) {
        if ($ApiProcess.HasExited) { break }
        try {
            $health = Invoke-WebRequest -UseBasicParsing -Uri "$($env:AIC_API_URL)/healthz" -TimeoutSec 2
            if ($health.StatusCode -eq 200) { $ApiHealthy = $true; break }
        }
        catch { }
        Start-Sleep -Seconds 1
    }
    if (-not $ApiHealthy) { throw "FastAPI did not become healthy on the requested loopback port." }

    Assert-PortAvailable -RequestedPort $Port -Phase "final UI launch"
    Write-Host "Launching AIC 2026 UI on http://127.0.0.1:$Port and API on http://127.0.0.1:$ApiPort (profile=$RuntimeProfile, batch=$BatchId, launch_id=$($env:AIC_LAUNCH_ID))"
    & $Python -m streamlit run app/streamlit_app.py `
        --server.headless true `
        --server.address 127.0.0.1 `
        --server.port $Port `
        --server.fileWatcherType none `
        --server.runOnSave false `
        --browser.gatherUsageStats false
    if ($LASTEXITCODE -ne 0) {
        throw "Streamlit exited with code $LASTEXITCODE."
    }
}
finally {
    if ($null -ne $ApiProcess -and -not $ApiProcess.HasExited) {
        Stop-Process -Id $ApiProcess.Id -Force -ErrorAction SilentlyContinue
        $ApiProcess.WaitForExit(10000) | Out-Null
    }
    if ($null -ne $OllamaProcess -and -not $OllamaProcess.HasExited) {
        Stop-Process -Id $OllamaProcess.Id -Force -ErrorAction SilentlyContinue
        $OllamaProcess.WaitForExit(10000) | Out-Null
    }
    Pop-Location
}
