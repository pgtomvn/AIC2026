[CmdletBinding()]
param(
    [switch]$SkipArchiveHash
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$IsWindowsHost = $env:OS -eq "Windows_NT"
$UvBinaryName = if ($IsWindowsHost) { "uv.exe" } else { "uv" }

function Test-LocalPort {
    param([int]$Port)

    $client = New-Object Net.Sockets.TcpClient
    try {
        $connection = $client.BeginConnect("127.0.0.1", $Port, $null, $null)
        if (-not $connection.AsyncWaitHandle.WaitOne(300)) { return $false }
        $client.EndConnect($connection)
        return $true
    }
    catch { return $false }
    finally { $client.Dispose() }
}

function Test-PathQuietly {
    param([Parameter(Mandatory = $true)][string]$Path)

    try { return Test-Path -LiteralPath $Path -ErrorAction Stop }
    catch { return $false }
}

Write-Host "AIC 2026 preflight"
Write-Host "Project: $ProjectRoot"
Write-Host "PowerShell: $($PSVersionTable.PSVersion) ($($PSVersionTable.PSEdition))"
Write-Host "OS: $([Environment]::OSVersion.VersionString)"

$drive = New-Object IO.DriveInfo([IO.Path]::GetPathRoot($ProjectRoot))
Write-Host ("Workspace drive free: {0:N1} GiB" -f ($drive.AvailableFreeSpace / 1GB))

foreach ($commandName in @("git", "docker", "nvidia-smi")) {
    $command = Get-Command $commandName -ErrorAction SilentlyContinue
    if ($command) { Write-Host "${commandName}: $($command.Source)" }
    else { Write-Warning "$commandName is not available on PATH." }
}
if (Get-Command git -ErrorAction SilentlyContinue) {
    Write-Host "Git version: $(& git --version)"
}
if (Get-Command docker -ErrorAction SilentlyContinue) {
    Write-Host "Docker CLI: $(& docker --version)"
    Write-Host "Docker Compose: $(& docker compose version)"
}
if (Get-Command nvidia-smi -ErrorAction SilentlyContinue) {
    Write-Host "GPU(s):"
    & nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader |
        ForEach-Object { Write-Host "  $_" }
}

$uvExe = Join-Path $ProjectRoot ".tools/uv/0.11.16/$UvBinaryName"
if (Test-Path -LiteralPath $uvExe -PathType Leaf) {
    Write-Host "uv: $(& $uvExe --version) at $uvExe"
}
else {
    Write-Warning "Project-local uv is absent; run scripts/bootstrap.ps1."
}

$missingPathEntries = @(
    $env:Path -split [regex]::Escape([IO.Path]::PathSeparator) |
        ForEach-Object { [Environment]::ExpandEnvironmentVariables($_.Trim('"')) } |
        Where-Object { $_ -and -not (Test-PathQuietly -Path $_) } |
        Select-Object -Unique
)
if ($missingPathEntries.Count -gt 0) {
    Write-Warning "PATH contains missing locations (not modified by this script):"
    $missingPathEntries | Select-Object -First 10 | ForEach-Object { Write-Host "  $_" }
}

foreach ($port in @(6333, 6334, 7700, 8501)) {
    Write-Host "127.0.0.1:$port open: $(Test-LocalPort -Port $port)"
}

$manifestPath = Join-Path $ProjectRoot "checksums/archives.sha256"
if (-not (Test-Path -LiteralPath $manifestPath -PathType Leaf)) {
    throw "Archive checksum manifest is missing: $manifestPath"
}

foreach ($line in Get-Content -LiteralPath $manifestPath -Encoding UTF8) {
    if ($line -match "^\s*#" -or [string]::IsNullOrWhiteSpace($line)) { continue }
    if ($line -notmatch "^([a-fA-F0-9]{64})\s+(.+)$") {
        throw "Malformed checksum manifest line: $line"
    }
    $expected = $Matches[1].ToLowerInvariant()
    $archiveName = $Matches[2].Trim().TrimStart("*")
    if ([IO.Path]::GetFileName($archiveName) -ne $archiveName) {
        throw "Archive manifest entries must be root filenames: $archiveName"
    }
    $archivePath = Join-Path $ProjectRoot $archiveName
    if (-not (Test-Path -LiteralPath $archivePath -PathType Leaf)) {
        Write-Warning "Archive is absent: $archiveName"
        continue
    }
    if ($SkipArchiveHash) {
        Write-Host "Archive present (hash skipped): $archiveName"
        continue
    }
    Write-Host "Hashing immutable archive: $archiveName"
    $actual = (Get-FileHash -LiteralPath $archivePath -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($actual -ne $expected) { throw "Archive checksum mismatch: $archiveName" }
    Write-Host "Archive checksum OK: $archiveName"
}

Write-Host "Preflight complete; no state was changed."
