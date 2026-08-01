[CmdletBinding()]
param(
    [switch]$PurgePackageCache
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$TrimChars = [char[]]@("\", "/")
$NormalizedRoot = [IO.Path]::GetFullPath($ProjectRoot).TrimEnd($TrimChars)
$RootPrefix = $NormalizedRoot + [IO.Path]::DirectorySeparatorChar

function Assert-NoReparsePointInProjectPath {
    param([Parameter(Mandatory = $true)][string]$Target)

    $relativeTarget = $Target.Substring($RootPrefix.Length)
    $currentPath = $NormalizedRoot
    $pathsToCheck = @($currentPath)

    foreach ($segment in $relativeTarget.Split(
        [IO.Path]::DirectorySeparatorChar,
        [StringSplitOptions]::RemoveEmptyEntries
    )) {
        $currentPath = Join-Path $currentPath $segment
        $pathsToCheck += $currentPath
    }

    foreach ($pathToCheck in $pathsToCheck) {
        if (-not (Test-Path -LiteralPath $pathToCheck)) {
            continue
        }

        $item = Get-Item -LiteralPath $pathToCheck -Force
        if ($item.Attributes -band [IO.FileAttributes]::ReparsePoint) {
            throw "Refusing to remove through a linked project path: $pathToCheck"
        }
    }
}

function Remove-ProjectPath {
    param([Parameter(Mandatory = $true)][string]$RelativePath)

    $target = [IO.Path]::GetFullPath((Join-Path $ProjectRoot $RelativePath))
    if (-not $target.StartsWith($RootPrefix, [StringComparison]::OrdinalIgnoreCase)) {
        throw "Refusing to remove a path outside the project: $target"
    }
    Assert-NoReparsePointInProjectPath -Target $target
    if (Test-Path -LiteralPath $target) {
        Write-Host "Removing disposable state: $RelativePath"
        Remove-Item -LiteralPath $target -Recurse -Force
    }
}

foreach ($relativePath in @(
    ".venv", ".cache/check", ".pytest_cache", ".ruff_cache", ".mypy_cache",
    ".coverage", "coverage.xml", "htmlcov"
)) {
    Remove-ProjectPath -RelativePath $relativePath
}
if ($PurgePackageCache) {
    Remove-ProjectPath -RelativePath ".cache/uv"
}

$cacheDirectories = foreach ($sourceRoot in @("app", "scripts", "src", "tests")) {
    $sourcePath = Join-Path $ProjectRoot $sourceRoot
    if (Test-Path -LiteralPath $sourcePath -PathType Container) {
        Get-ChildItem -LiteralPath $sourcePath -Directory -Recurse -Force |
            Where-Object {
                $_.Name -eq "__pycache__" -and
                -not ($_.Attributes -band [IO.FileAttributes]::ReparsePoint)
            }
    }
}
foreach ($cacheDirectory in $cacheDirectories) {
    $relative = $cacheDirectory.FullName.Substring($NormalizedRoot.Length).TrimStart($TrimChars)
    Remove-ProjectPath -RelativePath $relative
}

& (Join-Path $PSScriptRoot "bootstrap.ps1")
if ($LASTEXITCODE -ne 0) { throw "Environment bootstrap failed after cleanup." }

Write-Host "Environment reset complete. Archives, .env, data, indexes, models, and outputs were preserved."
