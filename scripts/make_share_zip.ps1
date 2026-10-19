<#
.SYNOPSIS
  Packs the project into one clean zip that can be shared (for example with the Track A colleague).

.DESCRIPTION
  Excludes everything machine-specific or private: the virtual environment, .env (it can hold the
  Neo4j password), caches, version-control data, build output, the bundled LaTeX tool and the
  university-report source files. The recipient unzips and runs  .\run_demo.ps1

.EXAMPLE
  .\scripts\make_share_zip.ps1
  .\scripts\make_share_zip.ps1 -Output C:\temp\axiom_demo.zip
#>
param([string]$Output = "")

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$name = "AXIOM_TrackB_demo_" + (Get-Date -Format "yyyyMMdd")
if (-not $Output) { $Output = Join-Path (Split-Path -Parent $root) "$name.zip" }

$stage = Join-Path ([System.IO.Path]::GetTempPath()) ("axiom_share_" + [guid]::NewGuid().ToString("N"))
$target = Join-Path $stage "project_trackB"
New-Item -ItemType Directory -Force $target | Out-Null

$excludeDirs = @(".venv", ".git", "__pycache__", ".pytest_cache", ".tools", "build", ".idea", ".vscode")
$excludeFiles = @(".env", "*.pyc", "*.docx", "*.doc", "build_log.txt", "*.aux", "*.log")

& robocopy $root $target /E /NFL /NDL /NJH /NJS /NP /XD @excludeDirs /XF @excludeFiles | Out-Null
if ($LASTEXITCODE -ge 8) { throw "robocopy failed with code $LASTEXITCODE" }

# Remove what the recipient does not need: the university report (it embeds a bundled LaTeX tool
# and personal material); keep the final report and the design records.
$univ = Join-Path $target "rapport\report_latex"
if (Test-Path $univ) { Remove-Item $univ -Recurse -Force }
Get-ChildItem (Join-Path $target "rapport") -File -ErrorAction SilentlyContinue |
  Where-Object { $_.Extension -in ".pdf", ".png", ".doc", ".docx" } | Remove-Item -Force

if (Test-Path $Output) { Remove-Item $Output -Force }
Compress-Archive -Path $target -DestinationPath $Output -CompressionLevel Optimal
Remove-Item $stage -Recurse -Force

$size = [math]::Round((Get-Item $Output).Length / 1MB, 1)
Write-Host "Created $Output ($size MB)"
Write-Host "Recipient: unzip, open PowerShell in project_trackB, run  .\run_demo.ps1"
