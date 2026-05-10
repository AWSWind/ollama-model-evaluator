<#
.SYNOPSIS
    Stop every process started by scripts\start.ps1.

.DESCRIPTION
    Reads PID files from ``.run\`` and terminates each process cleanly.
    Mirrors ``scripts/stop.sh``.

    Pass ``-StopOllama`` to terminate an Ollama we did not start ourselves.
    Pass ``-Force`` to skip the grace period and go straight to a hard stop.

.EXAMPLE
    .\scripts\stop.ps1
    .\scripts\stop.ps1 -StopOllama
    .\scripts\stop.ps1 -Force
#>

[CmdletBinding()]
param(
  [switch]$StopOllama,
  [switch]$Force
)

$ErrorActionPreference = "Continue"

$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$RunDir   = Join-Path $RepoRoot ".run"

function Write-Msg  { param([string]$m) Write-Host (">> " + $m) -ForegroundColor Cyan }
function Write-Ok   { param([string]$m) Write-Host ("OK " + $m) -ForegroundColor Green }
function Write-Warn { param([string]$m) Write-Host ("!! " + $m) -ForegroundColor Yellow }

function Get-PidFromFile {
  param([string]$PidFile)
  if (-not (Test-Path $PidFile)) { return $null }
  $raw = Get-Content $PidFile -ErrorAction SilentlyContinue | Select-Object -First 1
  if ($null -eq $raw) { return $null }
  $trimmed = ([string]$raw).Trim()
  if ([string]::IsNullOrWhiteSpace($trimmed)) { return $null }
  $pidNum = 0
  if ([int]::TryParse($trimmed, [ref]$pidNum)) { return $pidNum }
  return $null
}

function Stop-PidFile {
  # Stop the process whose PID is stored in ``$PidFile``. Optionally waits
  # for a graceful exit before escalating to ``-Force``.
  param([string]$PidFile, [bool]$Hard = $false)
  if (-not (Test-Path $PidFile)) { return }
  $pidNum = Get-PidFromFile $PidFile
  if ($pidNum) {
    $proc = Get-Process -Id $pidNum -ErrorAction SilentlyContinue
    if ($proc) {
      Write-Msg ("Stopping $(Split-Path -Leaf $PidFile) (pid $pidNum)")
      if ($Hard) {
        Stop-Process -Id $pidNum -Force -ErrorAction SilentlyContinue
      } else {
        Stop-Process -Id $pidNum -ErrorAction SilentlyContinue
        $end = (Get-Date).AddSeconds(5)
        while ((Get-Date) -lt $end) {
          if (-not (Get-Process -Id $pidNum -ErrorAction SilentlyContinue)) { break }
          Start-Sleep -Milliseconds 250
        }
        if (Get-Process -Id $pidNum -ErrorAction SilentlyContinue) {
          Write-Warn "Process $pidNum did not exit; escalating to -Force"
          Stop-Process -Id $pidNum -Force -ErrorAction SilentlyContinue
        }
      }
      Write-Ok "$(Split-Path -Leaf $PidFile) stopped"
    }
  }
  Remove-Item $PidFile -Force -ErrorAction SilentlyContinue
}

$backendPid = Join-Path $RunDir "backend.pid"
$vitePid    = Join-Path $RunDir "vite.pid"
$ollamaPid  = Join-Path $RunDir "ollama.pid"

Stop-PidFile -PidFile $backendPid -Hard:$Force.IsPresent
Stop-PidFile -PidFile $vitePid    -Hard:$Force.IsPresent
# ollama.pid exists only when start.ps1 started the daemon itself; stopping
# that one is the correct default. -StopOllama below also stops Ollama we did
# not start.
Stop-PidFile -PidFile $ollamaPid  -Hard:$Force.IsPresent

if ($StopOllama) {
  # Fall back to process-name matching for an Ollama we did not start.
  $procs = Get-Process -Name "ollama" -ErrorAction SilentlyContinue
  if ($procs) {
    Write-Warn "Stopping 'ollama' process(es) we did not start"
    $procs | ForEach-Object {
      if ($Force) {
        Stop-Process -Id $_.Id -Force -ErrorAction SilentlyContinue
      } else {
        Stop-Process -Id $_.Id -ErrorAction SilentlyContinue
      }
    }
  }
}

# Belt-and-braces fallback: kill any lingering backend that escaped our PID
# tracking (e.g. a previous run that crashed without writing the file).
$stray = Get-CimInstance Win32_Process -Filter "Name = 'python.exe' OR Name = 'py.exe'" |
         Where-Object { $_.CommandLine -match "ollama_evaluator.cli.*serve" }
if ($stray) {
  Write-Warn "Found lingering ollama_evaluator.cli serve processes; terminating"
  $stray | ForEach-Object {
    try { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue } catch {}
  }
}

Write-Ok "Stop complete."
