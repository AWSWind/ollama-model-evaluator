<#
.SYNOPSIS
    Ollama Model Evaluator - one-button project launcher for Windows.

.DESCRIPTION
    Brings up the full stack with a single command:

      1. Ollama server (if installed and not already running).
      2. The backend (FastAPI on port 8765 by default).
      3. The React UI bundle mounted by the backend at ``/``.

    Missing dependencies (``.venv``, ``ui\dist``) trigger an automatic
    call to ``scripts\install.ps1 -SkipTests``, so a fresh clone can
    go from zero to running with this one script.

    PID files land under ``.run\``; logs under ``logs\``.

.PARAMETER Config
    Path to the backend config file. Default ``examples\config.qwen.yaml``.

.PARAMETER BackendHost
    Bind address. Default ``0.0.0.0`` (reachable from the LAN).
    Use ``127.0.0.1`` to expose only on the loopback.

.PARAMETER BackendPort
    Backend TCP port. Default 8765.

.PARAMETER OllamaPort
    Ollama TCP port. Default 11434.

.PARAMETER UiDevPort
    Vite dev-server port (used with -Dev). Default 5173.

.PARAMETER Dev
    Also start the Vite dev server with HMR on ``UiDevPort``.

.PARAMETER Detach
    Return immediately after services are ready instead of tailing the
    backend log.

.PARAMETER SkipOllama
    Skip the Ollama liveness check / auto-start.

.PARAMETER NoInstall
    Refuse to run ``install.ps1`` even when dependencies are missing.

.PARAMETER LogLevel
    Backend ``--log-level``. One of ``debug``, ``info``, ``warn``,
    ``error``. Default ``info``.

.EXAMPLE
    # Default production start, foreground
    .\scripts\start.ps1

.EXAMPLE
    # Background launch on an alternate port
    .\scripts\start.ps1 -BackendPort 9000 -Detach

.EXAMPLE
    # Dev mode: also runs Vite on :5173 with HMR
    .\scripts\start.ps1 -Dev
#>

[CmdletBinding()]
param(
  [string]$Config = "examples\config.qwen.yaml",
  [string]$BackendHost = "0.0.0.0",
  [int]$BackendPort = 8765,
  [int]$OllamaPort = 11434,
  [int]$UiDevPort = 5173,
  [switch]$Dev,
  [switch]$Detach,
  [switch]$SkipOllama,
  [switch]$NoInstall,
  [string]$LogLevel = "info"
)

# Let native-command failures propagate via ``$LASTEXITCODE`` rather than
# via terminating errors. Mirrors the install.ps1 convention.
$ErrorActionPreference = "Continue"

$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $RepoRoot

$RunDir = Join-Path $RepoRoot ".run"
$LogDir = Join-Path $RepoRoot "logs"
New-Item -ItemType Directory -Force -Path $RunDir, $LogDir | Out-Null

$BackendPidFile = Join-Path $RunDir "backend.pid"
$OllamaPidFile  = Join-Path $RunDir "ollama.pid"
$VitePidFile    = Join-Path $RunDir "vite.pid"

$BackendLog = Join-Path $LogDir "backend.log"
$OllamaLog  = Join-Path $LogDir "ollama.log"
$ViteLog    = Join-Path $LogDir "vite.log"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

function Write-Section { param([string]$m)
  Write-Host ""
  Write-Host ("== " + $m) -ForegroundColor Blue
}
function Write-Msg  { param([string]$m) Write-Host (">> " + $m) -ForegroundColor Cyan }
function Write-Ok   { param([string]$m) Write-Host ("OK " + $m) -ForegroundColor Green }
function Write-Warn { param([string]$m) Write-Host ("!! " + $m) -ForegroundColor Yellow }
function Fail       { param([string]$m) Write-Host ("** " + $m) -ForegroundColor Red; exit 1 }

function Test-PidAlive {
  # Returns $true iff the PID stored in the given file points at a live process.
  # Handles empty / whitespace-only files and stale PIDs gracefully so callers
  # can use this as a simple "is-running" gate.
  param([string]$PidFile)
  if (-not (Test-Path $PidFile)) { return $false }
  $raw = (Get-Content $PidFile -ErrorAction SilentlyContinue | Select-Object -First 1)
  if ([string]::IsNullOrWhiteSpace($raw)) { return $false }
  $pidNum = 0
  if (-not [int]::TryParse($raw.Trim(), [ref]$pidNum)) { return $false }
  return [bool](Get-Process -Id $pidNum -ErrorAction SilentlyContinue)
}

function Get-PidFromFile {
  # Reads the first line of a PID file and returns it as an int, or $null
  # when the file is missing, empty, or unparseable. Centralising the
  # null-safety dance here keeps callers readable on Windows PowerShell
  # 5.1, which does not support the ``??`` null-coalescing operator.
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

function Test-PortInUse {
  # Returns $true iff something is already listening on the given local port.
  # Uses ``Get-NetTCPConnection`` where available (Windows 8+), falls back to
  # a raw TcpClient connect attempt for minimal images that lack the cmdlet.
  param([int]$Port)
  try {
    $null = Get-NetTCPConnection -State Listen -LocalPort $Port -ErrorAction Stop
    return $true
  } catch {
    try {
      $c = New-Object System.Net.Sockets.TcpClient
      $iar = $c.BeginConnect("127.0.0.1", $Port, $null, $null)
      $ok  = $iar.AsyncWaitHandle.WaitOne(500)
      if ($ok -and $c.Connected) { $c.Close(); return $true }
      $c.Close()
      return $false
    } catch { return $false }
  }
}

function Wait-Http {
  # Polls ``$Url`` until it returns 2xx/3xx or the deadline passes.
  # Uses ``127.0.0.1`` instead of ``localhost`` so Windows' IPv6 resolution
  # (which can take seconds on some machines) does not eat the timeout
  # budget. Keeps tight 250 ms retries so a slow-but-eventually-ready
  # service is still picked up well within the timeout.
  param([string]$Url, [int]$TimeoutSeconds = 30, [string]$Label = "service")
  $normalisedUrl = $Url -replace '://localhost', '://127.0.0.1'
  $deadline = (Get-Date).AddSeconds($TimeoutSeconds)
  while ((Get-Date) -lt $deadline) {
    try {
      $r = Invoke-WebRequest -Uri $normalisedUrl -UseBasicParsing -TimeoutSec 2 -Method Get -ErrorAction Stop
      if ($r.StatusCode -lt 400) { return $true }
    } catch { Start-Sleep -Milliseconds 250 }
  }
  return $false
}

function Stop-PidFile {
  # Terminate the process named in ``$PidFile`` and remove the file.
  # Tries a graceful ``Stop-Process`` first, then ``-Force`` if the process
  # outlives the grace period.
  param([string]$PidFile)
  if (-not (Test-Path $PidFile)) { return }
  $pidNum = Get-PidFromFile $PidFile
  if ($pidNum) {
    $proc = Get-Process -Id $pidNum -ErrorAction SilentlyContinue
    if ($proc) {
      try { Stop-Process -Id $pidNum -ErrorAction Stop } catch {}
      $end = (Get-Date).AddSeconds(5)
      while ((Get-Date) -lt $end) {
        if (-not (Get-Process -Id $pidNum -ErrorAction SilentlyContinue)) { break }
        Start-Sleep -Milliseconds 250
      }
      if (Get-Process -Id $pidNum -ErrorAction SilentlyContinue) {
        try { Stop-Process -Id $pidNum -Force -ErrorAction SilentlyContinue } catch {}
      }
    }
  }
  Remove-Item $PidFile -Force -ErrorAction SilentlyContinue
}

function Start-Detached {
  # Start a process fully detached from the current console so Ctrl-C in
  # PowerShell does not kill the background service. Writes stdout + stderr
  # to ``$LogPath`` and returns the spawned Process object.
  param(
    [Parameter(Mandatory)][string]$File,
    [Parameter(Mandatory)][string[]]$ArgumentList,
    [Parameter(Mandatory)][string]$LogPath,
    [string]$WorkingDirectory = $PWD
  )
  # Ensure the log file exists so Start-Process can redirect into it even
  # when the parent directory was just created.
  if (-not (Test-Path $LogPath)) { New-Item -ItemType File -Path $LogPath -Force | Out-Null }
  $proc = Start-Process -FilePath $File `
    -ArgumentList $ArgumentList `
    -WorkingDirectory $WorkingDirectory `
    -RedirectStandardOutput $LogPath `
    -RedirectStandardError  "$LogPath.err" `
    -WindowStyle Hidden `
    -PassThru
  return $proc
}

# ---------------------------------------------------------------------------
# 1. Dependency preflight
# ---------------------------------------------------------------------------

Write-Section "Preflight"

$missing = @()
$venvPython = Join-Path $RepoRoot ".venv\Scripts\python.exe"
$uiDist = Join-Path $RepoRoot "ui\dist\index.html"

if (-not (Test-Path $venvPython)) { $missing += "Python venv (.venv\)" }
if (-not (Test-Path $uiDist))     { $missing += "UI bundle (ui\dist\)" }

if ($missing.Count -gt 0) {
  Write-Warn ("Missing dependencies: " + ($missing -join ", "))
  if (-not $NoInstall) {
    Write-Msg "Running scripts\install.ps1 -SkipTests to set them up"
    & powershell -NoProfile -ExecutionPolicy Bypass -File (Join-Path $PSScriptRoot "install.ps1") -SkipTests
    if ($LASTEXITCODE -ne 0) { Fail "install.ps1 failed with status $LASTEXITCODE" }
  } else {
    Fail "-NoInstall was given; run scripts\install.ps1 manually first"
  }
}
Write-Ok "Python venv ready"
Write-Ok "UI bundle ready"

$PY = $venvPython

# ---------------------------------------------------------------------------
# 2. Ollama
# ---------------------------------------------------------------------------

$StartedOllama = $false

if ($SkipOllama) {
  Write-Warn "Skipping Ollama check (-SkipOllama)"
} else {
  Write-Section "Checking Ollama"
  if (Wait-Http -Url "http://localhost:$OllamaPort/api/version" -TimeoutSeconds 2 -Label "Ollama") {
    Write-Ok "Ollama already running on :$OllamaPort"
  } else {
    $ollama = Get-Command ollama -ErrorAction SilentlyContinue
    if ($null -eq $ollama) {
      Fail ("Ollama is not running on :$OllamaPort and the 'ollama' CLI is not installed.`n" +
            "Install from https://ollama.com/download, or pass -SkipOllama to continue without it.")
    }
    Write-Msg "Ollama not reachable; starting 'ollama serve' in the background"
    $oProc = Start-Detached -File $ollama.Source -ArgumentList @("serve") -LogPath $OllamaLog
    $oProc.Id | Out-File -FilePath $OllamaPidFile -Encoding ascii -Force
    $StartedOllama = $true
    if (-not (Wait-Http -Url "http://localhost:$OllamaPort/api/version" -TimeoutSeconds 30 -Label "Ollama")) {
      Fail "Ollama failed to start within 30s. See $OllamaLog."
    }
    Write-Ok ("Ollama up on :$OllamaPort (pid " + $oProc.Id + ")")
  }
}

# ---------------------------------------------------------------------------
# 3. Backend
# ---------------------------------------------------------------------------

Write-Section "Starting backend"

if (Test-PidAlive $BackendPidFile) {
  $existing = (Get-Content $BackendPidFile | Select-Object -First 1)
  Fail "Backend already running with PID $existing. Run scripts\stop.ps1 first."
}
if (Test-PortInUse $BackendPort) {
  Fail "Port $BackendPort is already bound. Choose another with -BackendPort."
}
if (-not (Test-Path (Join-Path $RepoRoot $Config))) {
  Fail "Config file not found: $Config"
}

# ``OLLAMA_EVAL_UI_DIR`` tells the backend where the built UI lives.
$env:OLLAMA_EVAL_UI_DIR = (Join-Path $RepoRoot "ui\dist")

$backendArgs = @(
  "-m", "ollama_evaluator.cli",
  "--config", $Config,
  "--log-level", $LogLevel,
  "serve",
  "--host", $BackendHost,
  "--port", "$BackendPort"
)
$bProc = Start-Detached -File $PY -ArgumentList $backendArgs -LogPath $BackendLog
$bProc.Id | Out-File -FilePath $BackendPidFile -Encoding ascii -Force

if (-not (Wait-Http -Url "http://localhost:$BackendPort/api/health" -TimeoutSeconds 30 -Label "Backend")) {
  Write-Warn ("Backend did not respond within 30s. Last 30 lines of " + $BackendLog + ":")
  Get-Content $BackendLog -Tail 30 -ErrorAction SilentlyContinue
  Stop-PidFile $BackendPidFile
  Fail "Aborting"
}
Write-Ok ("Backend up on http://${BackendHost}:${BackendPort} (pid " + $bProc.Id + ")")

# ---------------------------------------------------------------------------
# 4. Vite dev server (optional)
# ---------------------------------------------------------------------------

if ($Dev) {
  Write-Section "Starting Vite dev server"
  if (Test-PidAlive $VitePidFile) {
    Fail "Vite dev server already running. Run scripts\stop.ps1 first."
  }
  if (Test-PortInUse $UiDevPort) {
    Fail "Port $UiDevPort already in use."
  }
  $npm = Get-Command npm -ErrorAction SilentlyContinue
  if ($null -eq $npm) { Fail "npm is not installed; cannot start Vite dev server." }

  # Launch ``npm run dev`` with a fixed host/port so ``Wait-Http`` hits the
  # right socket. The working directory is ``ui\`` so Node picks up the
  # right ``package.json`` and lockfile.
  $vProc = Start-Detached `
    -File $npm.Source `
    -ArgumentList @("run","dev","--","--host","0.0.0.0","--port","$UiDevPort") `
    -LogPath $ViteLog `
    -WorkingDirectory (Join-Path $RepoRoot "ui")
  $vProc.Id | Out-File -FilePath $VitePidFile -Encoding ascii -Force
  if (-not (Wait-Http -Url "http://localhost:$UiDevPort/" -TimeoutSeconds 30 -Label "Vite")) {
    Write-Warn "Vite did not respond at :$UiDevPort within 30s; see $ViteLog"
  } else {
    Write-Ok ("Vite dev server up on http://localhost:$UiDevPort (pid " + $vProc.Id + ")")
  }
}

# ---------------------------------------------------------------------------
# 5. Ready banner
# ---------------------------------------------------------------------------

$hostForUser = $BackendHost
if ($hostForUser -eq "0.0.0.0") { $hostForUser = "localhost" }

Write-Host ""
Write-Host "All services ready." -ForegroundColor Green
Write-Host ""
Write-Host "  Web UI           http://${hostForUser}:${BackendPort}/"
Write-Host "  Health probe     http://${hostForUser}:${BackendPort}/api/health"
Write-Host "  REST API docs    http://${hostForUser}:${BackendPort}/openapi.json"
if ($Dev) {
  Write-Host "  Vite dev server  http://localhost:${UiDevPort}/"
}
Write-Host "  Backend log      $BackendLog"
if ($StartedOllama) {
  Write-Host "  Ollama log       $OllamaLog"
}
Write-Host ""
Write-Host "  PID files in $RunDir"
Write-Host ""
Write-Host "Stop everything:"
Write-Host "  .\scripts\stop.ps1"
Write-Host ""

# ---------------------------------------------------------------------------
# 6. Detach vs foreground
# ---------------------------------------------------------------------------

if ($Detach) {
  Write-Ok "Detached. Services continue running in the background."
  exit 0
}

Write-Msg "Tailing $BackendLog - press Ctrl-C to stop"

# Register a Ctrl-C / exit handler so background services come down cleanly
# when the user hits Ctrl-C or closes the window. ``Register-EngineEvent``
# on ``PowerShell.Exiting`` covers the window-close case.
$cleanupScript = {
  param($BackendPidFile, $VitePidFile, $OllamaPidFile, $StartedOllama, $RunDev)

  function _GetPid([string]$PidFile) {
    if (-not (Test-Path $PidFile)) { return $null }
    $raw = Get-Content $PidFile -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($null -eq $raw) { return $null }
    $t = ([string]$raw).Trim()
    if ([string]::IsNullOrWhiteSpace($t)) { return $null }
    $n = 0
    if ([int]::TryParse($t, [ref]$n)) { return $n }
    return $null
  }

  if (Test-Path $BackendPidFile) {
    Write-Host ">> Stopping backend" -ForegroundColor Cyan
    $pidNum = _GetPid $BackendPidFile
    if ($pidNum) { Stop-Process -Id $pidNum -ErrorAction SilentlyContinue }
    Remove-Item $BackendPidFile -Force -ErrorAction SilentlyContinue
  }
  if ($RunDev -and (Test-Path $VitePidFile)) {
    Write-Host ">> Stopping Vite" -ForegroundColor Cyan
    $pidNum = _GetPid $VitePidFile
    if ($pidNum) { Stop-Process -Id $pidNum -ErrorAction SilentlyContinue }
    Remove-Item $VitePidFile -Force -ErrorAction SilentlyContinue
  }
  if ($StartedOllama -and (Test-Path $OllamaPidFile)) {
    Write-Host ">> Stopping Ollama" -ForegroundColor Cyan
    $pidNum = _GetPid $OllamaPidFile
    if ($pidNum) { Stop-Process -Id $pidNum -ErrorAction SilentlyContinue }
    Remove-Item $OllamaPidFile -Force -ErrorAction SilentlyContinue
  }
}

Register-EngineEvent PowerShell.Exiting -Action {
  & $cleanupScript $BackendPidFile $VitePidFile $OllamaPidFile $StartedOllama $Dev.IsPresent
} | Out-Null

try {
  Get-Content -Path $BackendLog -Wait -Tail 20
} finally {
  & $cleanupScript $BackendPidFile $VitePidFile $OllamaPidFile $StartedOllama $Dev.IsPresent
}

exit 0
