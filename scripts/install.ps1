<#
.SYNOPSIS
    Ollama Model Evaluator - one-button local install for Windows.

.DESCRIPTION
    Windows PowerShell twin of ``scripts/install.sh``. Runs the same
    install flow using native Windows conventions:

      1. Checks prerequisites (Python 3.11+, Node 18+, npm).
      2. Creates a Python virtual environment under .\.venv
         (unless -NoVenv is passed).
      3. Installs the backend in editable mode with dev extras.
      4. Installs UI dependencies and builds ui/dist/ (unless -SkipUI).
      5. Regenerates the shared OpenAPI / JSON Schema artefacts.
      6. Runs a short smoke test (CLI help + pytest tests/unit)
         unless -SkipTests.
      7. Prints next-step instructions.

    Idempotent - safe to re-run after ``git pull`` to pick up new
    dependencies.

.PARAMETER SkipTests
    Skip the post-install smoke test (faster).

.PARAMETER SkipUI
    Skip the UI build (Python backend only).

.PARAMETER NoVenv
    Install into the system Python instead of .\.venv.

.PARAMETER Python
    Path to a specific python.exe to use.

.EXAMPLE
    .\scripts\install.ps1

.EXAMPLE
    .\scripts\install.ps1 -SkipUI -SkipTests
#>

[CmdletBinding()]
param(
  [switch]$SkipTests,
  [switch]$SkipUI,
  [switch]$NoVenv,
  [string]$Python = ""
)

# Stop on ``throw`` / cmdlet errors, but let native commands report failure
# via ``$LASTEXITCODE`` rather than having PowerShell promote their stderr
# writes to terminating errors. This keeps us from aborting on harmless
# ``PydanticJsonSchemaWarning`` or ``npm WARN`` lines.
$ErrorActionPreference = "Continue"

function Invoke-Step {
  <#
  .SYNOPSIS
      Invoke a native command and abort when the exit code is non-zero.
  .DESCRIPTION
      Thin wrapper used in place of raw ``& ...`` so we can consistently
      check ``$LASTEXITCODE`` without sprinkling the check at every call
      site. Pass the command path as ``-Exe``, a human label as ``-Label``,
      and the remaining arguments as ``-Args`` (an array). Keeping the
      arguments in a named array sidesteps PowerShell's prefix-matching
      of leading-dash arguments against our own parameter names.
  #>
  param(
    [Parameter(Mandatory)][string]$Label,
    [Parameter(Mandatory)][string]$Exe,
    [Parameter(Mandatory)][string[]]$ExeArgs
  )
  & $Exe @ExeArgs
  if ($LASTEXITCODE -ne 0) {
    Fail "$Label failed with exit code $LASTEXITCODE"
  }
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
# Plain ASCII glyphs only -- Unicode check / cross / arrow symbols get
# mis-encoded when the script is saved with any non-UTF-8 code page or
# transferred through a transport that does not preserve BOM. ASCII keeps
# every Windows shell happy.

$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path

function Write-Section { param([string]$msg)
  Write-Host ""
  Write-Host ("== " + $msg) -ForegroundColor Blue
}
function Write-Msg  { param([string]$msg) Write-Host (">> " + $msg) -ForegroundColor Cyan }
function Write-Ok   { param([string]$msg) Write-Host ("OK " + $msg) -ForegroundColor Green }
function Write-Warn { param([string]$msg) Write-Host ("!! " + $msg) -ForegroundColor Yellow }
function Fail       { param([string]$msg) Write-Host ("** " + $msg) -ForegroundColor Red; exit 1 }

# ---------------------------------------------------------------------------
# 1. Prerequisite checks
# ---------------------------------------------------------------------------

Write-Section "Checking prerequisites"

# Python
if ([string]::IsNullOrWhiteSpace($Python)) {
  $candidates = @("python3.13","python3.12","python3.11","python3","python","py")
  foreach ($c in $candidates) {
    if (Get-Command $c -ErrorAction SilentlyContinue) {
      $Python = (Get-Command $c).Source
      break
    }
  }
}
if ([string]::IsNullOrWhiteSpace($Python)) {
  Fail ("Python 3.11+ not found. Install from https://www.python.org/downloads/ " +
        "or pass -Python path-to-python.exe")
}

$pyVersion = & $Python -c "import sys; print(f'{sys.version_info[0]}.{sys.version_info[1]}')"
$parts = $pyVersion.Split(".")
$pyMajor = [int]$parts[0]
$pyMinor = [int]$parts[1]
if ($pyMajor -lt 3 -or ($pyMajor -eq 3 -and $pyMinor -lt 11)) {
  Fail "Python 3.11+ required; found $pyVersion at $Python"
}
Write-Ok "Python $pyVersion at $Python"

# Node + npm (unless -SkipUI)
if (-not $SkipUI) {
  if (-not (Get-Command node -ErrorAction SilentlyContinue)) {
    Fail "Node.js 18+ required to build the UI. Install from https://nodejs.org/ or re-run with -SkipUI."
  }
  if (-not (Get-Command npm -ErrorAction SilentlyContinue)) {
    Fail "npm required to build the UI. Ships with Node.js."
  }
  $nodeVer = (& node --version).TrimStart("v")
  $nodeMajor = [int]($nodeVer.Split(".")[0])
  if ($nodeMajor -lt 18) { Fail "Node.js 18+ required; found v$nodeVer" }
  Write-Ok "Node v$nodeVer"
  Write-Ok ("npm " + (& npm --version))
}

# ---------------------------------------------------------------------------
# 2. Python virtual environment
# ---------------------------------------------------------------------------

Write-Section "Setting up Python environment"

Set-Location $RepoRoot

if (-not $NoVenv) {
  $venvDir = Join-Path $RepoRoot ".venv"
  if (Test-Path $venvDir) {
    Write-Ok "Using existing virtual environment at .venv"
  } else {
    Write-Msg "Creating virtual environment at .venv"
    & $Python -m venv $venvDir
  }
  $Python = Join-Path $venvDir "Scripts\python.exe"
  if (-not (Test-Path $Python)) {
    Fail "Virtual environment creation failed: $Python missing"
  }
  Write-Ok "Virtual environment ready"
} else {
  Write-Warn "Using system Python (-NoVenv)."
}

Write-Msg "Upgrading pip inside the environment"
Invoke-Step -Label "pip upgrade" -Exe $Python -ExeArgs @("-m","pip","install","--upgrade","pip","--quiet")

# ---------------------------------------------------------------------------
# 3. Backend install
# ---------------------------------------------------------------------------

Write-Section "Installing backend (editable + dev extras)"

Invoke-Step -Label "backend install" -Exe $Python -ExeArgs @("-m","pip","install","--quiet","-e","$RepoRoot\backend[dev]")
Write-Ok "Backend installed"

# ---------------------------------------------------------------------------
# 4. UI install + build
# ---------------------------------------------------------------------------

if ($SkipUI) {
  Write-Warn "Skipping UI build (-SkipUI)."
} else {
  Write-Section "Installing UI dependencies"
  Set-Location (Join-Path $RepoRoot "ui")

  if (Test-Path "package-lock.json") {
    Invoke-Step -Label "npm ci" -Exe "npm" -ExeArgs @("ci","--silent","--no-audit","--no-fund")
  } else {
    Invoke-Step -Label "npm install" -Exe "npm" -ExeArgs @("install","--silent","--no-audit","--no-fund")
  }
  Write-Ok "UI dependencies installed"

  Write-Section "Building UI bundle"
  Invoke-Step -Label "npm run build" -Exe "npm" -ExeArgs @("run","build")
  if (-not (Test-Path (Join-Path $RepoRoot "ui\dist\index.html"))) {
    Fail "UI build completed without emitting ui\dist\index.html"
  }
  Write-Ok "UI bundle ready at ui\dist\"
  Set-Location $RepoRoot
}

# ---------------------------------------------------------------------------
# 5. Regenerate shared schemas
# ---------------------------------------------------------------------------

Write-Section "Regenerating shared schemas"

# ``regen_schemas.py`` emits harmless ``PydanticJsonSchemaWarning`` lines
# on stderr that PowerShell would otherwise promote to failures. The
# 2>$null redirect discards them; rely on $LASTEXITCODE to catch genuine
# crashes.
$schemaScript = Join-Path $RepoRoot "backend\scripts\regen_schemas.py"
& $Python $schemaScript 2>$null | Out-Null
if ($LASTEXITCODE -ne 0) {
  Fail "regen_schemas.py exited with status $LASTEXITCODE"
}
Write-Ok "shared\openapi.yaml + JSON Schemas refreshed"

# ---------------------------------------------------------------------------
# 6. Smoke test
# ---------------------------------------------------------------------------

if ($SkipTests) {
  Write-Warn "Skipping smoke tests (-SkipTests)."
} else {
  Write-Section "Running smoke tests"

  Write-Msg "CLI help check"
  & $Python -m ollama_evaluator.cli --help | Out-Null
  if ($LASTEXITCODE -ne 0) { Fail "CLI --help returned status $LASTEXITCODE" }
  Write-Ok "CLI imports and loads"

  Write-Msg "Unit tests"
  Push-Location (Join-Path $RepoRoot "backend")
  try {
    & $Python -m pytest tests\unit -q -x
    if ($LASTEXITCODE -ne 0) { Fail "pytest exited with status $LASTEXITCODE" }
  } finally {
    Pop-Location
  }
  Write-Ok "Unit tests passed"
}

# ---------------------------------------------------------------------------
# 7. Next steps
# ---------------------------------------------------------------------------

Write-Host ""
Write-Host "Install complete." -ForegroundColor Green

Write-Host ""
Write-Host "Activate the environment for every new shell session:"
Write-Host "  cd $RepoRoot"
Write-Host "  .\.venv\Scripts\Activate.ps1"

Write-Host ""
Write-Host "Next steps:"
Write-Host "  - Pull a small Ollama model:    ollama pull llama3:8b"
Write-Host "  - Edit examples\config.qwen.yaml and point models: at your model."
Write-Host "  - Validate the example suite:"
Write-Host "      python -m ollama_evaluator.cli validate-suite examples\suites\reasoning-basics.yaml"
Write-Host "  - List Ollama models:"
Write-Host "      python -m ollama_evaluator.cli list-models"
Write-Host "  - Run an evaluation:"
Write-Host "      python -m ollama_evaluator.cli --config examples\config.qwen.yaml run"
Write-Host "  - Start the web UI + API:"
Write-Host '      $env:OLLAMA_EVAL_UI_DIR = "$PWD\ui\dist"; python -m ollama_evaluator.cli --config examples\config.qwen.yaml serve'
Write-Host ""
Write-Host "User manual: docs\USER_MANUAL.md"
Write-Host ""

# Explicit zero exit to ensure the process reports success even if
# earlier stderr was ignored.
exit 0
