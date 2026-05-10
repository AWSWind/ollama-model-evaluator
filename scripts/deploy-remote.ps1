<#
.SYNOPSIS
    Ollama Model Evaluator - one-button remote deployment (Windows).

.DESCRIPTION
    Windows PowerShell twin of ``scripts/deploy-remote.sh``. Tars the
    repo, pushes to the remote host via scp, and runs ``install.sh``
    on the remote.

    Prerequisites on this machine:
      - OpenSSH (ssh/scp) - ships with Windows 10+.
      - bsdtar (tar.exe)  - ships with Windows 10+.

    Prerequisites on the remote:
      - bash, python3 >= 3.11, node >= 18, npm.
      - Ollama (optional; required for real evaluation runs).

.PARAMETER Target
    The ssh target, in ``user@host`` form. Required.

.PARAMETER RemoteDir
    The directory on the remote where the project should live. Default
    is ``~/ollama-model-evaluator``.

.PARAMETER Key
    Path to the ssh private key. Default looks for
    ``$HOME\.ssh\id_ed25519_azurewind`` then ``id_ed25519`` then
    ``id_rsa``.

.PARAMETER Port
    SSH port. Default 22.

.PARAMETER SkipTests
    Forward ``--skip-tests`` to the remote install.sh.

.PARAMETER SkipUI
    Forward ``--skip-ui`` to the remote install.sh.

.PARAMETER NoInstall
    Only sync files; do not run install.sh on the remote.

.PARAMETER ServePort
    After install, start the HTTP+WS+UI server in the background on
    this port.

.PARAMETER Config
    Path (relative to RemoteDir) of the config file the remote serve
    uses. Default ``examples/config.qwen.yaml``.

.EXAMPLE
    # Deploy + install
    .\scripts\deploy-remote.ps1 -Target azurewind@192.168.1.224

.EXAMPLE
    # Deploy, install, and also start the server on port 8765
    .\scripts\deploy-remote.ps1 -Target azurewind@192.168.1.224 `
        -RemoteDir /home/azurewind/workspaces/AI-Model-Evaluation `
        -ServePort 8765

.EXAMPLE
    # Only sync files (no remote install) - useful for quick code pushes
    .\scripts\deploy-remote.ps1 -Target user@host -NoInstall
#>

[CmdletBinding()]
param(
  [Parameter(Mandatory = $true)][string]$Target,
  [string]$RemoteDir = "~/ollama-model-evaluator",
  [string]$Key = "",
  [int]$Port = 22,
  [switch]$SkipTests,
  [switch]$SkipUI,
  [switch]$NoInstall,
  [int]$ServePort = 0,
  [string]$Config = "examples/config.qwen.yaml"
)

# Let native commands report failures via exit codes rather than having
# PowerShell promote stderr to terminating errors.
$ErrorActionPreference = "Continue"

$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

function Write-Section { param([string]$msg)
  Write-Host ""
  Write-Host ("== " + $msg) -ForegroundColor Blue
}
function Write-Msg  { param([string]$msg) Write-Host (">> " + $msg) -ForegroundColor Cyan }
function Write-Ok   { param([string]$msg) Write-Host ("OK " + $msg) -ForegroundColor Green }
function Write-Warn { param([string]$msg) Write-Host ("!! " + $msg) -ForegroundColor Yellow }
function Fail       { param([string]$msg) Write-Host ("** " + $msg) -ForegroundColor Red; exit 1 }

function Invoke-Remote {
  <#
  .SYNOPSIS
      Run a command on the remote host and abort on non-zero exit.
  .DESCRIPTION
      Centralises the ssh invocation so the key, port, and batch-mode
      options are always in sync and callers only have to think about
      the remote-side shell command.
  #>
  param(
    [Parameter(Mandatory)][string]$Command,
    [switch]$AllowFailure
  )
  $sshArgs = @()
  if (-not [string]::IsNullOrWhiteSpace($Key)) { $sshArgs += @("-i", $Key) }
  $sshArgs += @(
    "-p", $Port,
    "-o", "BatchMode=yes",
    "-o", "StrictHostKeyChecking=accept-new",
    $Target,
    $Command
  )
  & ssh @sshArgs
  if ($LASTEXITCODE -ne 0 -and -not $AllowFailure) {
    Fail "Remote command failed (exit $LASTEXITCODE): $Command"
  }
}

function Invoke-SCP {
  <#
  .SYNOPSIS
      Upload a local path to the remote via scp.
  #>
  param(
    [Parameter(Mandatory)][string]$Local,
    [Parameter(Mandatory)][string]$Remote
  )
  $scpArgs = @()
  if (-not [string]::IsNullOrWhiteSpace($Key)) { $scpArgs += @("-i", $Key) }
  $scpArgs += @(
    "-P", $Port,
    "-o", "BatchMode=yes",
    "-o", "StrictHostKeyChecking=accept-new",
    $Local,
    "${Target}:${Remote}"
  )
  & scp @scpArgs | Out-Null
  if ($LASTEXITCODE -ne 0) {
    Fail "scp failed (exit $LASTEXITCODE): $Local -> $Target`:$Remote"
  }
}

# ---------------------------------------------------------------------------
# 1. Preflight on this machine
# ---------------------------------------------------------------------------

Write-Section "Preflight (control host)"

foreach ($tool in @("ssh","scp","tar")) {
  if (-not (Get-Command $tool -ErrorAction SilentlyContinue)) {
    Fail "$tool not found on this machine. Install OpenSSH or Git for Windows."
  }
}
Write-Ok "ssh, scp, tar available"

if ([string]::IsNullOrWhiteSpace($Key)) {
  $candidateKeys = @(
    (Join-Path $HOME ".ssh\id_ed25519_azurewind"),
    (Join-Path $HOME ".ssh\id_ed25519"),
    (Join-Path $HOME ".ssh\id_rsa")
  )
  foreach ($c in $candidateKeys) {
    if (Test-Path $c) { $Key = $c; break }
  }
}
if (-not [string]::IsNullOrWhiteSpace($Key)) {
  if (-not (Test-Path $Key)) { Fail "SSH key not found: $Key" }
  Write-Ok "Using SSH key: $Key"
} else {
  Write-Warn "No SSH key found; relying on ssh-agent or password auth"
}

Write-Msg "Verifying SSH access to $Target"
Invoke-Remote -Command "echo REMOTE_OK" | Out-Null
Write-Ok "SSH access confirmed"

# ---------------------------------------------------------------------------
# 2. Tarball creation
# ---------------------------------------------------------------------------

Write-Section "Building deployment tarball"

# ``[IO.Path]::GetTempPath()`` is always writable and persists long enough
# for the script; we remove the file in the finally block below.
$tarball = Join-Path ([IO.Path]::GetTempPath()) ("ollama-eval-" + [Guid]::NewGuid().ToString("N").Substring(0,8) + ".tgz")

Push-Location $RepoRoot
try {
  # tar 1.30+ (ships with Windows 10 1803+) handles --exclude inline.
  # Every exclude corresponds to a generated artefact the remote
  # install.sh regenerates; shipping them would bloat the tarball and
  # leak ambient state.
  $tarArgs = @(
    "--exclude=__pycache__",
    "--exclude=*.egg-info",
    "--exclude=.pytest_cache",
    "--exclude=.mypy_cache",
    "--exclude=.ruff_cache",
    "--exclude=.hypothesis",
    "--exclude=.venv",
    "--exclude=node_modules",
    "--exclude=dist",
    "--exclude=.git",
    "-czf", $tarball,
    "backend", "ui", "shared", "examples", "scripts", "docs", "README.md"
  )
  & tar @tarArgs
  if ($LASTEXITCODE -ne 0) { Fail "tar failed with exit $LASTEXITCODE" }
  $sizeKB = [Math]::Round((Get-Item $tarball).Length / 1KB, 1)
  Write-Ok "Tarball built: $tarball ($sizeKB KiB)"
} finally {
  Pop-Location
}

try {
  # -----------------------------------------------------------------------
  # 3. Upload + extract
  # -----------------------------------------------------------------------

  Write-Section "Uploading to $Target`:$RemoteDir"

  $remoteTarball = "/tmp/ollama-evaluator-deploy.$([int]([DateTime]::UtcNow - [DateTime]'1970-01-01').TotalSeconds).tgz"

  Write-Msg "Uploading tarball"
  Invoke-SCP -Local $tarball -Remote $remoteTarball
  Write-Ok "Uploaded"

  Write-Msg "Extracting into $RemoteDir"
  Invoke-Remote -Command "mkdir -p $RemoteDir && tar -xzf $remoteTarball -C $RemoteDir && rm -f $remoteTarball"
  Write-Ok "Project synced"

  # -----------------------------------------------------------------------
  # 4. Remote install
  # -----------------------------------------------------------------------

  if (-not $NoInstall) {
    Write-Section "Running install.sh on the remote"

    $installFlags = @()
    if ($SkipTests) { $installFlags += "--skip-tests" }
    if ($SkipUI)    { $installFlags += "--skip-ui" }

    $remoteCmd = "cd $RemoteDir && chmod +x scripts/install.sh && ./scripts/install.sh $($installFlags -join ' ')"
    Invoke-Remote -Command $remoteCmd
    Write-Ok "Remote install finished"
  }

  # -----------------------------------------------------------------------
  # 5. Optional: start the server in the background
  # -----------------------------------------------------------------------

  if ($ServePort -gt 0) {
    Write-Section "Starting server on remote (port $ServePort)"

    # Kill any previous server so we do not hit "address already in use".
    Invoke-Remote -AllowFailure -Command "pkill -f 'ollama_evaluator.cli.*serve' 2>/dev/null; sleep 1" | Out-Null

    # ``nohup`` + ``&`` lets the process survive the ssh channel closing;
    # serve.log captures stdout+stderr for later inspection.
    $serveCmd = "cd $RemoteDir && nohup bash -c 'source .venv/bin/activate && OLLAMA_EVAL_UI_DIR=`$PWD/ui/dist python -m ollama_evaluator.cli --config $Config serve --host 0.0.0.0 --port $ServePort' > serve.log 2>&1 &"
    Invoke-Remote -Command $serveCmd
    Start-Sleep -Seconds 2

    # Verify the server is actually listening before we claim success.
    $check = ssh -o BatchMode=yes -p $Port $(if ($Key) { "-i", $Key } else { @() }) $Target "ss -tlnp 2>/dev/null | grep -q ':$ServePort ' && echo BOUND || echo UNBOUND"
    if ($check -match "BOUND") {
      Write-Ok "Server listening on ${Target}:$ServePort"
      $hostOnly = $Target -replace '^.*@', ''
      Write-Host ""
      Write-Host "Server ready. Try from your browser:"
      Write-Host "  http://${hostOnly}:${ServePort}/        # Web UI"
      Write-Host "  http://${hostOnly}:${ServePort}/api/health"
      Write-Host "  http://${hostOnly}:${ServePort}/api/models"
    } else {
      Write-Warn "Server did not bind port $ServePort within 2s; last 20 lines of serve.log:"
      Invoke-Remote -AllowFailure -Command "tail -20 $RemoteDir/serve.log"
    }
  }

  # -----------------------------------------------------------------------
  # 6. Done
  # -----------------------------------------------------------------------

  Write-Host ""
  Write-Host "Deployment complete." -ForegroundColor Green
  Write-Host "  Remote path: ${Target}:$RemoteDir"

  $hostOnly = $Target -replace '^.*@', ''
  Write-Host ""
  Write-Host "Try a quick run over ssh:"
  $sshTrail = ""
  if ($Key) { $sshTrail = "-i $Key " }
  Write-Host "  ssh ${sshTrail}${Target} `"cd $RemoteDir && source .venv/bin/activate && python -m ollama_evaluator.cli list-models`""

} finally {
  if (Test-Path $tarball) { Remove-Item $tarball -Force }
}

exit 0
