# Keepalive v2 = v1 logging, plus how long the VM spent RUNNING / TERMINATED.
# Requires Google Cloud SDK + `gcloud auth login`.
#
# powershell -ExecutionPolicy Bypass -File .\scripts\v2\gcp_spot_keepalive.ps1
# Leave this window open. Ctrl+C to stop.

$ErrorActionPreference = "Continue"

# --- config ---
$Project  = "project-87fa1e72-4d28-42f5-91e"
$Zone     = "us-central1-c"
$Instance = "minilml4usc1"

$PollWhenRunningSec = 60
$RetryWhenDownSec   = 1
$LogFile            = Join-Path $PSScriptRoot "gcp_spot_keepalive.log"

function Write-Log([string]$Message) {
    $line = "{0:yyyy-MM-dd HH:mm:ss}  {1}" -f (Get-Date), $Message
    Write-Host $line
    Add-Content -Path $LogFile -Value $line -Encoding utf8
}

function Format-Duration([TimeSpan]$Span) {
    if ($Span.TotalSeconds -lt 0) { $Span = [TimeSpan]::Zero }
    $parts = @()
    if ($Span.Days -gt 0) { $parts += "{0}d" -f $Span.Days }
    if ($Span.Hours -gt 0 -or $Span.Days -gt 0) { $parts += "{0}h" -f $Span.Hours }
    if ($Span.Minutes -gt 0 -or $parts.Count -gt 0) { $parts += "{0}m" -f $Span.Minutes }
    $parts += "{0}s" -f $Span.Seconds
    return ($parts -join " ")
}

function Get-InstanceInfo {
    $gcloudArgs = @(
        "compute", "instances", "describe", $Instance,
        "--zone", $Zone,
        "--format", "value(status,lastStartTimestamp)"
    )
    if ($Project) { $gcloudArgs += @("--project", $Project) }

    $raw = & gcloud @gcloudArgs 2>&1
    if ($LASTEXITCODE -ne 0) {
        return @{ Ok = $false; Error = ($raw | Out-String).Trim() }
    }

    $line = ($raw | Out-String).Trim()
    $parts = $line -split "`t", 2
    $status = $parts[0].Trim()
    $startAt = $null
    if ($parts.Count -gt 1 -and $parts[1].Trim()) {
        try { $startAt = [datetime]::Parse($parts[1].Trim()).ToLocalTime() } catch { }
    }

    return @{
        Ok      = $true
        Status  = $status
        StartAt = $startAt
        Error   = ""
    }
}

function Start-SpotInstance {
    $gcloudArgs = @(
        "compute", "instances", "start", $Instance,
        "--zone", $Zone,
        "--async"
    )
    if ($Project) { $gcloudArgs += @("--project", $Project) }

    $out = & gcloud @gcloudArgs 2>&1
    return @{
        Ok     = ($LASTEXITCODE -eq 0)
        Output = ($out | Out-String).Trim()
    }
}

if (-not (Get-Command gcloud -ErrorAction SilentlyContinue)) {
    Write-Host "gcloud not found. Install Google Cloud SDK and reopen the terminal."
    exit 1
}

$phase = "unknown"   # up | down
$upSince = $null
$downSince = $null

function Enter-Down([datetime]$Now, $Info, [string]$Status) {
    if ($script:phase -eq "up") {
        $ranFrom = if ($Info.StartAt) { $Info.StartAt } elseif ($script:upSince) { $script:upSince } else { $Now }
        Write-Log ("WENT OFF after running {0}  (status={1})" -f (Format-Duration ($Now - $ranFrom)), $Status)
    }
    if ($script:phase -ne "down") {
        $script:downSince = $Now
        $script:upSince = $null
        $script:phase = "down"
    }
}

Write-Log "keepalive v2 start  project=$Project  zone=$Zone  instance=$Instance"

while ($true) {
    $info = Get-InstanceInfo
    $now = Get-Date

    if (-not $info.Ok) {
        Write-Log "describe failed: $($info.Error)"
        Start-Sleep -Seconds $RetryWhenDownSec
        continue
    }

    $status = $info.Status

    switch -Regex ($status) {
        "^(RUNNING|STAGING|PROVISIONING)$" {
            if ($phase -eq "down" -and $downSince) {
                Write-Log ("BACK ONLINE after {0} of start retries" -f (Format-Duration ($now - $downSince)))
            }

            if ($phase -ne "up") {
                $phase = "up"
                $upSince = if ($info.StartAt) { $info.StartAt } else { $now }
                $downSince = $null
            } elseif ($info.StartAt) {
                $upSince = $info.StartAt
            }

            $upFor = if ($upSince) { Format-Duration ($now - $upSince) } else { "?" }
            Write-Log "status=$status  (ok)  up $upFor"

            if ($status -eq "RUNNING") {
                Start-Sleep -Seconds $PollWhenRunningSec
            } else {
                Start-Sleep -Seconds $RetryWhenDownSec
            }
        }
        "^(STOPPING|SUSPENDING|REPAIRING)$" {
            Enter-Down $now $info $status
            $downFor = if ($downSince) { Format-Duration ($now - $downSince) } else { "?" }
            Write-Log "status=$status  waiting...  down $downFor"
            Start-Sleep -Seconds $RetryWhenDownSec
        }
        default {
            Enter-Down $now $info $status
            $downFor = if ($downSince) { Format-Duration ($now - $downSince) } else { "?" }
            Write-Log "status=$status  trying start...  down $downFor"
            $start = Start-SpotInstance
            if ($start.Ok) {
                Write-Log "start requested"
            } else {
                Write-Log "start failed (spot full / quota / other): $($start.Output)"
            }
            Start-Sleep -Seconds $RetryWhenDownSec
        }
    }
}
