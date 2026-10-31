param(
  [switch]$SkipFrontendBuild
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$iconPath = Join-Path $projectRoot "assets\app-icon.ico"
$frontendDist = Join-Path $projectRoot "frontend\dist"
$backendPath = Join-Path $projectRoot "backend"
$entryPoint = Join-Path $projectRoot "desktop_app.py"
$appName = "Clinthinker" + [char]0x00A9
$webView2InstallerName = "MicrosoftEdgeWebView2RuntimeInstallerX64.exe"
$webView2DownloadUrl = "https://go.microsoft.com/fwlink/p/?LinkId=2124701"
$webView2CacheDirectory = Join-Path $projectRoot "build\prerequisites"
$webView2CachedInstaller = Join-Path $webView2CacheDirectory $webView2InstallerName

function Get-ProjectPythonCommand {
  if (Get-Command python -ErrorAction SilentlyContinue) {
    return ,@("python")
  }
  if (Get-Command py -ErrorAction SilentlyContinue) {
    return ,@("py", "-3")
  }
  return $null
}

function Invoke-ProjectPython {
  param(
    [Parameter(Mandatory = $true)]
    [string[]]$Arguments
  )

  $pythonCommand = Get-ProjectPythonCommand
  if ($null -eq $pythonCommand) {
    throw "Python 3.10+ is required to build the desktop app. Install Python 3.10+ and make sure either 'python' or 'py' is on PATH."
  }

  $pythonArgs = @()
  if ($pythonCommand.Length -gt 1) {
    $pythonArgs += $pythonCommand[1..($pythonCommand.Length - 1)]
  }

  & $pythonCommand[0] @($pythonArgs + $Arguments) | Out-Host
  $exitCode = $LASTEXITCODE
  return $exitCode
}

Push-Location $projectRoot

try {
  if (-not [Environment]::Is64BitOperatingSystem) {
    throw "The portable desktop app currently requires 64-bit Windows."
  }

  $pythonCommand = Get-ProjectPythonCommand
  if ($null -eq $pythonCommand) {
    throw "Python 3.10+ is required to build the x64 portable desktop app."
  }

  if ($pythonCommand[0] -eq "py") {
    $pythonBits = & $pythonCommand[0] @($pythonCommand[1..($pythonCommand.Length - 1)] + @("-c", "import struct; print(struct.calcsize('P') * 8)"))
  }
  else {
    $pythonBits = & $pythonCommand[0] @("-c", "import struct; print(struct.calcsize('P') * 8)")
  }
  if ($LASTEXITCODE -ne 0 -or $pythonBits.Trim() -ne "64") {
    if ($pythonBits -and $pythonBits.Trim() -eq "32") {
      throw "Use 64-bit Python to build the x64 portable desktop app."
    }
    throw "Could not detect a supported 64-bit Python interpreter."
  }

  if (-not $SkipFrontendBuild) {
    npm.cmd ci --prefix frontend
    if ($LASTEXITCODE -ne 0) {
      throw "Installing frontend dependencies failed."
    }

    npm.cmd run build:web --prefix frontend
    if ($LASTEXITCODE -ne 0) {
      throw "The frontend production build failed."
    }
  }

  $installResult = Invoke-ProjectPython -Arguments @("-m", "pip", "install", "-r", "desktop-requirements.txt")
  if ($installResult -ne 0) {
    throw "Installing desktop app dependencies failed."
  }

  New-Item -ItemType Directory -Path $webView2CacheDirectory -Force | Out-Null
  $signature = if (Test-Path -LiteralPath $webView2CachedInstaller) {
    Get-AuthenticodeSignature -FilePath $webView2CachedInstaller
  }
  if (-not $signature -or $signature.Status -ne "Valid" `
      -or $signature.SignerCertificate.Subject -notlike "*Microsoft Corporation*") {
    if (Test-Path -LiteralPath $webView2CachedInstaller) {
      Remove-Item -LiteralPath $webView2CachedInstaller -Force
    }
    Write-Host "Downloading Microsoft's x64 WebView2 offline runtime installer..."
    Invoke-WebRequest -Uri $webView2DownloadUrl -OutFile $webView2CachedInstaller
    $signature = Get-AuthenticodeSignature -FilePath $webView2CachedInstaller
    if ($signature.Status -ne "Valid" `
        -or $signature.SignerCertificate.Subject -notlike "*Microsoft Corporation*") {
      throw "The downloaded WebView2 installer failed Microsoft signature validation."
    }
  }

  $packagingResult = Invoke-ProjectPython -Arguments @(
    "-m", "PyInstaller",
    "--noconfirm",
    "--clean",
    "--windowed",
    "--onedir",
    "--name", $appName,
    "--icon", $iconPath,
    "--add-data", "$frontendDist;frontend_dist",
    "--collect-all", "sklearn",
    "--collect-all", "scipy",
    "--collect-all", "pdfplumber",
    "--collect-all", "pypdfium2",
    "--collect-all", "docx",
    "--collect-all", "pytesseract",
    "--collect-all", "webview",
    "--paths", $backendPath,
    "--distpath", "release",
    "--workpath", "build\pyinstaller",
    "--specpath", "build\pyinstaller",
    $entryPoint
  )
  if ($packagingResult -ne 0) {
    throw "Packaging the Windows desktop app failed."
  }

  $appExecutable = Join-Path $projectRoot "release\$appName\$appName.exe"
  if (-not (Test-Path -LiteralPath $appExecutable)) {
    throw "The packaged desktop executable was not created."
  }

  $prerequisitesDirectory = Join-Path $projectRoot "release\$appName\Prerequisites"
  New-Item -ItemType Directory -Path $prerequisitesDirectory -Force | Out-Null
  Copy-Item -LiteralPath $webView2CachedInstaller `
    -Destination (Join-Path $prerequisitesDirectory $webView2InstallerName) -Force

  $shell = New-Object -ComObject WScript.Shell
  $desktopDirectory = [Environment]::GetFolderPath("Desktop")
  $desktopShortcut = $shell.CreateShortcut((Join-Path $desktopDirectory "$appName.lnk"))
  $desktopShortcut.TargetPath = $appExecutable
  $desktopShortcut.WorkingDirectory = Split-Path -Parent $appExecutable
  $desktopShortcut.IconLocation = "$appExecutable,0"
  $desktopShortcut.Description = $appName
  $desktopShortcut.Save()

  $oldDesktopShortcut = Join-Path $desktopDirectory "Topic Prediction Assistant.lnk"
  if (Test-Path -LiteralPath $oldDesktopShortcut) {
    Remove-Item -LiteralPath $oldDesktopShortcut -Force
  }

  $programsDirectory = [Environment]::GetFolderPath("Programs")
  $startMenuDirectory = Join-Path $programsDirectory $appName
  New-Item -ItemType Directory -Path $startMenuDirectory -Force | Out-Null
  $startMenuShortcut = $shell.CreateShortcut(
    (Join-Path $startMenuDirectory "$appName.lnk")
  )
  $startMenuShortcut.TargetPath = $appExecutable
  $startMenuShortcut.WorkingDirectory = Split-Path -Parent $appExecutable
  $startMenuShortcut.IconLocation = "$appExecutable,0"
  $startMenuShortcut.Description = $appName
  $startMenuShortcut.Save()

  $oldStartMenuDirectory = Join-Path $programsDirectory "Topic Prediction Assistant"
  $oldStartMenuShortcut = Join-Path $oldStartMenuDirectory "Topic Prediction Assistant.lnk"
  if (Test-Path -LiteralPath $oldStartMenuShortcut) {
    Remove-Item -LiteralPath $oldStartMenuShortcut -Force
  }
  if ((Test-Path -LiteralPath $oldStartMenuDirectory) `
      -and -not (Get-ChildItem -LiteralPath $oldStartMenuDirectory -Force)) {
    Remove-Item -LiteralPath $oldStartMenuDirectory
  }

  Write-Host ""
  Write-Host "$appName desktop app created at:"
  Write-Host $appExecutable
  Write-Host "The portable app folder includes the offline WebView2 installer."
  Write-Host "Desktop and Start menu shortcuts were created."
}
finally {
  Pop-Location
}
