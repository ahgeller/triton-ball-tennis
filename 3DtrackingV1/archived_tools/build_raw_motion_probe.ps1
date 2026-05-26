param(
    [string]$CondaPrefix = "",
    [string]$Out = "3DtrackingV1\archived_tools\raw_motion_probe.exe"
)

if (-not $CondaPrefix) {
    try {
        $PythonPrefix = (& python -c "import sys; print(sys.prefix)" 2>$null).Trim()
    } catch {
        $PythonPrefix = ""
    }
    if ($PythonPrefix -and (Test-Path (Join-Path $PythonPrefix "Library\include\opencv2\opencv.hpp"))) {
        $CondaPrefix = $PythonPrefix
    } elseif ($env:CONDA_PREFIX -and (Test-Path (Join-Path $env:CONDA_PREFIX "Library\include\opencv2\opencv.hpp"))) {
        $CondaPrefix = $env:CONDA_PREFIX
    } else {
        $CondaPrefix = "C:\Users\Andrew\.conda\envs\tennis-analysis"
    }
}

$include = Join-Path $CondaPrefix "Library\include"
$lib = Join-Path $CondaPrefix "Library\lib"
$bin = Join-Path $CondaPrefix "Library\bin"

function Find-VcVars64 {
    $candidates = @()
    $vswhere = "C:\Program Files (x86)\Microsoft Visual Studio\Installer\vswhere.exe"
    if (Test-Path $vswhere) {
        try {
            $installPath = (& $vswhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath 2>$null).Trim()
            if ($installPath) {
                $candidates += (Join-Path $installPath "VC\Auxiliary\Build\vcvars64.bat")
            }
        } catch {}
    }
    $candidates += @(
        "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat",
        "C:\Program Files\Microsoft Visual Studio\2022\Professional\VC\Auxiliary\Build\vcvars64.bat",
        "C:\Program Files\Microsoft Visual Studio\2022\Enterprise\VC\Auxiliary\Build\vcvars64.bat",
        "C:\Program Files\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat",
        "C:\Program Files (x86)\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat",
        "C:\Program Files (x86)\Microsoft Visual Studio\2022\Professional\VC\Auxiliary\Build\vcvars64.bat",
        "C:\Program Files (x86)\Microsoft Visual Studio\2022\Enterprise\VC\Auxiliary\Build\vcvars64.bat",
        "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
    )
    foreach ($candidate in $candidates) {
        if ($candidate -and (Test-Path $candidate)) {
            return $candidate
        }
    }
    foreach ($root in @("C:\Program Files\Microsoft Visual Studio", "C:\Program Files (x86)\Microsoft Visual Studio")) {
        if (Test-Path $root) {
            $found = Get-ChildItem $root -Recurse -Filter vcvars64.bat -ErrorAction SilentlyContinue |
                Select-Object -First 1
            if ($found) {
                return $found.FullName
            }
        }
    }
    return $null
}

$clCommand = Get-Command cl.exe -ErrorAction SilentlyContinue
$clPath = if ($clCommand) { $clCommand.Source } else { "" }
$needsVcVars64 = (-not $clPath) -or ($clPath -notmatch '(?i)Hostx64\\x64')
$vcvars64 = $null
if ($needsVcVars64) {
    $vcvars64 = Find-VcVars64
    if (-not $vcvars64) {
        if ($clPath) {
            Write-Error "Found cl.exe, but it is not x64: $clPath. Could not find vcvars64.bat automatically."
        } else {
            Write-Error "cl.exe is not on PATH and vcvars64.bat was not found automatically."
        }
        exit 1
    }
    Write-Warning "Using x64 toolchain from: $vcvars64"
}

if (-not (Test-Path (Join-Path $include "opencv2\opencv.hpp"))) {
    Write-Error "OpenCV headers not found under $include"
    exit 1
}

$libs = @(
    "opencv_core490.lib",
    "opencv_imgproc490.lib",
    "opencv_video490.lib",
    "opencv_videoio490.lib",
    "opencv_imgcodecs490.lib",
    "opencv_highgui490.lib"
)

$missing = @()
foreach ($name in $libs) {
    if (-not (Test-Path (Join-Path $lib $name))) {
        $missing += $name
    }
}
if ($missing.Count -gt 0) {
    Write-Error "Missing OpenCV libs under ${lib}: $($missing -join ', ')"
    exit 1
}

$env:PATH = "$bin;$env:PATH"

$outParent = Split-Path -Parent $Out
if ($outParent -and -not (Test-Path $outParent)) {
    New-Item -ItemType Directory -Force -Path $outParent | Out-Null
}

$buildLog = "3DtrackingV1\archived_tools\raw_motion_probe_build.log"
$clArgs = @(
    "/nologo",
    "/std:c++17",
    "/O2",
    "/EHsc",
    "/I", "$include",
    "3DtrackingV1\archived_tools\raw_motion_probe.cpp",
    "/Fo:3DtrackingV1\archived_tools\raw_motion_probe.obj",
    "/Fe:$Out",
    "/link",
    "/LIBPATH:$lib"
) + $libs

Write-Host "Using CondaPrefix: $CondaPrefix"
Write-Host "Building $Out ..."
if ($needsVcVars64) {
    $quotedArgs = $clArgs | ForEach-Object {
        if ($_ -match '[\s&()]') { '"' + ($_ -replace '"', '\"') + '"' } else { $_ }
    }
    $cmdLine = 'call "' + $vcvars64 + '" >nul && cl.exe ' + ($quotedArgs -join ' ')
    & cmd.exe /s /c $cmdLine 2>&1 | Tee-Object -FilePath $buildLog
} else {
    & cl.exe @clArgs 2>&1 | Tee-Object -FilePath $buildLog
}
if ($LASTEXITCODE -ne 0) {
    Write-Error "Build failed. See $buildLog"
    exit $LASTEXITCODE
}

if (-not (Test-Path $Out)) {
    Write-Error "Build completed but $Out was not created. See $buildLog"
    exit 1
}

Write-Host "Built $Out"
