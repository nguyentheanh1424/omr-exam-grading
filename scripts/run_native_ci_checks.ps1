Param(
    [string]$RepoRoot = (Resolve-Path "$PSScriptRoot\..").Path
)

$ErrorActionPreference = "Stop"

$repo = Resolve-Path $RepoRoot
$buildDir = Join-Path $repo "build\native_core"
$vsDevCmd = "C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\Common7\Tools\VsDevCmd.bat"
$ninja = "C:\PROGRA~2\MICROS~3\18\BUILDT~1\Common7\IDE\COMMON~1\MICROS~1\CMake\Ninja\ninja.exe"

Write-Host "[1/4] Build native_core"
Push-Location $buildDir
cmd /c "`"$vsDevCmd`" -arch=x64 && `"$ninja`"" | Out-Host
if ($LASTEXITCODE -ne 0) {
    throw "native_core build failed with exit code $LASTEXITCODE"
}
Pop-Location

Write-Host "[2/4] Run C++ unit tests"
& (Join-Path $buildDir "omr_unit_tests.exe")
if ($LASTEXITCODE -ne 0) {
    throw "C++ unit tests failed with exit code $LASTEXITCODE"
}

Write-Host "[3/4] Run synthetic parity"
Push-Location $repo
$env:PYTHONPATH = "."
python -m native_core.tests.parity_grading
if ($LASTEXITCODE -ne 0) {
    throw "synthetic parity failed with exit code $LASTEXITCODE"
}

Write-Host "[4/4] Run best-path real-image parity"
python -m native_core.tests.parity_python_warp_python_prep_native_grading
if ($LASTEXITCODE -ne 0) {
    throw "best-path real-image parity failed with exit code $LASTEXITCODE"
}
Pop-Location

Write-Host "[DONE] native CI checks passed"
