param([switch]$Tests,[switch]$Check)
$ErrorActionPreference='Stop'
$taskCargo=Join-Path $env:USERPROFILE '.cargo/bin/cargo.exe'
$taskVcVars='C:\BuildTools\VC\Auxiliary\Build\vcvars64.bat'
if(-not(Test-Path -LiteralPath $taskVcVars)){
    $taskVsWhere=Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio/Installer/vswhere.exe'
    if(Test-Path -LiteralPath $taskVsWhere){
        $taskVsInstall=& $taskVsWhere -latest -products '*' -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
        $taskVcVars=Join-Path $taskVsInstall 'VC/Auxiliary/Build/vcvars64.bat'
    }
}
if(-not(Test-Path -LiteralPath $taskVcVars)){throw 'MSVC C++ Build Tools and Windows SDK are required.'}
# Load build environment only into this PowerShell process. No user/global PATH changes.
$taskCommand='call "'+$taskVcVars+'" >nul && set'
& cmd.exe /d /c $taskCommand | ForEach-Object {
    if($_ -match '^([^=]+)=(.*)$'){[Environment]::SetEnvironmentVariable($matches[1],$matches[2],'Process')}
}
$taskSdkBase=Join-Path $PSScriptRoot '.build/windows-sdk-x64/c'
if(Test-Path -LiteralPath (Join-Path $taskSdkBase 'um/x64/kernel32.lib')){
    $env:LIB=(Join-Path $taskSdkBase 'um/x64')+';'+(Join-Path $taskSdkBase 'ucrt/x64')+';'+$env:LIB
}
Push-Location $PSScriptRoot
try{
    if($Tests){& $taskCargo test --release}
    elseif($Check){& $taskCargo check}
    else{& $taskCargo build --release}
    if($LASTEXITCODE -ne 0){throw "cargo failed with exit $LASTEXITCODE"}
}finally{Pop-Location}
