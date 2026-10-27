param(
    [string]$NativeSource,
    [string]$BuildDir,
    [Parameter(Mandatory=$true)][string]$SdkInclude,
    [Parameter(Mandatory=$true)][string]$SdkLib,
    [Parameter(Mandatory=$true)][string]$SdkBin,
    [string]$VcVars,
    [string]$CMake,
    [string]$Ninja,
    [switch]$ConfigureOnly,
    [ValidateRange(1,2)][int]$Jobs=2
)
$ErrorActionPreference='Stop'
$taskRoot=Split-Path -Parent $PSScriptRoot
if(-not $NativeSource){$NativeSource=Join-Path $taskRoot 'evidence/cuda-route/single-output/native-src'}
if(-not $BuildDir){$BuildDir=Join-Path $taskRoot '.build/asr-single-output'}
if(-not $CMake){$CMake=(Get-Command cmake.exe -ErrorAction Stop).Source}
if(-not $Ninja){$Ninja=(Get-Command ninja.exe -ErrorAction Stop).Source}
if($VcVars){
    if(-not(Test-Path -LiteralPath $VcVars -PathType Leaf)){throw "Missing MSVC environment script: $VcVars"}
    & cmd.exe /d /c ('call "'+$VcVars+'" >nul && set') | ForEach-Object {
        if($_ -match '^([^=]+)=(.*)$'){[Environment]::SetEnvironmentVariable($matches[1],$matches[2],'Process')}
    }
    if($LASTEXITCODE -ne 0){throw "MSVC environment initialization failed: $LASTEXITCODE"}
}
$taskCompiler=Get-Command cl.exe -ErrorAction Stop
if(-not $env:VCToolsInstallDir){throw 'Use an MSVC developer shell, or pass -VcVars <vcvars64.bat>'}
$taskVc=[IO.Path]::GetFullPath($env:VCToolsInstallDir)
foreach($taskRequired in @($CMake,$Ninja,(Join-Path $NativeSource 'src/llama-context.cpp'),
    (Join-Path $SdkInclude 'um/Windows.h'),(Join-Path $SdkInclude 'ucrt/stdlib.h'),
    (Join-Path $SdkLib 'um/x64/kernel32.lib'),(Join-Path $SdkLib 'ucrt/x64/ucrt.lib'),(Join-Path $SdkBin 'rc.exe'))){
    if(-not(Test-Path -LiteralPath $taskRequired -PathType Leaf)){throw "Missing explicit build dependency: $taskRequired"}
}
$taskIncludes=@('ucrt','shared','um','winrt','cppwinrt') | ForEach-Object {Join-Path $SdkInclude $_}
$env:INCLUDE=(Join-Path $taskVc 'include')+';'+($taskIncludes -join ';')
$env:LIB=(Join-Path $SdkLib 'um/x64')+';'+(Join-Path $SdkLib 'ucrt/x64')+';'+(Join-Path $taskVc 'lib/x64')
$env:PATH=$SdkBin+';'+$env:PATH
$env:VSLANG='1033'
$taskArgs=@('-S',$NativeSource,'-B',$BuildDir,'-G','Ninja',
    "-DCMAKE_MAKE_PROGRAM=$Ninja",'-DCMAKE_BUILD_TYPE=Release',
    '-DBUILD_SHARED_LIBS=ON','-DGGML_BACKEND_DL=ON','-DGGML_NATIVE=OFF',
    '-DGGML_CPU=OFF','-DGGML_CUDA=OFF','-DGGML_VULKAN=OFF','-DGGML_OPENCL=OFF','-DGGML_SYCL=OFF',
    '-DLLAMA_BUILD_COMMON=OFF','-DLLAMA_BUILD_TESTS=OFF','-DLLAMA_BUILD_EXAMPLES=OFF',
    '-DLLAMA_BUILD_TOOLS=OFF','-DLLAMA_BUILD_SERVER=OFF')
& $CMake @taskArgs
if($LASTEXITCODE -ne 0){throw "Single-output CMake configuration failed: $LASTEXITCODE"}
if(-not $ConfigureOnly){
    & $CMake --build $BuildDir --target llama --parallel $Jobs
    if($LASTEXITCODE -ne 0){throw "Single-output llama target build failed: $LASTEXITCODE"}
}
