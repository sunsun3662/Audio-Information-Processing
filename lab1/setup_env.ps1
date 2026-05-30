param(
    [string]$EnvName = "speech_env",
    [string]$PythonVersion = "3.8",
    [string]$RequirementsPath = "requirements.txt"
)

$ErrorActionPreference = "Stop"

Write-Host "[1/5] Checking conda..."
$condaCmd = Get-Command conda -ErrorAction SilentlyContinue
if (-not $condaCmd) {
    throw "Conda not found in PATH. Open Anaconda Prompt or add conda to PATH first."
}

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$reqPath = Join-Path $scriptDir $RequirementsPath
if (-not (Test-Path $reqPath)) {
    throw "requirements file not found: $reqPath"
}

Write-Host "[2/5] Checking whether env '$EnvName' exists..."
$envListJson = conda env list --json | Out-String
$envData = $envListJson | ConvertFrom-Json
$envExists = $false
foreach ($p in $envData.envs) {
    if ((Split-Path $p -Leaf) -eq $EnvName) {
        $envExists = $true
        break
    }
}

if (-not $envExists) {
    Write-Host "[3/5] Creating env '$EnvName' with Python $PythonVersion..."
    conda create -n $EnvName python=$PythonVersion -y
} else {
    Write-Host "[3/5] Env '$EnvName' already exists, skip create."
}

Write-Host "[4/5] Installing dependencies from $reqPath ..."
conda run -n $EnvName python -m pip install --upgrade pip
conda run -n $EnvName python -m pip install -r $reqPath

Write-Host "[5/5] Done."
Write-Host "Use this kernel/interpreter:"
Write-Host "  conda run -n $EnvName python -m ipykernel install --user --name $EnvName --display-name '$EnvName (Python $PythonVersion)'"
Write-Host "Or run scripts with:"
Write-Host "  conda run -n $EnvName python your_script.py"
