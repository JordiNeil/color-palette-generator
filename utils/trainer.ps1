param (
    [string]$scriptPath,
    [string]$venvPath,
    [string]$requirementsFile,
    [string]$rootPath,
    [string]$kmin,
    [string]$kmax,
    [string]$imagePicker
)

# Check if virtual environment exists, if not, create it
if (-not (Test-Path $venvPath)) {
    python -m venv $venvPath
}

# Activate the virtual environment
$activateScript = Join-Path $venvPath "Scripts\Activate"
& $activateScript

# Install dependencies from requirements.txt
pip install -r $requirementsFile

"Executing parallel script... do not be afraid :D"
# Run the Python script and capture the output
$output = & python.exe $scriptPath $rootPath $kmin $kmax $imagePicker

# Output the captured output
Write-Output $output