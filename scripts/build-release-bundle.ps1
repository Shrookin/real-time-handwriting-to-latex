param(
  [Parameter(Mandatory = $true)]
  [string]$OutputDirectory
)

$ErrorActionPreference = 'Stop'
$repositoryRoot = Split-Path -Parent $PSScriptRoot
$outputRoot = [System.IO.Path]::GetFullPath($OutputDirectory)
$bundleName = 'handwriting-to-latex-v1'
$bundleRoot = Join-Path $outputRoot $bundleName
$zipPath = Join-Path $outputRoot "$bundleName.zip"

if (Test-Path -LiteralPath $bundleRoot) {
  throw "Bundle directory already exists: $bundleRoot"
}
if (Test-Path -LiteralPath $zipPath) {
  throw "Bundle ZIP already exists: $zipPath"
}

New-Item -ItemType Directory -Force -Path $bundleRoot | Out-Null
New-Item -ItemType Directory -Force -Path (Join-Path $bundleRoot 'model') | Out-Null
New-Item -ItemType Directory -Force -Path (Join-Path $bundleRoot 'examples') | Out-Null
New-Item -ItemType Directory -Force -Path (Join-Path $bundleRoot 'docs') | Out-Null

$rootFiles = @('README.md', 'LICENSE', 'NOTICE', 'requirements.txt')
foreach ($file in $rootFiles) {
  Copy-Item (Join-Path $repositoryRoot $file) (Join-Path $bundleRoot $file)
}
Copy-Item (Join-Path $repositoryRoot 'model\*') (Join-Path $bundleRoot 'model')
Copy-Item (Join-Path $repositoryRoot 'examples\run_onnx.py') (Join-Path $bundleRoot 'examples\run_onnx.py')
Copy-Item (Join-Path $repositoryRoot 'examples\sample_request.json') (Join-Path $bundleRoot 'examples\sample_request.json')
foreach ($file in @('DOWNLOADS.md', 'INTEGRATION.md', 'MODEL_CARD.md', 'ATTRIBUTION_AND_LICENSE.md')) {
  Copy-Item (Join-Path $repositoryRoot "docs\\$file") (Join-Path $bundleRoot "docs\\$file")
}

Compress-Archive -Path $bundleRoot -DestinationPath $zipPath
Get-FileHash -Algorithm SHA256 $zipPath | Format-List Path, Hash
