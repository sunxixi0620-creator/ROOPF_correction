param([Parameter(Mandatory=$true)][string]$RequestFile, [int]$TimeoutSeconds = 90)
$ErrorActionPreference = 'Stop'
[Console]::OutputEncoding = New-Object System.Text.UTF8Encoding($false)
$keyPath = '\\wsl.localhost\Ubuntu\home\skq\.config\roopf_agent\worldcodes.env'
$keyLine = [System.IO.File]::ReadAllText($keyPath).Trim()
if (-not $keyLine.StartsWith('WORLDCODES_API_KEY=')) { throw 'Invalid key file format' }
$apiKey = $keyLine.Substring('WORLDCODES_API_KEY='.Length)
$body = [System.IO.File]::ReadAllText($RequestFile)
try {
    $response = Invoke-WebRequest -UseBasicParsing -Uri 'https://worldcodes.online/v1/chat/completions' -Headers @{ Authorization = "Bearer $apiKey" } -ContentType 'application/json; charset=utf-8' -Method Post -Body ([System.Text.Encoding]::UTF8.GetBytes($body)) -TimeoutSec $TimeoutSeconds
    [System.IO.File]::WriteAllText(($RequestFile + '.response.raw'), $response.Content, (New-Object System.Text.UTF8Encoding($false)))
    [Console]::Write($response.Content)
} catch {
    [Console]::Error.Write('WorldCodes request failed: ' + $_.Exception.Message)
    exit 1
} finally {
    $apiKey = $null
    $keyLine = $null
}
