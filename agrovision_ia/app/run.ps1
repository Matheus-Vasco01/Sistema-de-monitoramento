param (
    [switch]$Foreground,
    [string]$CameraSource = "0"
)

$env:CAMERA_SOURCE = $CameraSource

$uvicornExe = "..\..\.venv\Scripts\python.exe"
$appPath = "app:app"

Write-Host "Verificando se o Ollama está rodando..." -ForegroundColor Cyan
try {
    $response = Invoke-WebRequest -UseBasicParsing http://127.0.0.1:11434/api/tags -ErrorAction SilentlyContinue
    Write-Host "Ollama já está ativo." -ForegroundColor Green
} catch {
    Write-Host "Iniciando Ollama..." -ForegroundColor Yellow
    $startedOllamaProcess = Start-Process "ollama" -ArgumentList "serve" -PassThru -NoNewWindow
    Start-Sleep -Seconds 5
}

if ($Foreground) {
    Write-Host "Iniciando FastAPI (uvicorn) em foreground..." -ForegroundColor Cyan
    Write-Host "Use Ctrl + C para encerrar o servidor." -ForegroundColor Yellow
    try {
        & $uvicornExe -m uvicorn $appPath --host 127.0.0.1 --port 8000 --reload
    } finally {
        if ($startedOllamaProcess -and -not $startedOllamaProcess.HasExited) {
            Stop-Process -Id $startedOllamaProcess.Id -Force -ErrorAction SilentlyContinue
        }
    }
} else {
    Write-Host "Iniciando FastAPI (uvicorn) em background..." -ForegroundColor Cyan
    Start-Process $uvicornExe -ArgumentList "-m uvicorn $appPath --host 127.0.0.1 --port 8000 --reload" -NoNewWindow
    Write-Host "Servidor rodando em http://127.0.0.1:8000" -ForegroundColor Green
}
