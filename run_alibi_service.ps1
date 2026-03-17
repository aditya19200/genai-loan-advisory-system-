$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

& ".\\venv_alibi\\Scripts\\python.exe" -m uvicorn alibi_service.api:app --reload --app-dir .
