$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

$backend = Start-Process -FilePath ".\\venv\\Scripts\\python.exe" `
  -ArgumentList "-m", "uvicorn", "backend_services.api:app", "--reload", "--app-dir", "." `
  -PassThru

try {
  & ".\\venv\\Scripts\\streamlit.exe" run "frontend\\streamlit_app.py"
}
finally {
  if ($backend -and !$backend.HasExited) {
    Stop-Process -Id $backend.Id -Force
  }
}
