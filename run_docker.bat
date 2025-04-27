@echo off
REM Build and run LexiFocus in Docker

REM Set your OpenAI API key here or pass it as an argument
set OPENAI_API_KEY=your-openai-api-key

REM Build the Docker image
docker build -f Dockerfile -t lexifocus .

REM Create persistent data directories if they don't exist
if not exist "%cd%\database" mkdir "%cd%\database"
if not exist "%cd%\data" mkdir "%cd%\data"

REM Run the container, mapping ports and mounting volumes for persistence
docker run -p 8501:8501 ^
  -e OPENAI_API_KEY=%OPENAI_API_KEY% ^
  -v "%cd%\database:/app/database" ^
  -v "%cd%\data:/app/data" ^
  lexifocus

pause