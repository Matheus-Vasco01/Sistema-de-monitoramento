import yt_dlp
import os
import subprocess
import sys
import time

print("Buscando URL da transmissão ao vivo no YouTube...")

# Jackson Hole Town Square - Transmissão ao vivo 24/7 com muito tráfego de carros
url = "https://www.youtube.com/watch?v=1EiC9bvVGnk"

ydl_opts = {'format': 'best'}
try:
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=False)
        stream_url = info.get('url')
        if not stream_url:
            print("Não foi possível extrair a URL.")
            sys.exit(1)
        
        print("URL extraída com sucesso!")
        
        # Define a variável de ambiente para o FastAPI
        os.environ["CAMERA_SOURCE"] = stream_url
        
        # Garante que o Ollama está rodando
        print("Verificando se o Ollama está rodando...")
        try:
            import requests
            requests.get("http://127.0.0.1:11434/api/tags", timeout=2)
            print("Ollama já está ativo.")
        except:
            print("Iniciando Ollama em segundo plano...")
            subprocess.Popen(["ollama", "serve"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            time.sleep(3)

        # Inicia o servidor uvicorn
        print("Iniciando FastAPI (uvicorn)...")
        uvicorn_exe = os.path.join("..", "..", ".venv", "Scripts", "python.exe")
        
        # Executa o uvicorn passando as mesmas flags do run.ps1
        subprocess.run([uvicorn_exe, "-m", "uvicorn", "app:app", "--host", "127.0.0.1", "--port", "8000", "--reload"])

except Exception as e:
    print(f"Erro ao buscar stream do YouTube: {e}")
