# 🚀 Guia de Execução - AgroVision AI

Este documento explica como rodar o sistema AgroVision AI no seu computador.

## 📋 Pré-requisitos

1.  **Ollama**: Instalado e rodando ([baixar aqui](https://ollama.com/download)).
2.  **Modelo Llama 3**: Baixado no Ollama.
    *   No terminal: `ollama pull llama3`
3.  **Python 3.12.7**: Instalado (com a opção "Add Python to PATH" marcada).

---

## 🛠️ Como Iniciar o Sistema

### **Passo 1: Abrir o Terminal**
Abra o terminal do VS Code (`Ctrl + '`) e garanta que você está na pasta do aplicativo:
```powershell
cd "c:\Users\Mathe\OneDrive\Documents\Materia_IA\Sistema-de-monitoramento\agrovision_ia\app"
```

### **Passo 2: Rodar o Script de Automação**
Use o comando abaixo para iniciar o Ollama (se estiver fechado) e o servidor do AgroVision:

**Para usar a Webcam Local:**
```powershell
powershell -ExecutionPolicy Bypass -File .\run.ps1 -Foreground -CameraSource "0"
```

**Para usar uma Câmera Externa (IP/RTSP):**
```powershell
powershell -ExecutionPolicy Bypass -File .\run.ps1 -Foreground -CameraSource "rtsp://usuario:senha@ip_da_camera:554/stream"
```

---

## 🌐 Acessando o Dashboard

Após rodar o comando, o terminal mostrará:
`INFO: Uvicorn running on http://127.0.0.1:8000`

Abra no seu navegador:
👉 **[http://127.0.0.1:8000](http://127.0.0.1:8000)**

---

## 💡 Dicas de Uso

*   **Status do Ollama**: No topo do dashboard, verifique se aparece **"Ollama: Online"**. Se estiver offline, verifique se o Ollama está aberto no seu Windows.
*   **Chat com IA**: Você pode escolher entre diferentes modelos (Llama 3, Mistral) no seletor do topo.
*   **Histórico**: O chat mantém o contexto da conversa. Use o botão **"Limpar"** para reiniciar o diálogo.
*   **Eventos**: As detecções do YOLO aparecerão automaticamente em formato de cards com fotos.

---

## ⚠️ Solução de Problemas

*   **Erro de Módulo não encontrado**: Rode `pip install -r ..\..\requirements.txt` com o ambiente virtual ativado.
*   **Câmera não abre**: Verifique se outro aplicativo (Teams, Zoom) está usando a webcam.
*   **Ollama Lento**: Na primeira pergunta, o sistema faz um "warmup". As próximas respostas serão mais rápidas.

---
*AgroVision AI - Monitoramento Inteligente para o Campo* 🚜🌱
