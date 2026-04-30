# 📚 Documentação Técnica - AgroVision AI (com Integração YouTube)

Esta documentação detalha a arquitetura, o funcionamento interno e a integração de transmissões ao vivo via YouTube no sistema AgroVision AI.

---

## 🏗️ 1. Arquitetura do Sistema

O projeto é estruturado em torno de tecnologias modernas para visão computacional e inteligência artificial generativa, sendo dividido nos seguintes componentes principais:

*   **Backend & Servidor Web (FastAPI):** O núcleo da aplicação roda em Python usando o framework FastAPI. Ele gerencia as requisições HTTP, hospeda o servidor WebSocket/Streaming (para o feed de vídeo) e expõe a API para o dashboard e para o chat de IA.
*   **Visão Computacional (YOLO & OpenCV):** O modelo **YOLO11n** (da Ultralytics) é utilizado para a detecção de objetos em tempo real. O OpenCV (cv2) é responsável pela captura dos quadros da câmera, desenho das "bounding boxes" (caixas delimitadoras) em torno dos objetos e conversão dos frames em um stream (MJPEG) para a interface web.
*   **IA Conversacional Local (Ollama):** A aplicação integra um modelo de linguagem local (como o Llama 3) executado via Ollama. Ele atua como um assistente inteligente. Sempre que ocorre uma detecção de evento (ex: carro, caminhão), o sistema passa os metadados do último evento (como data, objeto e nível de confiança) para o prompt da IA, dando contexto local para as respostas.
*   **Armazenamento de Eventos (SQLite):** Detecções com grau de confiança (confidence threshold) acima do limite configurado acionam o salvamento do quadro detectado na pasta `/static/captures` e a criação de um registro estruturado no banco de dados SQLite (`detections.db`).

---

## 🔄 2. O Fluxo de Dados e Funcionamento

1.  **Captura (Thread de Vídeo):** Uma thread rodando em segundo plano (`process_stream`) fica responsável por puxar continuamente quadros da fonte de vídeo (Webcam, Câmera IP, ou Stream do YouTube).
2.  **Inferência:** A cada quadro recebido, ele é submetido ao modelo YOLO. Se o objeto pertencer às classes-alvo (ex: `car`, `person`, `truck`), a sua caixa e label são desenhados no quadro de saída.
3.  **Filtragem de Alertas:** Para evitar o registro excessivo (flood), há um controle de intervalo (`ALERT_COOLDOWN_SECONDS`) para que o mesmo tipo de objeto não gere múltiplas fotos de evento no mesmo minuto.
4.  **Interface de Usuário:** O frontend consome a rota `/video_feed`, que exibe visualmente o fluxo final (com as caixas da IA renderizadas). Na mesma tela, os cartões de "Últimos Eventos" são consumidos do SQLite e mostrados na página.

---

## 🎥 3. A Integração com o YouTube Ao Vivo

Para tornar o sistema mais dinâmico e capaz de observar cruzamentos contínuos de rodovias sem possuir uma câmera IP física, foi implementada uma ponte utilizando o script `start_yt_stream.py`.

### Como Funciona a Integração:

O OpenCV (`cv2.VideoCapture`) é capaz de processar streams através de links puros de protocolo HTTP (arquivos `.m3u8`), porém ele **não é capaz** de acessar uma página do YouTube diretamente e ler o player da página. 

Para contornar isso, o script executa as seguintes etapas:

1.  **Uso da biblioteca `yt-dlp`:** Utilizamos o pacote Python `yt-dlp` para analisar a página do YouTube da live alvo (por exemplo, a câmera 24 horas de Jackson Hole).
2.  **Extração Silenciosa:** O `yt-dlp` resolve os protocolos internos e a segurança do YouTube, devolvendo nativamente a URL direta da transmissão no formato HLS (`.m3u8`).
3.  **Injeção de Variável:** Essa URL gigantesca e temporária é extraída na memória e atribuída imediatamente à variável de ambiente de sistema `CAMERA_SOURCE`.
4.  **Início do Servidor:** O script, logo após configurar o ambiente local, aciona silenciosamente o Ollama e inicia o processo do **Uvicorn/FastAPI**.
5.  **Processamento Final:** No arquivo `app.py`, o OpenCV lê a variável `CAMERA_SOURCE` que foi previamente configurada com o link `.m3u8`. O OpenCV então conecta-se diretamente aos servidores do YouTube e passa a baixar o vídeo frame-a-frame da rodovia, passando-os para o YOLO detectar os carros.

Essa abordagem garante que, toda vez que a aplicação é iniciada, um link "fresco" do YouTube seja obtido, evitando problemas com links expirados do HLS.

---

## 🛠️ Tecnologias Envolvidas:
*   **Python 3.12**
*   **FastAPI / Uvicorn**
*   **Ultralytics (YOLO11)**
*   **OpenCV-Python**
*   **yt-dlp** (Buscador de Streams)
*   **Ollama** (IA LLM Local)
*   **SQLite** (Banco de dados de Eventos)
