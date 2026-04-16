import os
import cv2
import time
import uuid
import sqlite3
import threading
import requests
import json
from datetime import datetime
from collections import defaultdict
from typing import List, Optional

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from ultralytics import YOLO

# =========================
# CONFIGURAÇÕES DE CAMINHOS
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Fonte da Câmera: 
# - Use 0 para webcam local
# - Use "rtsp://usuario:senha@ip:porta/stream" para câmeras IP/Segurança
# - Use "http://ip:porta/video" para streams HTTP (MJPEG)
CAMERA_SOURCE = os.getenv("CAMERA_SOURCE", "0")
if CAMERA_SOURCE.isdigit():
    CAMERA_SOURCE = int(CAMERA_SOURCE)

MODEL_PATH = os.path.join(BASE_DIR, "models", "yolo11n.pt")
SAVE_DIR = os.path.join(BASE_DIR, "static", "captures")
DB_PATH = os.path.join(BASE_DIR, "detections.db")

# Ajustes de Detecção
CONFIDENCE_THRESHOLD = 0.45
TARGET_CLASSES = {"person", "car", "motorcycle", "truck", "bus"}
ALERT_COOLDOWN_SECONDS = 20

# =========================
# CONFIGURAÇÕES OLLAMA
# =========================
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://127.0.0.1:11434/api/chat")
MODEL_NAME = os.getenv("OLLAMA_MODEL", "llama3")
OLLAMA_TIMEOUT = int(os.getenv("OLLAMA_TIMEOUT", "120"))
OLLAMA_KEEP_ALIVE = os.getenv("OLLAMA_KEEP_ALIVE", "30m")

# =========================
# MODELOS DE DADOS (PYDANTIC)
# =========================
class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    message: str
    history: List[Message] = []
    model: Optional[str] = "llama3"

class ChatResponse(BaseModel):
    answer: str
    history: List[Message]
    response_time: float

# =========================
# INICIALIZAÇÃO DO APP
# =========================
app = FastAPI(title="AgroVision AI")

os.makedirs(os.path.join(BASE_DIR, "static"), exist_ok=True)
os.makedirs(SAVE_DIR, exist_ok=True)

app.mount("/static", StaticFiles(directory=os.path.join(BASE_DIR, "static")), name="static")
templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "templates"))

model = YOLO(MODEL_PATH)
last_frame = None
last_frame_lock = threading.Lock()
last_alert_time = defaultdict(lambda: 0.0)

# =========================
# FUNÇÕES DE BANCO DE DADOS
# =========================
def init_db():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS events (
            id TEXT PRIMARY KEY,
            event_time TEXT,
            label TEXT,
            confidence REAL,
            image_path TEXT
        )
    """)
    conn.commit()
    conn.close()

def save_event(event_id, label, confidence, image_path):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("INSERT INTO events VALUES (?, ?, ?, ?, ?)", 
               (event_id, datetime.now().strftime("%Y-%m-%d %H:%M:%S"), label, confidence, image_path))
    conn.commit()
    conn.close()

def list_events(limit=15):
    if not os.path.exists(DB_PATH): return []
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT id, event_time, label, confidence, image_path FROM events ORDER BY event_time DESC LIMIT ?", (limit,))
    rows = cur.fetchall()
    conn.close()
    return [{"id": r[0], "event_time": r[1], "label": r[2], "confidence": r[3], "image_path": r[4]} for r in rows]

# =========================
# FUNÇÕES DO CHAT COM IA
# =========================
def get_last_event():
    if not os.path.exists(DB_PATH): return None
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    cur.execute("SELECT id, event_time, label, confidence, image_path FROM events ORDER BY event_time DESC LIMIT 1")
    row = cur.fetchone()
    conn.close()
    if row:
        return dict(row)
    return None

def build_chat_messages(message: str, history: List[Message]):
    system_message = {
        "role": "system",
        "content": "Você é um assistente do sistema AgroVision. Responda em português, de forma clara, objetiva e útil."
    }
    
    last_event = get_last_event()
    if last_event:
        event_ctx = (
            f"Último evento detectado:\n"
            f"- ID: {last_event['id']}\n"
            f"- Horário: {last_event['event_time']}\n"
            f"- Objeto: {last_event['label']}\n"
            f"- Confiança: {last_event['confidence']:.2f}\n"
            f"- Imagem: {last_event['image_path']}\n"
        )
        # Inserimos o contexto do evento no histórico como uma mensagem de sistema
        history = [Message(role="system", content=event_ctx)] + history

    # Combinamos tudo para enviar ao Ollama
    return [system_message] + [h.model_dump() for h in history] + [{"role": "user", "content": message}]

def ask_ollama(message: str, history: List[Message], model_name: str = MODEL_NAME):
    messages = build_chat_messages(message, history)
    payload = {
        "model": model_name,
        "messages": messages,
        "stream": False,
        "keep_alive": OLLAMA_KEEP_ALIVE
    }
    
    start_time = time.time()
    response = requests.post(OLLAMA_URL, json=payload, timeout=(10, OLLAMA_TIMEOUT))
    response.raise_for_status()
    end_time = time.time()
    
    data = response.json()
    return data.get("message", {}).get("content", ""), messages, (end_time - start_time)

# =========================
# LOOP DA CÂMERA (THREAD)
# =========================
def process_stream():
    global last_frame
    
    while True:
        print(f"Conectando à câmera: {CAMERA_SOURCE}...")
        cap = cv2.VideoCapture(CAMERA_SOURCE)
        
        # Otimização para streams remotos (RTSP/HTTP)
        if isinstance(CAMERA_SOURCE, str) and ("rtsp" in CAMERA_SOURCE or "http" in CAMERA_SOURCE):
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1) # Reduz o atraso

        while cap.isOpened():
            ok, frame = cap.read()
            if not ok:
                print("Conexão com a câmera perdida. Tentando reconectar...")
                break
            
            results = model(frame, conf=CONFIDENCE_THRESHOLD, verbose=False)

            for result in results:
                for box in result.boxes:
                    cls_id = int(box.cls[0].item())
                    label = model.names[cls_id]
                    
                    if label in TARGET_CLASSES:
                        conf = float(box.conf[0].item())
                        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                        
                        # Desenha Retângulo
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(frame, f"{label}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                        
                        # Lógica de Alerta/Gravação
                        if (time.time() - last_alert_time[label]) > ALERT_COOLDOWN_SECONDS:
                            event_id = str(uuid.uuid4())[:8]
                            filename = f"{event_id}.jpg"
                            cv2.imwrite(os.path.join(SAVE_DIR, filename), frame)
                            save_event(event_id, label, conf, f"/static/captures/{filename}")
                            last_alert_time[label] = time.time()

            with last_frame_lock:
                last_frame = frame.copy()
            
            # Pequeno delay para não sobrecarregar a CPU
            time.sleep(0.01)
            
        cap.release()
        time.sleep(5) # Espera 5 segundos antes de tentar reconectar

# =========================
# ROTAS DO CHAT
# =========================
@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    try:
        answer, messages, r_time = ask_ollama(req.message, req.history, req.model or MODEL_NAME)
        new_history = req.history + [Message(role="assistant", content=answer)]
        return ChatResponse(answer=answer, history=new_history, response_time=r_time)
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.post("/chat/stream")
async def chat_stream(req: ChatRequest):
    return StreamingResponse(
        stream_ollama_generator(req.message, req.history, req.model or MODEL_NAME),
        media_type="application/x-ndjson"
    )

async def stream_ollama_generator(message: str, history: List[Message], model_name: str):
    messages = build_chat_messages(message, history)
    payload = {
        "model": model_name,
        "messages": messages,
        "stream": True,
        "keep_alive": OLLAMA_KEEP_ALIVE
    }
    
    start_time = time.time()
    try:
        with requests.post(OLLAMA_URL, json=payload, stream=True, timeout=(10, OLLAMA_TIMEOUT)) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                if line:
                    chunk = json.loads(line.decode("utf-8"))
                    if "message" in chunk:
                        content = chunk["message"].get("content", "")
                        yield json.dumps({"content": content}) + "\n"
                    if chunk.get("done"):
                        end_time = time.time()
                        yield json.dumps({"response_time": end_time - start_time}) + "\n"
                        break
    except Exception as e:
        yield json.dumps({"error": str(e)}) + "\n"

@app.get("/health")
def health():
    ollama_status = "offline"
    try:
        base_url = OLLAMA_URL.rsplit("/api/", 1)[0]
        resp = requests.get(f"{base_url}/api/tags", timeout=5)
        if resp.status_code == 200:
            ollama_status = "online"
    except:
        pass
    return {"status": "ok", "ollama": ollama_status}

# =========================
# WARMUP OLLAMA
# =========================
def warmup_ollama():
    try:
        print(f"Iniciando warmup do Ollama com o modelo {MODEL_NAME}...")
        ask_ollama("Responda apenas: pronto", [])
        print(f"Ollama aquecido com sucesso.")
    except Exception as exc:
        print(f"Warmup do Ollama falhou: {exc}")

@app.on_event("startup")
def startup_event():
    init_db()
    threading.Thread(target=process_stream, daemon=True).start()
    threading.Thread(target=warmup_ollama, daemon=True).start()

# =========================
# ROTAS WEB
# =========================
@app.get("/", response_class=HTMLResponse)
def dashboard(request: Request):
    # Pegamos os eventos
    events_list = list_events()
    return templates.TemplateResponse(
        request, 
        "index.html", 
        {"events": events_list}
    )

@app.get("/frame")
def get_frame():
    with last_frame_lock:
        if last_frame is None: return Response(status_code=503)
        _, buffer = cv2.imencode(".jpg", last_frame)
        return Response(content=buffer.tobytes(), media_type="image/jpeg")

def generate_mjpeg_stream():
    while True:
        with last_frame_lock:
            if last_frame is None:
                time.sleep(0.1)
                continue
            _, buffer = cv2.imencode(".jpg", last_frame)
            frame_bytes = buffer.tobytes()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        time.sleep(0.05)

@app.get("/video_feed")
def video_feed():
    return StreamingResponse(
        generate_mjpeg_stream(), 
        media_type="multipart/x-mixed-replace; boundary=frame"
    )