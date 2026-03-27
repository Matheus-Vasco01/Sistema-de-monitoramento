import os
import cv2
import time
import uuid
import sqlite3
import threading
from datetime import datetime
from collections import defaultdict

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from ultralytics import YOLO

# =========================
# CONFIGURAÇÕES DE CAMINHOS
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CAMERA_SOURCE = 0 # Use 0 para webcam ou "video.mp4" para arquivo
MODEL_PATH = os.path.join(BASE_DIR, "models", "yolo11n.pt")
SAVE_DIR = os.path.join(BASE_DIR, "static", "captures")
DB_PATH = os.path.join(BASE_DIR, "detections.db")

# Ajustes de Detecção
CONFIDENCE_THRESHOLD = 0.45
TARGET_CLASSES = {"person", "car", "motorcycle", "truck", "bus"}
ALERT_COOLDOWN_SECONDS = 20

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
# LOOP DA CÂMERA (THREAD)
# =========================
def process_stream():
    global last_frame
    cap = cv2.VideoCapture(CAMERA_SOURCE)
    
    while True:
        ok, frame = cap.read()
        if not ok:
            continue
        
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
        time.sleep(0.03)

@app.on_event("startup")
def startup_event():
    init_db()
    threading.Thread(target=process_stream, daemon=True).start()

# =========================
# ROTAS WEB
# =========================
@app.get("/", response_class=HTMLResponse)
def dashboard(request: Request):
    # Pegamos os eventos
    events_list = list_events()
    
    # Tentativa definitiva: passando apenas o dicionário de contexto
    # O FastAPI se encarrega de organizar para o Jinja2
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

@app.post("/chat")
async def chat_endpoint(request: Request):
    return JSONResponse({"answer": "O sistema está monitorando a propriedade. Qualquer movimento detectado aparecerá na lista de eventos."})