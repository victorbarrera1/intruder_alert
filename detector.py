"""
intruder_alert/detector.py
Detecta personas en cámara y envía alerta por WhatsApp (Twilio).
"""

import cv2
import time
import os
import logging
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv
from ultralytics import YOLO
from twilio.rest import Client

# ─── Config ────────────────────────────────────────────────────────────────────
load_dotenv()

TWILIO_SID        = os.getenv("TWILIO_ACCOUNT_SID")
TWILIO_TOKEN      = os.getenv("TWILIO_AUTH_TOKEN")
TWILIO_FROM       = os.getenv("TWILIO_WHATSAPP_FROM")   # "whatsapp:+14155238886"
ALERT_TO          = os.getenv("ALERT_WHATSAPP_TO")       # "whatsapp:+569XXXXXXXX"

CAMERA_INDEX      = int(os.getenv("CAMERA_INDEX", "0"))
CONFIDENCE        = float(os.getenv("CONFIDENCE", "0.5"))
COOLDOWN_SECONDS  = int(os.getenv("COOLDOWN_SECONDS", "30"))
SAVE_SNAPSHOTS    = os.getenv("SAVE_SNAPSHOTS", "true").lower() == "true"
SNAPSHOT_DIR      = Path(os.getenv("SNAPSHOT_DIR", "snapshots"))
SHOW_PREVIEW      = os.getenv("SHOW_PREVIEW", "true").lower() == "true"
MODEL_SIZE        = os.getenv("MODEL_SIZE", "n")   # n=nano | s=small | m=medium

# ─── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("intruder_alert")

# ─── Setup ─────────────────────────────────────────────────────────────────────
if SAVE_SNAPSHOTS:
    SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)


def send_whatsapp_alert(snapshot_path: Path | None = None) -> bool:
    """Envía mensaje WhatsApp via Twilio. Retorna True si fue exitoso."""
    if not all([TWILIO_SID, TWILIO_TOKEN, TWILIO_FROM, ALERT_TO]):
        log.error("Faltan variables de Twilio en .env — revisa la configuración.")
        return False

    try:
        client = Client(TWILIO_SID, TWILIO_TOKEN)
        ts = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
        body = f"🚨 *ALERTA DE INTRUSO*\n\nSe detectó una persona en el perímetro.\n🕐 {ts}"

        kwargs = dict(from_=TWILIO_FROM, to=ALERT_TO, body=body)

        # Twilio puede enviar imagen si tienes cuenta upgraded (no sandbox)
        # if snapshot_path:
        #     kwargs["media_url"] = [f"https://tu-servidor/{snapshot_path.name}"]

        msg = client.messages.create(**kwargs)
        log.info(f"WhatsApp enviado ✅  SID: {msg.sid}")
        return True

    except Exception as e:
        log.error(f"Error enviando WhatsApp: {e}")
        return False


def save_snapshot(frame) -> Path | None:
    """Guarda captura con timestamp."""
    if not SAVE_SNAPSHOTS:
        return None
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = SNAPSHOT_DIR / f"intruso_{ts}.jpg"
    cv2.imwrite(str(path), frame)
    log.info(f"Snapshot guardado: {path}")
    return path


def draw_detections(frame, results):
    """Dibuja bounding boxes de personas detectadas."""
    for box in results[0].boxes:
        cls = int(box.cls[0])
        conf = float(box.conf[0])
        if cls != 0 or conf < CONFIDENCE:   # clase 0 = person en COCO
            continue
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(
            frame,
            f"PERSONA {conf:.0%}",
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 255),
            2,
        )
    return frame


def main():
    log.info(f"Cargando YOLOv8{MODEL_SIZE}...")
    model = YOLO(f"yolov8{MODEL_SIZE}.pt")   # se descarga automáticamente la primera vez

    log.info(f"Abriendo cámara {CAMERA_INDEX}...")
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        log.error(f"No se pudo abrir la cámara {CAMERA_INDEX}.")
        return

    last_alert_time = 0
    log.info("✅ Sistema activo. Presiona Q para salir.")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                log.warning("No se pudo leer frame. Reintentando...")
                time.sleep(0.5)
                continue

            # Inferencia (device='mps' en Apple Silicon, 'cpu' en Intel Mac)
            results = model(frame, verbose=False, device="mps" if _has_mps() else "cpu")

            # Filtrar solo personas con confianza suficiente
            persons = [
                b for b in results[0].boxes
                if int(b.cls[0]) == 0 and float(b.conf[0]) >= CONFIDENCE
            ]

            now = time.time()
            if persons and (now - last_alert_time) > COOLDOWN_SECONDS:
                log.warning(f"🚨 {len(persons)} persona(s) detectada(s)!")
                snapshot = save_snapshot(frame)
                send_whatsapp_alert(snapshot)
                last_alert_time = now

            if SHOW_PREVIEW:
                frame = draw_detections(frame, results)
                status = f"ALERTA: {len(persons)} persona(s)" if persons else "Monitoreando..."
                color  = (0, 0, 255) if persons else (0, 200, 0)
                cv2.putText(frame, status, (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                cv2.imshow("Intruder Alert — Q para salir", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

    except KeyboardInterrupt:
        log.info("Detenido por usuario.")
    finally:
        cap.release()
        if SHOW_PREVIEW:
            cv2.destroyAllWindows()
        log.info("Sistema apagado.")


def _has_mps() -> bool:
    """Detecta si hay Apple Silicon MPS disponible."""
    try:
        import torch
        return torch.backends.mps.is_available()
    except Exception:
        return False


if __name__ == "__main__":
    main()
