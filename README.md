# Intruder Alert
Sistema de vigilancia con detección de personas en tiempo real usando YOLOv8, cámara local y alertas por WhatsApp con Twilio.

## Características
- Detección de personas (clase `person`) con YOLOv8.
- Envío de alerta por WhatsApp al detectar intrusos.
- Cooldown configurable para evitar spam de mensajes.
- Guardado de snapshots con timestamp.
- Vista previa en ventana con cajas de detección.

## Requisitos
- Python 3.10+ (recomendado)
- Cámara conectada o webcam integrada
- Cuenta de Twilio con WhatsApp habilitado

## Instalación
1. Clona el repositorio:
   - `git clone https://github.com/victorbarrera1/intruder_alert.git`
   - `cd intruder_alert`
2. Crea y activa un entorno virtual:
   - `python3 -m venv venv`
   - `source venv/bin/activate`
3. Instala dependencias:
   - `pip install -r requirements.txt`

## Configuración (`.env`)
Crea un archivo `.env` en la raíz del proyecto con estas variables:

```env
TWILIO_ACCOUNT_SID=tu_sid
TWILIO_AUTH_TOKEN=tu_token
TWILIO_WHATSAPP_FROM=whatsapp:+14155238886
ALERT_WHATSAPP_TO=whatsapp:+569XXXXXXXX

CAMERA_INDEX=0
CONFIDENCE=0.5
COOLDOWN_SECONDS=30
SAVE_SNAPSHOTS=true
SNAPSHOT_DIR=snapshots
SHOW_PREVIEW=true
MODEL_SIZE=n
```

### Significado rápido de variables
- `CAMERA_INDEX`: índice de cámara (`0` suele ser la cámara principal).
- `CONFIDENCE`: umbral mínimo de confianza para detectar persona.
- `COOLDOWN_SECONDS`: tiempo mínimo entre alertas.
- `MODEL_SIZE`: `n`, `s`, `m` (más grande = más preciso, más lento).

## Uso
Con el entorno virtual activo:

- `python detector.py`

Al iniciar por primera vez, YOLO puede descargar automáticamente el modelo (`yolov8n.pt` o según `MODEL_SIZE`).

## Estructura del proyecto
- `detector.py`: lógica principal de detección y alertas.
- `requirements.txt`: dependencias Python.
- `snapshots/`: capturas guardadas cuando hay detecciones.
- `yolov8n.pt`: pesos del modelo (si ya fueron descargados).

## Notas
- En Mac con Apple Silicon, el script intenta usar `mps`; si no está disponible, usa `cpu`.
- Si usas Twilio Sandbox, revisa que el número destino esté unido al sandbox.
