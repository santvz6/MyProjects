import logging
from datetime import datetime
from pathlib import Path
from config import LOGS_DIR, NUM_LOGS # Es importante utilizar ..

# Aseguramos que LOGS_DIR sea un objeto Path
logs_path = Path(LOGS_DIR)
logs_path.mkdir(parents=True, exist_ok=True)

# --- ROTACIÓN DE LOGS (PATHLIB) ---
# Listamos archivos .log, los ordenamos por tiempo de modificación
existing_logs = sorted(
    logs_path.glob("*.log"), 
    key=lambda x: x.stat().st_mtime
)

# Eliminamos los más antiguos si superamos NUM_LOGS
while len(existing_logs) >= NUM_LOGS:
    oldest_log = existing_logs.pop(0)
    oldest_log.unlink()  # Elimina el archivo de forma segura

# --- CONFIGURACIÓN DE RUTA DEL NUEVO LOG ---
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = logs_path / f"simulacion_{timestamp}.log"

# --- CONFIGURACIÓN DEL LOGGER ---
logger = logging.getLogger("app_logger")  
logger.setLevel(logging.DEBUG)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

if logger.hasHandlers():
    logger.handlers.clear()

# Handler para consola
console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

# Handler para archivo
file_handler = logging.FileHandler(log_file) 
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)