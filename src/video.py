"""Utilidades de video para la app: salto de fotogramas y videos temporales.

  * Salto de fotogramas: en CPU la red tarda más en analizar un fotograma de lo
    que ese fotograma dura en el video. Si se analizan todos, el video se ve en
    cámara lenta y la cámara acumula retraso. `frames_a_saltar` calcula cuántos
    hay que descartar para seguir el ritmo real y `descartar_frames` los
    descarta sin decodificarlos a imagen.
  * Videos temporales: OpenCV necesita una ruta en disco, así que el video que
    se sube a la app se copia a una carpeta temporal propia. Aquí se crea, se
    borra cuando deja de usarse y se limpian los que quedaron de sesiones
    anteriores (por ejemplo, si se cerró el navegador).
"""
import os
import tempfile
import time
from pathlib import Path

DIR_VIDEOS_TEMP = Path(tempfile.gettempdir()) / "deteccion_incendios_videos"
PREFIJO_VIDEO = "video_"
MAX_SALTO = 30            # nunca se descartan más fotogramas seguidos que estos
HORAS_VIDEO_ANTIGUO = 6   # un temporal más viejo que esto es de una sesión abandonada


# ==========================================================================
# Salto de fotogramas
# ==========================================================================
def frames_a_saltar(segundos_ciclo, fps, maximo=MAX_SALTO):
    """Fotogramas que hay que descartar después de analizar uno para ir en tiempo real.

    Si el último ciclo (leer + analizar + dibujar) tardó `segundos_ciclo` y la
    fuente entrega `fps` fotogramas por segundo, en ese tiempo pasaron
    `segundos_ciclo * fps` fotogramas: uno es el analizado y el resto sobra.
    """
    if not fps or fps <= 0 or segundos_ciclo <= 0:
        return 0
    return int(min(maximo, max(0, round(segundos_ciclo * fps) - 1)))


def descartar_frames(cap, n, limite_s=None):
    """Descarta hasta `n` fotogramas con `cap.grab()` y retorna cuántos descartó.

    limite_s: solo para cámaras. Si un `grab` tarda más que esto, el búfer ya
    estaba vacío (se esperó un fotograma nuevo) y no tiene sentido seguir.
    """
    descartados = 0
    for _ in range(max(0, int(n))):
        t0 = time.perf_counter()
        if not cap.grab():          # fin del video o cámara desconectada
            break
        descartados += 1
        if limite_s is not None and time.perf_counter() - t0 > limite_s:
            break
    return descartados


# ==========================================================================
# Videos temporales
# ==========================================================================
def guardar_video_temporal(datos, sufijo=".mp4", dir_temp=None):
    """Copia el video subido a la carpeta temporal y retorna su ruta."""
    dir_temp = Path(dir_temp or DIR_VIDEOS_TEMP)
    dir_temp.mkdir(parents=True, exist_ok=True)
    fd, ruta = tempfile.mkstemp(prefix=PREFIJO_VIDEO, suffix=sufijo, dir=dir_temp)
    with os.fdopen(fd, "wb") as f:
        f.write(datos)
    return ruta


def borrar_video_temporal(ruta):
    """Borra un video temporal. Retorna True si el archivo ya no existe."""
    if not ruta:
        return True
    try:
        os.remove(ruta)
    except FileNotFoundError:
        return True
    except OSError:                 # en Windows falla si el video sigue abierto
        return False
    return True


def limpiar_videos_antiguos(horas=HORAS_VIDEO_ANTIGUO, dir_temp=None):
    """Borra los videos temporales de sesiones anteriores. Retorna cuántos borró.

    Solo toca archivos `video_*` de la carpeta temporal propia de la app.
    """
    dir_temp = Path(dir_temp or DIR_VIDEOS_TEMP)
    if not dir_temp.is_dir():
        return 0
    limite = time.time() - horas * 3600
    borrados = 0
    for archivo in dir_temp.glob(PREFIJO_VIDEO + "*"):
        try:
            if archivo.is_file() and archivo.stat().st_mtime < limite:
                archivo.unlink()
                borrados += 1
        except OSError:
            continue
    return borrados
