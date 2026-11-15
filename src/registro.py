"""Registro en disco de las alertas: una captura y una fila de CSV por alerta.

El historial de la app vive en la memoria de la sesión, así que se pierde al
reiniciarla o al cerrar el navegador. Este módulo deja evidencia permanente en
`resultados/alertas/`:

    alertas.csv                                      una fila por alerta
    <sesion>_evento001_fuego_incipiente.jpg          fotograma anotado del
                                                     momento en que se confirmó

La fila se escribe cuando empieza la alerta y se actualiza mientras sigue
abierta y cuando termina (duración, confianza máxima y área máxima).
"""
import csv
import os
import threading
import time
import uuid
from datetime import datetime
from pathlib import Path

import cv2

from . import config

COLUMNAS = ["sesion", "fecha", "hora", "fuente", "id", "tipo", "nivel", "inicio",
            "duracion_s", "conf_max_%", "area_max_%", "estado", "captura"]
_DATOS_EVENTO = ["tipo", "nivel", "inicio", "duracion_s", "conf_max_%", "area_max_%", "estado"]
_CANDADO = threading.Lock()     # varias pestañas de la app comparten el mismo CSV


def leer_alertas(ruta_csv):
    """Lee `alertas.csv` como lista de diccionarios (vacía si aún no existe)."""
    try:
        with open(ruta_csv, newline="", encoding="utf-8-sig") as f:
            return list(csv.DictReader(f))
    except OSError:
        return []


class RegistroAlertas:
    def __init__(self, dir_salida=None, intervalo_s=5.0):
        self.dir = Path(dir_salida or config.DIR_ALERTAS)
        self.ruta_csv = self.dir / "alertas.csv"
        # Identifica las filas de esta sesión: al reiniciar, los eventos vuelven
        # a numerarse desde 1 y no deben pisar los de una sesión anterior.
        self.sesion = f"{datetime.now():%Y%m%d_%H%M%S}_{uuid.uuid4().hex[:4]}"
        self.intervalo_s = intervalo_s      # cada cuánto se actualiza una alerta abierta
        self._filas = {}                    # id del evento -> fila del CSV
        self._pendiente = False
        self._ultima_escritura = 0.0

    def actualizar(self, eventos, imagen_bgr=None, fuente="", forzar=False):
        """Sincroniza el registro con `MonitorEventos.eventos`.

        imagen_bgr: fotograma anotado actual; se guarda como captura de las
                    alertas que aparecen por primera vez.
        forzar    : escribe ya los cambios pendientes (al terminar el video).
        Retorna True si escribió en disco.
        """
        urgente = False
        for e in eventos:
            fila = self._filas.get(e["id"])
            if fila is None:                                    # alerta nueva
                ahora = datetime.now()
                fila = {"sesion": self.sesion, "fecha": f"{ahora:%Y-%m-%d}",
                        "hora": f"{ahora:%H:%M:%S}", "fuente": fuente, "id": e["id"],
                        "captura": self._guardar_captura(e, imagen_bgr)}
                self._filas[e["id"]] = fila
                urgente = True
            datos = {k: e[k] for k in _DATOS_EVENTO}
            if any(fila.get(k) != v for k, v in datos.items()):
                urgente |= fila.get("estado") not in (None, datos["estado"])   # se cerró
                fila.update(datos)
                self._pendiente = True

        vencido = time.monotonic() - self._ultima_escritura >= self.intervalo_s
        if self._pendiente and (urgente or forzar or vencido):
            return self._escribir()
        return False

    def leer(self):
        """Todas las alertas guardadas en disco (de esta sesión y de las anteriores)."""
        return leer_alertas(self.ruta_csv)

    # ------------------------------------------------------------------
    def _guardar_captura(self, evento, imagen_bgr):
        if imagen_bgr is None:
            return ""
        nombre = f"{self.sesion}_evento{int(evento['id']):03d}_{evento.get('_clave', 'alerta')}.jpg"
        try:
            ok, datos = cv2.imencode(".jpg", imagen_bgr, [cv2.IMWRITE_JPEG_QUALITY, 90])
            if not ok:
                return ""
            self.dir.mkdir(parents=True, exist_ok=True)
            datos.tofile(str(self.dir / nombre))    # a diferencia de imwrite, admite tildes en la ruta
        except (OSError, cv2.error):
            return ""
        return nombre

    def _escribir(self):
        """Reescribe el CSV: filas de otras sesiones + filas de esta."""
        temporal = self.ruta_csv.with_name(f"alertas_{self.sesion}.tmp")
        try:
            with _CANDADO:
                self.dir.mkdir(parents=True, exist_ok=True)
                otras = [f for f in leer_alertas(self.ruta_csv) if f.get("sesion") != self.sesion]
                with open(temporal, "w", newline="", encoding="utf-8-sig") as f:
                    escritor = csv.DictWriter(f, fieldnames=COLUMNAS, extrasaction="ignore")
                    escritor.writeheader()
                    escritor.writerows(otras)
                    escritor.writerows(self._filas.values())
                os.replace(temporal, self.ruta_csv)
        except OSError:
            # Por ejemplo, el CSV está abierto en Excel. La detección no se
            # detiene por esto: los cambios quedan pendientes y se reintenta.
            try:
                os.remove(temporal)
            except OSError:
                pass
            self._ultima_escritura = time.monotonic()   # no reintentar en cada fotograma
            return False
        self._pendiente = False
        self._ultima_escritura = time.monotonic()
        return True
