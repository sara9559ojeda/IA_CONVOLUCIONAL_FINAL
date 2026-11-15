"""Clasificación de eventos, niveles de alerta e historial.

La CNN analiza cada fotograma por separado. Este módulo mira los últimos
VENTANA_FRAMES fotogramas para:
  * confirmar una detección solo si se repite (evita falsas alarmas por un
    fotograma aislado),
  * clasificar el evento (humo blanco, humo negro, fuego incipiente, incendio
    declarado),
  * asignar un nivel de alerta y llevar un historial con hora y duración.
"""
from collections import deque
from dataclasses import dataclass, field

from . import config

# tipo -> (nombre para mostrar, nombre sin tildes para OpenCV, nivel de alerta)
TIPOS_EVENTO = {
    "sin_incidentes": ("Sin incidentes", "SIN INCIDENTES", "NORMAL"),
    "humo_blanco": ("Humo blanco (posible fase inicial de incendio)", "HUMO BLANCO", "PRECAUCION"),
    "humo_negro": ("Humo negro (combustión de materiales sintéticos o combustibles)", "HUMO NEGRO", "ALERTA"),
    "fuego_incipiente": ("Fuego incipiente", "FUEGO INCIPIENTE", "ALERTA"),
    "incendio_declarado": ("Incendio declarado", "INCENDIO DECLARADO", "EMERGENCIA"),
}


def pct(x):
    """0.8234 -> 82 (entero, para textos sin puntos decimales)."""
    return int(round(100 * x))


@dataclass
class EstadoAlerta:
    tipo: str = "sin_incidentes"
    conf_fuego: float = 0.0
    conf_humo: float = 0.0
    color_humo: str = ""
    area_fuego: float = 0.0
    area_humo: float = 0.0
    frames_fuego: int = 0
    frames_humo: int = 0
    ventana: int = 0
    tendencia: str = "estable"
    num_fuego: int = 0
    num_humo: int = 0
    razon: str = "no se detecta fuego ni humo"
    evento: dict = field(default_factory=dict)

    @property
    def nombre(self):
        return TIPOS_EVENTO[self.tipo][0]

    @property
    def nombre_corto(self):
        return TIPOS_EVENTO[self.tipo][1]

    @property
    def nivel(self):
        return TIPOS_EVENTO[self.tipo][2]


class MonitorEventos:
    def __init__(self, ventana=None, min_frames=None):
        self.ventana = ventana or config.VENTANA_FRAMES
        self.min_frames = min_frames or config.MIN_FRAMES_CONFIRMAR
        self.historial = deque(maxlen=self.ventana)
        self.eventos = []          # lista de dicts (el último puede estar abierto)
        self.tipo_actual = "sin_incidentes"

    def reiniciar(self):
        self.__init__(self.ventana, self.min_frames)

    # ------------------------------------------------------------------
    def actualizar(self, detecciones, t, marca, inmediato=False):
        """Procesa las detecciones de un fotograma.

        t        : tiempo en segundos (del video o del reloj) para calcular duraciones.
        marca    : texto para mostrar el inicio del evento (hora o minuto del video).
        inmediato: True para imágenes sueltas (confirma con un solo fotograma).
        """
        if inmediato:
            self.historial.clear()
        fuego = [d for d in detecciones if d.clase == "fuego"]
        humo = [d for d in detecciones if d.clase == "humo"]
        self.historial.append({
            "fuego": max((d.confianza for d in fuego), default=0.0),
            "humo": max((d.confianza for d in humo), default=0.0),
            "humo_negro": any(d.subtipo == "negro" for d in humo),
            "area_fuego": min(1.0, sum(d.area_relativa for d in fuego)),
            "area_humo": min(1.0, sum(d.area_relativa for d in humo)),
        })
        h = list(self.historial)
        minimo = 1 if inmediato else self.min_frames

        con_fuego = [r for r in h if r["fuego"] > 0]
        con_humo = [r for r in h if r["humo"] > 0]
        frames_negro = sum(r["humo_negro"] for r in con_humo)
        fuego_ok = len(con_fuego) >= minimo
        humo_ok = len(con_humo) >= minimo
        humo_negro = humo_ok and frames_negro * 2 >= len(con_humo)

        media = lambda filas, k: sum(r[k] for r in filas) / len(filas) if filas else 0.0
        estado = EstadoAlerta(
            conf_fuego=media(con_fuego, "fuego"), conf_humo=media(con_humo, "humo"),
            color_humo=("negro" if humo_negro else "blanco") if humo_ok else "",
            area_fuego=media(con_fuego, "area_fuego"), area_humo=media(con_humo, "area_humo"),
            frames_fuego=len(con_fuego), frames_humo=len(con_humo), ventana=len(h),
            num_fuego=len(fuego), num_humo=len(humo),
        )
        clave_area = "area_fuego" if fuego_ok else "area_humo"
        estado.tendencia = self._tendencia([r[clave_area] for r in h])

        # ---------------- Clasificación del evento ----------------
        if fuego_ok:
            grande = estado.area_fuego >= config.AREA_INCENDIO
            crece = (estado.tendencia == "creciendo" and
                     estado.area_fuego >= config.AREA_FUEGO_CRECIENDO)
            estado.tipo = "incendio_declarado" if (grande or crece or humo_negro) else "fuego_incipiente"
            estado.razon = (f"se detectó fuego en {len(con_fuego)} de los últimos {len(h)} fotogramas "
                            f"con una confianza media de {pct(estado.conf_fuego)} por ciento, "
                            f"ocupando cerca del {pct(estado.area_fuego)} por ciento de la imagen")
            if humo_ok:
                estado.razon += f", acompañado de humo {estado.color_humo}"
            if estado.tipo == "incendio_declarado":
                motivos = []
                if grande:
                    motivos.append("el fuego es extenso")
                if crece:
                    motivos.append("el fuego está creciendo")
                if humo_negro:
                    motivos.append("hay humo negro")
                estado.razon += " y se clasificó como incendio declarado porque " + " y ".join(motivos)
        elif humo_ok:
            estado.tipo = "humo_negro" if humo_negro else "humo_blanco"
            estado.razon = (f"se detectó humo {estado.color_humo} en {len(con_humo)} de los últimos "
                            f"{len(h)} fotogramas con una confianza media de "
                            f"{pct(estado.conf_humo)} por ciento, sin llamas visibles")
        else:
            estado.tipo = "sin_incidentes"

        estado.evento = self._registrar(estado, t, marca)
        return estado

    # ------------------------------------------------------------------
    @staticmethod
    def _tendencia(areas):
        if len(areas) < 4:
            return "estable"
        mitad = len(areas) // 2
        antes = sum(areas[:mitad]) / mitad
        despues = sum(areas[mitad:]) / (len(areas) - mitad)
        if despues > antes * 1.3 + 1e-3:
            return "creciendo"
        if despues < antes * 0.7:
            return "disminuyendo"
        return "estable"

    def _registrar(self, estado, t, marca):
        """Abre/cierra eventos en el historial cuando cambia el tipo."""
        abierto = self.eventos[-1] if self.eventos and self.eventos[-1]["_abierto"] else None
        if estado.tipo != self.tipo_actual:
            if abierto:
                abierto["_abierto"] = False
                abierto["estado"] = "finalizado"
                abierto = None
            if estado.tipo != "sin_incidentes":
                abierto = {
                    "id": len(self.eventos) + 1, "tipo": estado.nombre, "nivel": estado.nivel,
                    "inicio": marca, "duracion_s": 0, "conf_max_%": 0, "area_max_%": 0,
                    "estado": "en curso", "_abierto": True, "_t0": t, "_clave": estado.tipo,
                }
                self.eventos.append(abierto)
            self.tipo_actual = estado.tipo
        if abierto:
            abierto["duracion_s"] = int(round(t - abierto["_t0"]))
            abierto["conf_max_%"] = max(abierto["conf_max_%"],
                                        pct(max(estado.conf_fuego, estado.conf_humo)))
            abierto["area_max_%"] = max(abierto["area_max_%"],
                                        pct(max(estado.area_fuego, estado.area_humo)))
            return abierto
        return {}

    def tabla_eventos(self):
        """Eventos sin las claves internas (para mostrar o guardar en CSV)."""
        return [{k: v for k, v in e.items() if not k.startswith("_")} for e in self.eventos]


def resumen_para_asistente(estado, eventos):
    """Convierte la predicción de la CNN en texto: es el CONTEXTO del transformer.

    Se usan oraciones completas y porcentajes en palabras ("82 por ciento")
    para que el modelo de pregunta-respuesta encuentre respuestas limpias.
    """
    if estado.tipo == "sin_incidentes":
        partes = ["Actualmente el sistema no detecta fuego ni humo.",
                  "El nivel de alerta actual es NORMAL y no se generó ninguna alerta."]
    else:
        partes = [f"La red convolucional detectó el evento {estado.nombre}.",
                  f"El nivel de alerta actual es {estado.nivel}.",
                  f"La alerta se generó porque {estado.razon}."]
        if estado.frames_fuego:
            partes.append(f"La confianza de la detección de fuego es del {pct(estado.conf_fuego)} por ciento.")
            partes.append(f"El fuego ocupa aproximadamente el {pct(estado.area_fuego)} por ciento de la imagen.")
        if estado.color_humo:
            partes.append(f"El humo detectado es de color {estado.color_humo} y su confianza es del "
                          f"{pct(estado.conf_humo)} por ciento.")
        partes.append(f"La tendencia del evento es {estado.tendencia}.")
        if estado.evento:
            partes.append(f"El evento actual comenzó en {estado.evento['inicio']} y lleva "
                          f"{estado.evento['duracion_s']} segundos.")
    cerrados = [e for e in eventos if not e.get("_abierto")]
    partes.append(f"Durante la sesión se han registrado {len(eventos)} eventos en total.")
    for e in cerrados[-5:]:
        partes.append(f"El evento número {e['id']} fue {e['tipo']} con nivel {e['nivel']}, "
                      f"comenzó en {e['inicio']} y duró {e['duracion_s']} segundos.")
    return " ".join(partes)
