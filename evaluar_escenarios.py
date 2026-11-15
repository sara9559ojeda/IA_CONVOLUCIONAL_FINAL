"""Evalúa el sistema en distintos escenarios (videos) y genera tablas para el reporte.

Ejemplos:
    python evaluar_escenarios.py videos/interior.mp4 videos/bosque.mp4
    python evaluar_escenarios.py videos/                # todos los videos de la carpeta
    python evaluar_escenarios.py videos/ --sin-video    # no guarda los videos anotados

Por cada video se guarda en `resultados/`:
    <nombre>_anotado.mp4    video con cajas y alertas
    <nombre>_frames.csv     detección fotograma a fotograma
    <nombre>_eventos.csv    eventos clasificados (tipo, nivel, inicio, duración)
y un `resumen_escenarios.csv` con una fila por escenario:
    % de fotogramas con fuego/humo, tiempo hasta la primera alerta,
    nivel máximo alcanzado, número de eventos y velocidad (FPS).

Para medir falsas alarmas incluye videos SIN fuego (atardeceres, luces rojas,
niebla, cocina, etc.): en ellos lo ideal es 0 eventos.
"""
import argparse
import time
from pathlib import Path

import cv2
import pandas as pd

from src import config
from src.detector import DetectorFuegoHumo, dibujar
from src.eventos import MonitorEventos

EXT_VIDEO = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
ORDEN_NIVEL = ["NORMAL", "PRECAUCION", "ALERTA", "EMERGENCIA"]


def evaluar_video(detector, ruta, dir_salida, umbral, guardar_video, mostrar):
    cap = cv2.VideoCapture(str(ruta))
    if not cap.isOpened():
        print(f"  No se pudo abrir {ruta}")
        return None
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    escritor = None
    monitor = MonitorEventos()
    filas, idx, t_ini = [], 0, time.time()
    primera_alerta = None

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        idx += 1
        t = idx / fps
        dets = detector.detectar(frame, umbral)
        estado = monitor.actualizar(dets, t, f"{t:.1f} s")
        if estado.nivel != "NORMAL" and primera_alerta is None:
            primera_alerta = t
        filas.append({"frame": idx, "t_s": round(t, 2), "tipo": estado.tipo, "nivel": estado.nivel,
                      "n_fuego": estado.num_fuego, "n_humo": estado.num_humo,
                      "conf_fuego_max": max((d.confianza for d in dets if d.clase == "fuego"), default=0),
                      "conf_humo_max": max((d.confianza for d in dets if d.clase == "humo"), default=0)})
        anotado = dibujar(frame, dets, estado)
        if guardar_video:
            if escritor is None:
                alto, ancho = frame.shape[:2]
                escritor = cv2.VideoWriter(str(dir_salida / f"{ruta.stem}_anotado.mp4"),
                                           cv2.VideoWriter_fourcc(*"mp4v"), fps, (ancho, alto))
            escritor.write(anotado)
        if mostrar:
            cv2.imshow("Deteccion de fuego y humo (q para salir)", anotado)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    cap.release()
    if escritor:
        escritor.release()

    if not filas:
        return None
    df = pd.DataFrame(filas)
    df.to_csv(dir_salida / f"{ruta.stem}_frames.csv", index=False)
    eventos = monitor.tabla_eventos()
    pd.DataFrame(eventos).to_csv(dir_salida / f"{ruta.stem}_eventos.csv", index=False)
    return {
        "escenario": ruta.stem,
        "frames": len(df),
        "fps_procesamiento": round(len(df) / (time.time() - t_ini), 1),
        "%_frames_con_fuego": round(100 * (df["n_fuego"] > 0).mean(), 1),
        "%_frames_con_humo": round(100 * (df["n_humo"] > 0).mean(), 1),
        "%_frames_en_alerta": round(100 * (df["nivel"] != "NORMAL").mean(), 1),
        "primera_alerta_s": None if primera_alerta is None else round(primera_alerta, 1),
        "nivel_maximo": max(df["nivel"], key=ORDEN_NIVEL.index),
        "eventos": len(eventos),
        "tipos_evento": ", ".join(sorted({e["tipo"] for e in eventos})) or "ninguno",
    }


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("entradas", nargs="+", help="videos o carpetas con videos")
    p.add_argument("--salida", default=str(config.DIR_RESULTADOS))
    p.add_argument("--umbral", type=float, default=None,
                   help="un solo umbral para fuego y humo (por defecto: los de config.UMBRALES_CLASE)")
    p.add_argument("--sin-video", action="store_true", help="no guardar videos anotados")
    p.add_argument("--mostrar", action="store_true", help="mostrar ventana mientras procesa")
    args = p.parse_args()

    videos = []
    for e in map(Path, args.entradas):
        videos += sorted(v for v in e.iterdir() if v.suffix.lower() in EXT_VIDEO) if e.is_dir() else [e]
    dir_salida = Path(args.salida)
    dir_salida.mkdir(parents=True, exist_ok=True)

    detector = DetectorFuegoHumo()
    resumen = []
    for v in videos:
        print(f"Procesando {v} ...")
        r = evaluar_video(detector, v, dir_salida, args.umbral, not args.sin_video, args.mostrar)
        if r:
            resumen.append(r)
    if args.mostrar:
        cv2.destroyAllWindows()
    if resumen:
        tabla = pd.DataFrame(resumen)
        tabla.to_csv(dir_salida / "resumen_escenarios.csv", index=False)
        print("\n" + tabla.to_string(index=False))
        print(f"\nResultados guardados en {dir_salida}")


if __name__ == "__main__":
    main()
