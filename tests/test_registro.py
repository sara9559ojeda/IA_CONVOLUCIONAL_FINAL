"""Pruebas del registro en disco de las alertas (src/registro.py)."""
import os
from types import SimpleNamespace

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

import src.registro as registro_mod  # noqa: E402
from src.eventos import MonitorEventos  # noqa: E402
from src.registro import COLUMNAS, RegistroAlertas  # noqa: E402

IMAGEN = np.full((120, 160, 3), 127, np.uint8)


def evento(id_=1, estado="en curso", duracion=0, clave="fuego_incipiente"):
    """Un evento con la misma forma que los de MonitorEventos.eventos."""
    return {"id": id_, "tipo": "Fuego incipiente", "nivel": "ALERTA", "inicio": "t3",
            "duracion_s": duracion, "conf_max_%": 80, "area_max_%": 2, "estado": estado,
            "_abierto": estado == "en curso", "_t0": 3, "_clave": clave}


def test_sin_eventos_no_se_crea_nada(tmp_path):
    registro = RegistroAlertas(tmp_path / "alertas")
    assert registro.actualizar([], IMAGEN, "camara") is False
    assert registro.leer() == []
    assert not (tmp_path / "alertas").exists()


def test_una_alerta_nueva_guarda_la_fila_y_la_captura(tmp_path):
    registro = RegistroAlertas(tmp_path)
    assert registro.actualizar([evento()], IMAGEN, "incendio.mp4") is True

    (fila,) = registro.leer()
    assert list(fila) == COLUMNAS
    assert fila["sesion"] == registro.sesion
    assert (fila["fuente"], fila["id"], fila["tipo"]) == ("incendio.mp4", "1", "Fuego incipiente")
    assert (fila["nivel"], fila["estado"], fila["conf_max_%"]) == ("ALERTA", "en curso", "80")
    assert fila["captura"].endswith("_evento001_fuego_incipiente.jpg")

    captura = cv2.imdecode(np.fromfile(tmp_path / fila["captura"], np.uint8), cv2.IMREAD_COLOR)
    assert captura.shape == IMAGEN.shape


def test_sin_cambios_no_se_vuelve_a_escribir(tmp_path):
    registro = RegistroAlertas(tmp_path, intervalo_s=0)
    registro.actualizar([evento()], IMAGEN)
    assert registro.actualizar([evento()], IMAGEN) is False


def test_una_alerta_abierta_se_actualiza_por_intervalo_o_al_forzar(tmp_path):
    registro = RegistroAlertas(tmp_path, intervalo_s=3600)
    registro.actualizar([evento(duracion=0)], IMAGEN)

    assert registro.actualizar([evento(duracion=7)], IMAGEN) is False    # aún no toca
    assert registro.leer()[0]["duracion_s"] == "0"
    assert registro.actualizar([evento(duracion=7)], forzar=True) is True
    assert registro.leer()[0]["duracion_s"] == "7"


def test_al_cerrarse_la_alerta_se_escribe_de_inmediato(tmp_path):
    registro = RegistroAlertas(tmp_path, intervalo_s=3600)
    registro.actualizar([evento(duracion=0)], IMAGEN)
    assert registro.actualizar([evento(estado="finalizado", duracion=12)], IMAGEN) is True

    (fila,) = registro.leer()
    assert (fila["estado"], fila["duracion_s"]) == ("finalizado", "12")
    assert len(list(tmp_path.glob("*.jpg"))) == 1          # la captura es la del inicio


def test_las_sesiones_anteriores_se_conservan(tmp_path):
    anterior = RegistroAlertas(tmp_path)
    anterior.actualizar([evento()], IMAGEN, "ayer.mp4")
    nueva = RegistroAlertas(tmp_path)                       # p. ej. después de "Reiniciar sesión"
    nueva.actualizar([evento()], IMAGEN, "hoy.mp4")         # vuelve a ser el evento 1
    nueva.actualizar([evento(), evento(2, clave="humo_negro")], IMAGEN, "hoy.mp4")

    filas = nueva.leer()
    assert [(f["fuente"], f["id"]) for f in filas] == [("ayer.mp4", "1"), ("hoy.mp4", "1"),
                                                      ("hoy.mp4", "2")]
    assert len({f["captura"] for f in filas}) == 3
    assert len(list(tmp_path.glob("*.jpg"))) == 3


def test_admite_tildes_en_la_ruta(tmp_path):
    carpeta = tmp_path / "evaluación señales"
    registro = RegistroAlertas(carpeta)
    registro.actualizar([evento()], IMAGEN)
    (fila,) = registro.leer()
    assert (carpeta / fila["captura"]).stat().st_size > 0


def test_borrar_guardadas_con_dos_sesiones_activas(tmp_path):
    registros = [RegistroAlertas(tmp_path), RegistroAlertas(tmp_path)]
    for registro in registros:
        registro.actualizar([evento()], IMAGEN)
    otro = tmp_path / "otra_imagen.jpg"
    otro.touch()
    registros[0].borrar_guardadas()
    assert list(tmp_path.iterdir()) == [otro]
    for registro in registros:
        registro.actualizar([evento(duracion=2), evento(2)], IMAGEN)
    assert [fila["id"] for fila in registros[0].leer()] == ["2", "2"]
    assert len(list(tmp_path.glob("*_evento*.jpg"))) == 2


def test_si_el_csv_esta_bloqueado_no_falla_y_reintenta(tmp_path, monkeypatch):
    registro = RegistroAlertas(tmp_path, intervalo_s=0)

    def bloqueado(origen, destino):
        raise PermissionError("alertas.csv está abierto en Excel")

    monkeypatch.setattr(registro_mod.os, "replace", bloqueado)
    assert registro.actualizar([evento()], IMAGEN) is False      # no lanza excepción
    assert registro.leer() == []
    assert list(tmp_path.glob("*.tmp")) == []

    monkeypatch.undo()
    assert registro.actualizar([evento()], IMAGEN) is True       # el cambio seguía pendiente
    assert len(registro.leer()) == 1
    assert os.path.exists(tmp_path / registro.leer()[0]["captura"])


def test_con_el_monitor_real(tmp_path):
    """MonitorEventos -> RegistroAlertas, como en la app."""
    monitor, registro = MonitorEventos(), RegistroAlertas(tmp_path, intervalo_s=0)
    llama = SimpleNamespace(clase="fuego", subtipo="llama", confianza=0.8, area_relativa=0.02)
    for t in range(8):                                   # 8 fotogramas con fuego
        monitor.actualizar([llama], t, f"t{t}")
        registro.actualizar(monitor.eventos, IMAGEN, "prueba.mp4")
    for t in range(8, 13):                               # 5 sin nada: el evento se cierra
        monitor.actualizar([], t, f"t{t}")
        registro.actualizar(monitor.eventos, IMAGEN, "prueba.mp4")

    (fila,) = registro.leer()
    assert (fila["tipo"], fila["inicio"], fila["duracion_s"], fila["estado"]) == (
        "Fuego incipiente", "t3", "9", "finalizado")
    assert len(list(tmp_path.glob("*.jpg"))) == 1
