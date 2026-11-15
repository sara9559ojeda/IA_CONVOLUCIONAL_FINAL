"""Pruebas de la confirmación temporal, la clasificación de eventos y el
historial (src/eventos.py). No necesitan TensorFlow ni OpenCV."""
from types import SimpleNamespace

import pytest

from src.eventos import EstadoAlerta, MonitorEventos, resumen_para_asistente


def fuego(confianza=0.8, area=0.02):
    return SimpleNamespace(clase="fuego", subtipo="llama", confianza=confianza, area_relativa=area)


def humo(subtipo="blanco", confianza=0.6, area=0.05):
    return SimpleNamespace(clase="humo", subtipo=subtipo, confianza=confianza, area_relativa=area)


def alimentar(monitor, fotogramas, t0=0):
    """Pasa una lista de fotogramas (cada uno, una lista de detecciones) y
    retorna el estado después de cada uno. El tiempo avanza 1 s por fotograma."""
    return [monitor.actualizar(dets, t0 + k, f"t{t0 + k}") for k, dets in enumerate(fotogramas)]


# ==========================================================================
# Confirmación: 4 de los últimos 8 fotogramas
# ==========================================================================
def test_sin_detecciones_el_estado_es_normal():
    estado = alimentar(MonitorEventos(), [[]] * 10)[-1]
    assert (estado.tipo, estado.nivel) == ("sin_incidentes", "NORMAL")


def test_un_fotograma_aislado_no_genera_alerta():
    monitor = MonitorEventos()
    estados = alimentar(monitor, [[fuego()]] + [[]] * 9)
    assert all(e.nivel == "NORMAL" for e in estados)
    assert monitor.eventos == []


def test_el_fuego_se_confirma_al_cuarto_fotograma():
    estados = alimentar(MonitorEventos(), [[fuego()]] * 4)
    assert [e.tipo for e in estados[:3]] == ["sin_incidentes"] * 3
    assert (estados[3].tipo, estados[3].nivel) == ("fuego_incipiente", "ALERTA")
    assert estados[3].frames_fuego == 4
    assert estados[3].conf_fuego == pytest.approx(0.8)


def test_una_imagen_suelta_se_confirma_de_inmediato():
    monitor = MonitorEventos()
    assert monitor.actualizar([fuego()], 0, "imagen 1", inmediato=True).tipo == "fuego_incipiente"
    # la siguiente imagen no hereda nada de la anterior
    assert monitor.actualizar([], 1, "imagen 2", inmediato=True).tipo == "sin_incidentes"


# ==========================================================================
# Clasificación del evento
# ==========================================================================
def test_humo_blanco_es_precaucion():
    estado = alimentar(MonitorEventos(), [[humo("blanco")]] * 4)[-1]
    assert (estado.tipo, estado.nivel, estado.color_humo) == ("humo_blanco", "PRECAUCION", "blanco")


def test_humo_negro_es_alerta():
    estado = alimentar(MonitorEventos(), [[humo("negro")]] * 4)[-1]
    assert (estado.tipo, estado.nivel, estado.color_humo) == ("humo_negro", "ALERTA", "negro")


def test_el_color_del_humo_se_decide_por_mayoria():
    estado = alimentar(MonitorEventos(), [[humo("negro")]] + [[humo("blanco")]] * 3)[-1]
    assert estado.tipo == "humo_blanco"


def test_fuego_extenso_es_incendio_declarado():
    estado = alimentar(MonitorEventos(), [[fuego(area=0.15)]] * 4)[-1]
    assert (estado.tipo, estado.nivel) == ("incendio_declarado", "EMERGENCIA")
    assert "el fuego es extenso" in estado.razon


def test_fuego_con_humo_negro_es_incendio_declarado():
    estado = alimentar(MonitorEventos(), [[fuego(), humo("negro")]] * 4)[-1]
    assert estado.tipo == "incendio_declarado"
    assert "hay humo negro" in estado.razon


def test_fuego_que_crece_es_incendio_declarado():
    areas = [0.03, 0.04, 0.06, 0.08]
    estado = alimentar(MonitorEventos(), [[fuego(area=a)] for a in areas])[-1]
    assert estado.tendencia == "creciendo"
    assert estado.tipo == "incendio_declarado"
    assert "el fuego está creciendo" in estado.razon


@pytest.mark.parametrize("areas, esperada", [
    ([0.01, 0.01, 0.02, 0.02], "creciendo"),
    ([0.02, 0.02, 0.01, 0.01], "disminuyendo"),
    ([0.02, 0.02, 0.02, 0.02], "estable"),
    ([0.01, 0.05], "estable"),                  # con menos de 4 fotogramas no se opina
])
def test_tendencia(areas, esperada):
    assert MonitorEventos._tendencia(areas) == esperada


# ==========================================================================
# Historial de eventos
# ==========================================================================
def test_el_historial_abre_y_cierra_el_evento():
    monitor = MonitorEventos()
    alimentar(monitor, [[fuego()]] * 8)                 # t = 0..7 ; se confirma en t = 3
    (evento,) = monitor.tabla_eventos()
    assert evento["estado"] == "en curso"
    assert evento["inicio"] == "t3"
    assert evento["duracion_s"] == 4

    estados = alimentar(monitor, [[]] * 5, t0=8)        # t = 8..12
    # sigue confirmado mientras queden 4 fotogramas con fuego en la ventana (hasta t = 11)
    assert [e.tipo for e in estados] == ["fuego_incipiente"] * 4 + ["sin_incidentes"]
    (evento,) = monitor.tabla_eventos()
    assert evento == {"id": 1, "tipo": "Fuego incipiente", "nivel": "ALERTA", "inicio": "t3",
                      "duracion_s": 8, "conf_max_%": 80, "area_max_%": 2, "estado": "finalizado"}


def test_un_cambio_de_tipo_cierra_el_evento_y_abre_otro():
    monitor = MonitorEventos()
    alimentar(monitor, [[humo("blanco")]] * 4 + [[fuego()]] * 4)
    primero, segundo = monitor.tabla_eventos()
    assert (primero["id"], primero["nivel"], primero["estado"]) == (1, "PRECAUCION", "finalizado")
    assert (segundo["id"], segundo["tipo"], segundo["estado"]) == (2, "Fuego incipiente", "en curso")


def test_reiniciar_borra_el_historial():
    monitor = MonitorEventos()
    alimentar(monitor, [[fuego()]] * 4)
    monitor.reiniciar()
    assert monitor.eventos == []
    assert monitor.actualizar([], 0, "t0").tipo == "sin_incidentes"


# ==========================================================================
# Texto que recibe el asistente
# ==========================================================================
def test_resumen_sin_incidentes():
    texto = resumen_para_asistente(EstadoAlerta(), [])
    assert "no detecta fuego ni humo" in texto
    assert "0 eventos" in texto


def test_resumen_describe_la_deteccion():
    monitor = MonitorEventos()
    estado = alimentar(monitor, [[fuego()]] * 4)[-1]
    texto = resumen_para_asistente(estado, monitor.eventos)
    assert "Fuego incipiente" in texto
    assert "El nivel de alerta actual es ALERTA." in texto
    assert "80 por ciento" in texto
    assert "1 eventos" in texto
