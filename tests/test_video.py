"""Pruebas del salto de fotogramas y de los videos temporales (src/video.py)."""
import os
import time

import pytest

from src.video import (MAX_SALTO, borrar_video_temporal, descartar_frames, frames_a_saltar,
                       guardar_video_temporal, limpiar_videos_antiguos)


class CapturaFalsa:
    """Imita a cv2.VideoCapture: entrega `disponibles` fotogramas y luego se acaba."""

    def __init__(self, disponibles, demora_s=0.0):
        self.disponibles = disponibles
        self.demora_s = demora_s
        self.llamadas = 0

    def grab(self):
        self.llamadas += 1
        time.sleep(self.demora_s)
        if self.disponibles <= 0:
            return False
        self.disponibles -= 1
        return True


# ==========================================================================
# Salto de fotogramas
# ==========================================================================
@pytest.mark.parametrize("segundos, fps, esperado", [
    (0.20, 30, 5),      # en 0,2 s pasaron 6 fotogramas: el analizado + 5 que sobran
    (0.12, 25, 2),      # 3 fotogramas: el analizado + 2
    (0.02, 30, 0),      # la red va más rápido que el video: no se descarta nada
    (0.00, 30, 0),
    (0.20, 0, 0),       # fps desconocido
    (0.20, None, 0),
    (60.0, 30, MAX_SALTO),
])
def test_frames_a_saltar(segundos, fps, esperado):
    assert frames_a_saltar(segundos, fps) == esperado


def test_descartar_frames_descarta_los_pedidos():
    cap = CapturaFalsa(disponibles=100)
    assert descartar_frames(cap, 5) == 5
    assert cap.disponibles == 95


def test_descartar_frames_se_detiene_al_final_del_video():
    assert descartar_frames(CapturaFalsa(disponibles=2), 5) == 2


def test_descartar_frames_con_cero_no_toca_la_fuente():
    cap = CapturaFalsa(disponibles=10)
    assert descartar_frames(cap, 0) == 0
    assert cap.llamadas == 0


def test_descartar_frames_en_vivo_para_cuando_el_bufer_esta_vacio():
    # cada grab tarda más que el límite: se esperó un fotograma nuevo de la cámara
    cap = CapturaFalsa(disponibles=100, demora_s=0.03)
    assert descartar_frames(cap, 10, limite_s=0.005) == 1


# ==========================================================================
# Videos temporales
# ==========================================================================
def test_guardar_y_borrar_video_temporal(tmp_path):
    ruta = guardar_video_temporal(b"contenido del video", ".mp4", dir_temp=tmp_path / "videos")
    assert os.path.dirname(ruta) == str(tmp_path / "videos")
    assert ruta.endswith(".mp4")
    with open(ruta, "rb") as f:
        assert f.read() == b"contenido del video"

    assert borrar_video_temporal(ruta) is True
    assert not os.path.exists(ruta)
    assert borrar_video_temporal(ruta) is True      # borrar dos veces no falla


def test_borrar_video_temporal_sin_ruta():
    assert borrar_video_temporal(None) is True
    assert borrar_video_temporal("") is True


def test_cada_video_temporal_tiene_su_propio_archivo(tmp_path):
    a = guardar_video_temporal(b"a", ".mp4", dir_temp=tmp_path)
    b = guardar_video_temporal(b"b", ".mp4", dir_temp=tmp_path)
    assert a != b


def test_limpiar_videos_antiguos_solo_borra_los_viejos_de_la_app(tmp_path):
    viejo = guardar_video_temporal(b"x", ".mp4", dir_temp=tmp_path)
    reciente = guardar_video_temporal(b"x", ".mp4", dir_temp=tmp_path)
    ajeno = tmp_path / "otro_archivo.mp4"            # no lo creó la app: no se toca
    ajeno.write_bytes(b"x")
    hace_un_dia = time.time() - 24 * 3600
    os.utime(viejo, (hace_un_dia, hace_un_dia))
    os.utime(ajeno, (hace_un_dia, hace_un_dia))

    assert limpiar_videos_antiguos(horas=6, dir_temp=tmp_path) == 1
    assert not os.path.exists(viejo)
    assert os.path.exists(reciente)
    assert ajeno.exists()


def test_limpiar_videos_antiguos_sin_carpeta(tmp_path):
    assert limpiar_videos_antiguos(dir_temp=tmp_path / "no_existe") == 0
