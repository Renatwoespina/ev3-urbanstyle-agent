import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.rag_pipeline import LocalRetriever
from src import tools as T


def get_retriever():
    r = LocalRetriever(k=4)
    r.build()
    return r


def test_buscar_conocimiento():
    r = get_retriever()
    out = T.buscar_conocimiento_fn(r, "política de devoluciones")
    assert "Documentos recuperados" in out
    assert len(out) > 50


def test_consultar_stock_por_nombre():
    r = get_retriever()
    out = T.consultar_stock_fn(r, "remera")
    assert "URB-001" in out
    assert "12.990" in out or "12990" in out


def test_consultar_stock_inexistente():
    r = get_retriever()
    out = T.consultar_stock_fn(r, "zapatos de moto")
    assert "No encontré" in out


def test_calcular_devolucion_dentro_de_plazo():
    r = get_retriever()
    from datetime import date, timedelta
    reciente = (date.today() - timedelta(days=10)).isoformat()
    out = T.calcular_devolucion_fn(r, reciente)
    assert "Dentro del plazo" in out
    assert "Quedan 20 días" in out


def test_calcular_devolucion_vencida():
    r = get_retriever()
    from datetime import date, timedelta
    vieja = (date.today() - timedelta(days=45)).isoformat()
    out = T.calcular_devolucion_fn(r, vieja)
    assert "PLAZO VENCIDO" in out


def test_calcular_devolucion_fecha_invalida():
    r = get_retriever()
    out = T.calcular_devolucion_fn(r, "ayer")
    assert "Formato de fecha inválido" in out


def test_comparar_por_presupuesto():
    r = get_retriever()
    out = T.comparar_por_presupuesto_fn(r, "remeras", 15000)
    assert "Remera Básica" in out


def test_evaluar_envio_gratis():
    r = get_retriever()
    out = T.evaluar_envio_fn(r, 30000, "santiago")
    assert "GRATIS" in out


def test_evaluar_envio_con_costo():
    r = get_retriever()
    out = T.evaluar_envio_fn(r, 12990, "regiones")
    assert "faltan" in out.lower()


def test_evaluar_envio_express_fuera_de_santiago():
    r = get_retriever()
    out = T.evaluar_envio_fn(r, 30000, "concepción", express=True)
    assert "solo está disponible para Santiago" in out


def test_crear_ticket(tmp_path, monkeypatch):
    monkeypatch.setattr(T, "TICKETS_DIR", str(tmp_path))
    out = T.crear_ticket_fn(None, "Devolución vencida", "Cliente compra del 2026-01-01", "alta")
    assert "TCK-" in out
    assert "prioridad alta" in out
    archivos = list(tmp_path.iterdir())
    assert len(archivos) == 1
    contenido = archivos[0].read_text(encoding="utf-8")
    assert "Devolución vencida" in contenido


def test_guardar_y_consultar_memoria(tmp_path):
    from src.memory import LongTermMemory
    mem = LongTermMemory(path=str(tmp_path / "mem.json"))
    out_save = T.guardar_memoria_fn(mem, "nombre_cliente", "Camila Rojas")
    assert "Memoria actualizada" in out_save
    out_get = T.consultar_memoria_fn(mem, "nombre_cliente")
    assert "Camila Rojas" in out_get
    out_miss = T.consultar_memoria_fn(mem, "pedido_favorito")
    assert "No tengo ese dato exacto" in out_miss or "No tengo registrados" in out_miss
