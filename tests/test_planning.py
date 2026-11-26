import os
import sys
from datetime import date, timedelta

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.planning import IntentPlanner, classify_intent, extract_date


planner = IntentPlanner()


def test_clasificar_devolucion():
    assert classify_intent("Quiero devolver una remera que compré") == "devolucion"


def test_clasificar_stock():
    assert classify_intent("¿Cuánto stock queda de jeans?") == "stock"


def test_clasificar_envio():
    assert classify_intent("¿Cuánto demora el envío a regiones?") == "envio"


def test_clasificar_producto():
    assert classify_intent("¿Cuánto cuesta la polera básica?") == "producto"


def test_clasificar_queja():
    assert classify_intent("Llegó el producto defectuoso y estoy furioso") == "queja"


def test_extraccion_fecha_iso():
    assert extract_date("Compré el 2026-09-15") == "2026-09-15"


def test_extraccion_fecha_espanol():
    assert extract_date("Compré el 15 de septiembre de 2026") == "2026-09-15"


def test_plan_devolucion_con_fecha():
    plan = planner.plan("Quiero devolver mi compra del 2026-10-01", history=[])
    tools = plan.tools_in_order()
    assert "calcular_devolucion" in tools
    assert "buscar_conocimiento" in tools
    assert "crear_ticket" in tools
    assert any("fecha detectada" in n for n in plan.adaptive_notes)


def test_plan_devolucion_sin_fecha():
    plan = planner.plan("Me arrepentí de la compra, quiero devolver", history=[])
    assert any("sin fecha" in n for n in plan.adaptive_notes)


def test_plan_stock_critico_agrega_alternativas():
    plan = planner.plan("¿Quedan chaquetas denim disponibles?", history=[])
    tools = plan.tools_in_order()
    assert "consultar_stock" in tools
    assert "comparar_por_presupuesto" in tools
    assert any("stock crítico" in n for n in plan.adaptive_notes)


def test_plan_stock_normal_sin_alternativas():
    plan = planner.plan("¿Cuántas remeras básicas hay?", history=[])
    assert "comparar_por_presupuesto" not in plan.tools_in_order()


def test_plan_sesion_nueva_recupera_memoria():
    plan = planner.plan("hola", history=[])
    assert "consultar_memoria" in plan.tools_in_order()
    assert any("sesión nueva" in n for n in plan.adaptive_notes)


def test_plan_continuacion_conserva_contexto():
    historial = [{"role": "user", "content": "hola"}, {"role": "assistant", "content": "hola"}]
    plan = planner.plan("y los envíos?", history=historial)
    assert any("continuación" in n for n in plan.adaptive_notes)


def test_plan_queja_escala():
    plan = planner.plan("El producto llegó roto y quiero un reclamo formal", history=[])
    assert "crear_ticket" in plan.tools_in_order()
    assert any("escalamiento" in n for n in plan.adaptive_notes)
