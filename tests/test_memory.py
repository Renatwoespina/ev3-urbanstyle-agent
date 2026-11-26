import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.memory import ShortTermMemory, LongTermMemory


def test_ventana_corto_plazo():
    mem = ShortTermMemory(window=2)
    for i in range(5):
        mem.add("user", f"pregunta {i}")
        mem.add("assistant", f"respuesta {i}")
    historia = mem.get_history()
    assert len(historia) == 4
    assert historia[-1]["content"] == "respuesta 4"


def test_persistencia_largo_plazo(tmp_path):
    path = str(tmp_path / "long.json")
    mem1 = LongTermMemory(path=path)
    mem1.save_fact("nombre", "Javier")
    mem1.update_summary("Cliente preguntó por una chaqueta")
    mem1.register_session()

    mem2 = LongTermMemory(path=path)
    assert mem2.get_fact("nombre") == "Javier"
    assert "chaqueta" in mem2.get_summary()
    assert mem2.data["sessions"] == 1


def test_recuperacion_semantica(tmp_path):
    mem = LongTermMemory(path=str(tmp_path / "long.json"))
    mem.save_fact("ultima_compra", "Chaqueta Denim Unisex el 2026-09-01")
    mem.save_fact("talla_preferida", "M")
    mem.save_fact("canal_favorito", "tienda Las Condes")
    relacionados = mem.recall_semantic("¿qué se compró en la última visita?")
    assert len(relacionados) >= 1
    assert relacionados[0][0] == "ultima_compra"


def test_memoria_vacia(tmp_path):
    mem = LongTermMemory(path=str(tmp_path / "nueva.json"))
    assert mem.render_facts() == "(sin hechos registrados)"
    assert mem.recall_semantic("algo") == []
