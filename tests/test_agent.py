import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from langchain_core.messages import AIMessage

from src.agent import create_agent


class FakeLLM:
    """LLM guionizado para probar el loop de herramientas sin API."""

    def __init__(self, script):
        self.script = list(script)
        self.i = 0
        self.calls = []

    def bind_tools(self, tools):
        return self

    def invoke(self, messages):
        self.calls.append(messages)
        msg = self.script[min(self.i, len(self.script) - 1)]
        self.i += 1
        return msg


def test_loop_ejecuta_herramientas_y_responde(tmp_path):
    llm = FakeLLM([
        AIMessage(content="", tool_calls=[
            {"name": "consultar_stock", "args": {"producto": "remera"}, "id": "call_1"}
        ]),
        AIMessage(content="", tool_calls=[
            {"name": "guardar_memoria", "args": {"clave": "interes", "valor": "remera basica"}, "id": "call_2"}
        ]),
        AIMessage(content="La Remera Básica Clásica cuesta $12.990 y tiene stock de 150 unidades."),
    ])
    agent = create_agent(llm=llm, memory_dir=str(tmp_path))
    result = agent.run("¿Cuánto cuesta la remera básica?")

    assert result["answer"].startswith("La Remera Básica")
    assert "consultar_stock" in result["tools_used"]
    assert "guardar_memoria" in result["tools_used"]
    assert result["plan"].intent == "producto"
    assert agent.long_mem.get_fact("interes") == "remera basica"
    assert len(agent.short_mem.get_history()) == 2


def test_loop_recupera_memoria_en_sesion_nueva(tmp_path):
    mem_dir = tmp_path
    llm1 = FakeLLM([
        AIMessage(content="", tool_calls=[
            {"name": "guardar_memoria", "args": {"clave": "nombre", "valor": "Camila"}, "id": "c1"}
        ]),
        AIMessage(content="Listo Camila, guardé tu nombre."),
    ])
    agent1 = create_agent(llm=llm1, memory_dir=str(mem_dir))
    agent1.run("Me llamo Camila")

    llm2 = FakeLLM([
        AIMessage(content="", tool_calls=[
            {"name": "consultar_memoria", "args": {"clave": "nombre"}, "id": "c2"}
        ]),
        AIMessage(content="Hola Camila, ¿en qué te ayudo hoy?"),
    ])
    agent2 = create_agent(llm=llm2, memory_dir=str(mem_dir))
    result = agent2.run("Hola")
    assert "consultar_memoria" in result["tools_used"]
    assert agent2.long_mem.data["sessions"] == 2


def test_loop_tolera_herramienta_invalida(tmp_path):
    llm = FakeLLM([
        AIMessage(content="", tool_calls=[
            {"name": "tool_inexistente", "args": {}, "id": "bad"}
        ]),
        AIMessage(content="Disculpa, tuve un problema técnico. Derivo tu consulta a soporte."),
    ])
    agent = create_agent(llm=llm, memory_dir=str(tmp_path))
    result = agent.run("algo raro")
    assert "no existe" in " ".join(
        m.content for m in llm.calls[-1] if hasattr(m, "content") and isinstance(m.content, str)
    ) or result["tools_used"] == ["tool_inexistente"]


def test_audit_se_escibe(tmp_path, monkeypatch):
    from src import agent as agent_mod
    log_path = str(tmp_path / "consultas.jsonl")
    monkeypatch.setattr(agent_mod, "LOG_PATH", log_path)
    llm = FakeLLM([AIMessage(content="Respuesta de prueba.")])
    ag = create_agent(llm=llm, memory_dir=str(tmp_path))
    ag.run("hola")
    assert os.path.exists(log_path)
    contenido = open(log_path, encoding="utf-8").read()
    assert "hola" in contenido
