import json
import os
from datetime import datetime, timezone

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_groq import ChatGroq
from langchain_core.tools import tool

from src.rag_pipeline import LocalRetriever
from src.memory import ShortTermMemory, LongTermMemory
from src.planning import IntentPlanner, extract_date
from src.prompts import SYSTEM_PROMPT, MEMORY_BLOCK, PLAN_BLOCK, FALLBACK_RESPONSE
from src import tools as T

load_dotenv()

BASE_DIR = os.path.join(os.path.dirname(__file__), "..")
LOG_PATH = os.path.join(BASE_DIR, "logs", "consultas.jsonl")
MAX_TOOL_ITERATIONS = 5


class UrbanAgent:
    """Agente orquestador: memoria (corto/largo plazo), planificación adaptativa
    y loop de herramientas de consulta, razonamiento y escritura sobre un LLM con tool calling."""

    def __init__(self, llm=None, memory_dir: str | None = None):
        self.retriever = LocalRetriever(k=4)
        self.retriever.build()
        mem_dir = memory_dir or os.path.join(BASE_DIR, "memory_store")
        self.short_mem = ShortTermMemory(window=6)
        self.long_mem = LongTermMemory(path=os.path.join(mem_dir, "long_term.json"))
        self.long_mem.register_session()
        self.planner = IntentPlanner()
        self.tools = self._build_tools()
        self.llm = llm if llm is not None else ChatGroq(
            model=os.getenv("MODEL_NAME", "llama-3.3-70b-versatile"),
            temperature=0.2,
        )
        self.llm_with_tools = self.llm.bind_tools(self.tools) if hasattr(self.llm, "bind_tools") else self.llm
        self._tool_map = {t.name: t for t in self.tools}

    def _build_tools(self):
        retriever = self.retriever
        long_mem = self.long_mem

        @tool
        def buscar_conocimiento(query: str) -> str:
            """Herramienta de CONSULTA. Busca en el corpus interno de UrbanStyle (catálogo, FAQ, políticas, tallas) la información más relevante para la consulta del cliente."""
            return T.buscar_conocimiento_fn(retriever, query)

        @tool
        def consultar_stock(producto: str) -> str:
            """Herramienta de CONSULTA. Devuelve precio, stock y tallas de un producto del catálogo por nombre, categoría o SKU."""
            return T.consultar_stock_fn(retriever, producto)

        @tool
        def calcular_devolucion(fecha_compra: str) -> str:
            """Herramienta de RAZONAMIENTO. Recibe una fecha AAAA-MM-DD de compra y calcula si está dentro del plazo de devolución de 30 días, cuántos días quedan o si está vencida."""
            return T.calcular_devolucion_fn(retriever, fecha_compra)

        @tool
        def comparar_por_presupuesto(categoria: str, max_precio: int) -> str:
            """Herramienta de RAZONAMIENTO. Lista productos de una categoría hasta un precio máximo, ordenados de menor a mayor. Útil para ofrecer alternativas."""
            return T.comparar_por_presupuesto_fn(retriever, categoria, max_precio)

        @tool
        def evaluar_envio(monto_total: int, region: str = "santiago", express: bool = False) -> str:
            """Herramienta de RAZONAMIENTO. Calcula costo y plazo de envío según el monto total, la región y si se solicita express."""
            return T.evaluar_envio_fn(retriever, monto_total, region, express)

        @tool
        def guardar_memoria(clave: str, valor: str) -> str:
            """Herramienta de ESCRITURA. Guarda un hecho del cliente en la memoria de largo plazo (ej: nombre, compra realizada, caso abierto)."""
            return T.guardar_memoria_fn(long_mem, clave, valor)

        @tool
        def consultar_memoria(clave: str) -> str:
            """Herramienta de CONSULTA. Recupera un hecho de la memoria de largo plazo por clave. Usa '*' para listar todos los hechos."""
            return T.consultar_memoria_fn(long_mem, clave)

        @tool
        def crear_ticket(motivo: str, detalle: str, prioridad: str = "media") -> str:
            """Herramienta de ESCRITURA. Crea un ticket de escalamiento a soporte humano en el sistema de tickets y devuelve el número de caso."""
            return T.crear_ticket_fn(retriever, motivo, detalle, prioridad)

        return [
            buscar_conocimiento,
            consultar_stock,
            calcular_devolucion,
            comparar_por_presupuesto,
            evaluar_envio,
            guardar_memoria,
            consultar_memoria,
            crear_ticket,
        ]

    def _system_message(self, plan) -> SystemMessage:
        memory_block = MEMORY_BLOCK.format(
            facts=self.long_mem.render_facts(),
            summary=self.long_mem.get_summary(),
        )
        plan_block = PLAN_BLOCK.format(
            intent=plan.intent,
            steps=", ".join(plan.tools_in_order()),
            notes="; ".join(plan.adaptive_notes) or "ninguna",
        )
        return SystemMessage(content=SYSTEM_PROMPT + "\n\n" + memory_block + "\n" + plan_block)

    def run(self, user_input: str) -> dict:
        plan = self.planner.plan(user_input, self.short_mem.get_history())
        messages = [self._system_message(plan), HumanMessage(content=user_input)]

        answer = None
        used_tools: list[str] = []
        iterations = 0

        try:
            while iterations < MAX_TOOL_ITERATIONS:
                iterations += 1
                ai_msg = self.llm_with_tools.invoke(messages)
                messages.append(ai_msg)

                if not getattr(ai_msg, "tool_calls", None):
                    answer = ai_msg.content
                    break

                for tc in ai_msg.tool_calls:
                    name = tc["name"]
                    args = tc.get("args", {})
                    used_tools.append(name)
                    result = self._execute_tool(name, args, user_input)
                    messages.append(ToolMessage(content=result, tool_call_id=tc.get("id", name)))
        except Exception as e:
            answer = f"{FALLBACK_RESPONSE}\n[error: {e}]"

        if answer is None:
            answer = "No logré completar el plan de herramientas. Derivo tu caso a soporte humano: soporte@urbanstyle.cl"

        self.short_mem.add("user", user_input)
        self.short_mem.add("assistant", answer)
        self.long_mem.update_summary(f"[{plan.intent}] Cliente: {user_input[:120]} → Agente usó: {', '.join(used_tools) or 'respuesta directa'}")
        self._audit(user_input, answer, plan, used_tools)

        return {
            "answer": answer,
            "plan": plan,
            "tools_used": used_tools,
            "iterations": iterations,
        }

    def _execute_tool(self, name: str, args: dict, fallback_query: str) -> str:
        if name == "calcular_devolucion" and "fecha_compra" in args:
            args["fecha_compra"] = args["fecha_compra"] or extract_date(fallback_query) or ""
        tool_obj = self._tool_map.get(name)
        if tool_obj is None:
            return f"Herramienta '{name}' no existe. Usa solo las herramientas disponibles."
        try:
            return tool_obj.invoke(args)
        except Exception as e:
            return f"Error ejecutando {name}: {e}. Corrige los argumentos e intenta de nuevo."

    def _audit(self, question: str, answer: str, plan, tools_used: list[str]):
        os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
        registro = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "question": question,
            "answer": answer[:500],
            "intent": plan.intent,
            "plan": plan.tools_in_order(),
            "tools_used": tools_used,
        }
        with open(LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(registro, ensure_ascii=False) + "\n")


def create_agent(llm=None, memory_dir: str | None = None) -> UrbanAgent:
    return UrbanAgent(llm=llm, memory_dir=memory_dir)
