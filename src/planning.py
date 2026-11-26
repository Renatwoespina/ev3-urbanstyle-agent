import re
from dataclasses import dataclass, field

from src.tools import LOW_STOCK_THRESHOLD, RETURN_WINDOW_DAYS
from src.load_documents import load_catalog_rows


INTENT_RULES: list[tuple[str, tuple[str, ...]]] = [
    ("devolucion", ("devolver", "devolución", "devolucion", "reembolso", "arrepent", "cambio", "retornar")),
    ("queja", ("queja", "reclamo", "defectuoso", "roto", "molesto", "furioso", "pérdida", "perdida")),
    ("stock", ("stock", "disponible", "quedan", "inventario", "hay del", "agotad")),
    ("envio", ("envío", "envio", "despacho", "llega", "courier", "seguimiento", "rastrear", "despachar")),
    ("producto", ("precio", "cuesta", "cuánto", "cuanto", "producto", "remera", "jean", "vestido",
                  "chaqueta", "polo", "falda", "buzo", "short", "camisa", "cargo", "crop", "chompa",
                  "zapatilla", "bolso", "gorro", "talla")),
]

DATE_PATTERN = re.compile(r"(\d{4}-\d{2}-\d{2})|(\d{1,2}\s+de\s+[a-záéíóú]+\s+(?:de\s+)?\d{4})")


@dataclass
class PlanStep:
    tool: str
    reason: str
    priority: int


@dataclass
class Plan:
    intent: str
    steps: list[PlanStep] = field(default_factory=list)
    adaptive_notes: list[str] = field(default_factory=list)

    def tools_in_order(self) -> list[str]:
        return [s.tool for s in sorted(self.steps, key=lambda s: s.priority)]

    def render(self) -> str:
        pasos = "; ".join(f"{i+1}. {s.tool}" for i, s in enumerate(sorted(self.steps, key=lambda s: s.priority)))
        notas = " | ".join(self.adaptive_notes) if self.adaptive_notes else "ninguna"
        return f"intención={self.intent}; pasos={pasos}; notas={notas}"


def classify_intent(query: str) -> str:
    q = query.lower()
    for intent, keywords in INTENT_RULES:
        if any(k in q for k in keywords):
            return intent
    return "general"


def extract_date(query: str) -> str | None:
    m = DATE_PATTERN.search(query)
    if not m:
        return None
    if m.group(1):
        return m.group(1)
    meses = {
        "enero": "01", "febrero": "02", "marzo": "03", "abril": "04", "mayo": "05", "junio": "06",
        "julio": "07", "agosto": "08", "septiembre": "09", "setiembre": "09", "octubre": "10",
        "noviembre": "11", "diciembre": "12",
    }
    partes = m.group(2).lower().split()
    try:
        dia = int(partes[0])
        mes = meses[partes[2]]
        anio = partes[-1]
        return f"{anio}-{mes}-{dia:02d}"
    except (IndexError, KeyError, ValueError):
        return None


def _stock_for(query: str) -> int | None:
    q = query.lower()
    for row in load_catalog_rows():
        nombre = row["nombre"].lower()
        if any(token in q for token in nombre.split() if len(token) > 4):
            return int(row["stock"])
    return None


class IntentPlanner:
    """Planificador determinístico: clasifica la intención del cliente y
    secuencia las herramientas por prioridad, ajustando el plan según
    condiciones cambiantes (fecha presente, stock crítico, historial)."""

    def plan(self, query: str, history: list[dict]) -> Plan:
        intent = classify_intent(query)
        plan = Plan(intent=intent)
        fecha = extract_date(query)

        if history:
            plan.adaptive_notes.append("continuación de conversación: se conserva el contexto del turno anterior")
        else:
            plan.adaptive_notes.append("sesión nueva: recuperar hechos de memoria de largo plazo")
            plan.steps.append(PlanStep("consultar_memoria", "recuperar contexto del cliente de sesiones previas", 0))

        if intent == "devolucion":
            plan.steps.append(PlanStep("buscar_conocimiento", "consultar política de devoluciones vigente", 1))
            if fecha:
                plan.steps.append(PlanStep("calcular_devolucion", f"evaluar plazo con fecha {fecha}", 2))
                plan.adaptive_notes.append(f"fecha detectada ({fecha}): se calcula el plazo automáticamente")
            else:
                plan.steps.append(PlanStep("calcular_devolucion", "pedir fecha de compra al cliente para calcular plazo", 2))
                plan.adaptive_notes.append("sin fecha en la consulta: el agente debe solicitarla antes de pronunciarse")
            plan.steps.append(PlanStep("crear_ticket", "abrir caso si el plazo está vencido o hay conflicto", 3))

        elif intent == "stock":
            plan.steps.append(PlanStep("consultar_stock", "verificar disponibilidad en catálogo", 1))
            stock = _stock_for(query)
            if stock is not None and stock < LOW_STOCK_THRESHOLD:
                plan.steps.append(PlanStep("comparar_por_presupuesto", "ofrecer alternativas ante stock crítico", 2))
                plan.adaptive_notes.append(f"stock crítico ({stock} < {LOW_STOCK_THRESHOLD}): se agregan alternativas")

        elif intent == "envio":
            plan.steps.append(PlanStep("buscar_conocimiento", "consultar políticas de envío del FAQ", 1))
            plan.steps.append(PlanStep("evaluar_envio", "calcular costo y plazo según monto y región", 2))

        elif intent == "producto":
            plan.steps.append(PlanStep("consultar_stock", "obtener precio, tallas y disponibilidad", 1))
            plan.steps.append(PlanStep("evaluar_envio", "informar condición de envío según monto", 2))

        elif intent == "queja":
            plan.steps.append(PlanStep("buscar_conocimiento", "revisar protocolo de atención y defectos", 1))
            plan.steps.append(PlanStep("crear_ticket", "escalar a soporte humano con prioridad alta", 2))
            plan.adaptive_notes.append("caso de reclamo: escalamiento obligatorio con prioridad alta")

        else:
            plan.steps.append(PlanStep("buscar_conocimiento", "recuperar contexto general del corpus", 1))

        if intent == "devolucion" and fecha:
            plan.adaptive_notes.append(
                f"regla de negocio: plazo máximo {RETURN_WINDOW_DAYS} días desde la compra"
            )

        return plan
