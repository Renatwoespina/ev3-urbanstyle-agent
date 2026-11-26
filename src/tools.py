import os
from datetime import date, datetime

from src.load_documents import load_catalog_rows


RETURN_WINDOW_DAYS = 30
FREE_SHIPPING_THRESHOLD = 25000
EXPRESS_COST = 4990
LOW_STOCK_THRESHOLD = 35

BASE_DIR = os.path.join(os.path.dirname(__file__), "..")
TICKETS_DIR = os.path.join(BASE_DIR, "tickets")


def _catalog() -> list[dict]:
    return load_catalog_rows()


def buscar_conocimiento_fn(retriever, query: str) -> str:
    docs = retriever.get_relevant_documents(query)
    if not docs:
        return "No se encontró información en el corpus organizacional."
    parts = []
    for d in docs:
        fuente = d.metadata.get("source", "interno")
        partes = d.page_content.strip().split("\n")
        resumen = " | ".join(p.strip() for p in partes[:4])
        parts.append(f"[{fuente}] {resumen}")
    return "Documentos recuperados:\n" + "\n".join(parts)


def consultar_stock_fn(_retriever, producto: str) -> str:
    query = producto.lower().strip()
    rows = _catalog()
    matches = [
        r for r in rows
        if query in r["nombre"].lower() or query in r["sku"].lower() or query in r["categoria"].lower()
    ]
    if not matches:
        return f"No encontré productos que coincidan con '{producto}'. Prueba con el nombre completo o el SKU."
    lines = []
    for r in matches:
        stock = int(r["stock"])
        estado = "CRÍTICO" if stock < LOW_STOCK_THRESHOLD else "disponible"
        lines.append(
            f"- {r['nombre']} (SKU {r['sku']}): ${int(r['precio_stock']):,}".replace(",", ".")
            + f" | stock {stock} unidades ({estado}) | tallas {r['talla']}"
        )
    return "\n".join(lines)


def calcular_devolucion_fn(_retriever, fecha_compra: str) -> str:
    try:
        compra = datetime.strptime(fecha_compra.strip(), "%Y-%m-%d").date()
    except ValueError:
        return "Formato de fecha inválido. Usa AAAA-MM-DD (ej: 2026-09-15)."
    hoy = date.today()
    if compra > hoy:
        return "La fecha de compra es posterior a hoy. Verifica los datos ingresados."
    transcurridos = (hoy - compra).days
    restantes = RETURN_WINDOW_DAYS - transcurridos
    if restantes >= 0:
        return (
            f"Compra del {compra.isoformat()}: lleva {transcurridos} días. "
            f"Dentro del plazo legal de {RETURN_WINDOW_DAYS} días. Quedan {restantes} días para solicitar cambio o devolución."
        )
    return (
        f"Compra del {compra.isoformat()}: lleva {transcurridos} días. "
        f"PLAZO VENCIDO hace {-restantes} días. Requiere escalamiento a soporte humano (crear ticket)."
    )


def comparar_por_presupuesto_fn(_retriever, categoria: str, max_precio: int) -> str:
    cat = categoria.lower().strip()
    rows = [r for r in _catalog() if cat in r["categoria"].lower()]
    candidatos = [r for r in rows if int(r["precio_stock"]) <= int(max_precio)]
    if not candidatos:
        opciones = [r for r in rows[:3]]
        if not opciones:
            return f"No hay productos en la categoría '{categoria}'."
        alt = "\n".join(
            f"- {r['nombre']}: ${int(r['precio_stock']):,}".replace(",", ".") for r in opciones
        )
        return f"Ningún producto de '{categoria}' llega bajo ${int(max_precio):,}. Opciones más económicas:\n{alt}"
    candidatos.sort(key=lambda r: int(r["precio_stock"]))
    lines = [
        f"- {r['nombre']} (SKU {r['sku']}): ${int(r['precio_stock']):,}".replace(",", ".")
        + f" | stock {r['stock']} | tallas {r['talla']}"
        for r in candidatos[:5]
    ]
    return f"Opciones de '{categoria}' hasta ${int(max_precio):,}:\n" + "\n".join(lines)


def evaluar_envio_fn(_retriever, monto_total: int, region: str = "santiago", express: bool = False) -> str:
    monto = int(monto_total)
    region_l = region.lower().strip()
    envio_gratis = monto >= FREE_SHIPPING_THRESHOLD
    if express:
        if "santiago" not in region_l:
            return "El envío express solo está disponible para Santiago. Para regiones el plazo es de 5-7 días hábiles."
        costo = 0 if envio_gratis else EXPRESS_COST
        return (
            f"Envío EXPRESS en Santiago (mismo día, pedido antes de las 14:00). "
            f"Costo: ${costo:,}".replace(",", ".")
            + (" (gratis por superar $25.000)." if envio_gratis else ".")
        )
    if envio_gratis:
        plazo = "2-3 días hábiles" if "santiago" in region_l else "5-7 días hábiles"
        return f"Envío GRATIS por superar $25.000. Plazo estimado para {region}: {plazo}."
    faltan = FREE_SHIPPING_THRESHOLD - monto
    return (
        f"El envío tiene costo por defecto. Te faltan ${faltan:,}".replace(",", ".")
        + f" para el envío gratis sobre $25.000. Plazo estimado para {region}: "
        + ("2-3 días hábiles." if "santiago" in region_l else "5-7 días hábiles.")
    )


def guardar_memoria_fn(long_mem, clave: str, valor: str) -> str:
    long_mem.save_fact(clave.strip(), valor.strip())
    return f"Memoria actualizada: '{clave}' → '{valor}'."


def consultar_memoria_fn(long_mem, clave: str) -> str:
    if clave.strip() == "*":
        return "Hechos en memoria:\n" + long_mem.render_facts()
    value = long_mem.get_fact(clave.strip())
    if value is None:
        relacionados = long_mem.recall_semantic(clave, k=2)
        if relacionados:
            lines = [f"- {k}: {v}" for k, v in relacionados]
            return f"No tengo ese dato exacto. Hechos relacionados:\n" + "\n".join(lines)
        return "No tengo registrados hechos del cliente aún."
    return f"{clave}: {value}"


def crear_ticket_fn(_retriever, motivo: str, detalle: str, prioridad: str = "media") -> str:
    os.makedirs(TICKETS_DIR, exist_ok=True)
    existentes = [f for f in os.listdir(TICKETS_DIR) if f.startswith("TCK-") and f.endswith(".md")]
    numero = len(existentes) + 1
    ticket_id = f"TCK-{numero:04d}"
    prioridad_l = prioridad.lower().strip()
    if prioridad_l not in ("baja", "media", "alta"):
        prioridad_l = "media"
    contenido = (
        f"# {ticket_id} - {motivo}\n\n"
        f"- **Prioridad:** {prioridad_l}\n"
        f"- **Fecha:** {date.today().isoformat()}\n"
        f"- **Estado:** abierto\n\n"
        f"## Detalle\n\n{detalle}\n\n"
        f"## Acción requerida\n\nRevisión por soporte humano.\n"
    )
    path = os.path.join(TICKETS_DIR, f"{ticket_id}.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write(contenido)
    return f"Ticket {ticket_id} creado con prioridad {prioridad_l}. Informa este número al cliente."
