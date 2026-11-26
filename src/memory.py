import json
import os
import unicodedata
from datetime import datetime, timezone

from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np


def _normalize(text: str) -> str:
    text = unicodedata.normalize("NFKD", text.lower().replace("_", " "))
    return "".join(c for c in text if not unicodedata.combining(c))


class ShortTermMemory:
    """Memoria de corto plazo: búfer de ventana fija con los últimos turnos de la sesión actual."""

    def __init__(self, window: int = 6):
        self.window = window
        self.messages: list[dict] = []

    def add(self, role: str, content: str):
        self.messages.append({"role": role, "content": content})

    def get_history(self) -> list[dict]:
        return self.messages[-self.window * 2:]

    def render(self) -> str:
        if not self.messages:
            return "(sin interacciones previas en esta sesión)"
        lines = [f"[{m['role']}] {m['content']}" for m in self.get_history()]
        return "\n".join(lines)

    def clear(self):
        self.messages = []


class LongTermMemory:
    """Memoria de largo plazo persistente en disco (JSON) entre sesiones.

    Almacena hechos clave del cliente, un resumen de interacciones previas
    y el contador de sesiones. Permite recuperación semántica de hechos
    mediante indexación TF-IDF sobre los valores almacenados.
    """

    def __init__(self, path: str = "memory_store/long_term.json"):
        self.path = path
        self.data = {"facts": {}, "summary": "", "sessions": 0, "updated_at": ""}
        self._load()

    def _load(self):
        if os.path.exists(self.path):
            try:
                with open(self.path, encoding="utf-8") as f:
                    self.data = json.load(f)
            except (json.JSONDecodeError, OSError):
                self.data = {"facts": {}, "summary": "", "sessions": 0, "updated_at": ""}

    def persist(self):
        self.data["updated_at"] = datetime.now(timezone.utc).isoformat()
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(self.data, f, ensure_ascii=False, indent=2)

    def save_fact(self, key: str, value: str):
        self.data["facts"][key] = value
        self.persist()

    def get_fact(self, key: str) -> str | None:
        return self.data["facts"].get(key)

    def all_facts(self) -> dict:
        return dict(self.data["facts"])

    def render_facts(self) -> str:
        facts = self.data["facts"]
        if not facts:
            return "(sin hechos registrados)"
        return "\n".join(f"- {k}: {v}" for k, v in facts.items())

    def update_summary(self, text: str):
        prev = self.data.get("summary", "")
        self.data["summary"] = (prev + "\n" + text).strip()[-1500:]
        self.persist()

    def get_summary(self) -> str:
        return self.data.get("summary") or "(sin resumen previo)"

    def register_session(self):
        self.data["sessions"] = self.data.get("sessions", 0) + 1
        self.persist()

    def recall_semantic(self, query: str, k: int = 3) -> list[tuple[str, str]]:
        """Recuperación semántica de hechos: indexa clave+valor con TF-IDF y
        devuelve los k hechos más similares a la consulta del agente."""
        facts = self.data["facts"]
        if not facts:
            return []
        items = list(facts.items())
        corpus = [_normalize(f"{key}: {value}") for key, value in items]
        vectorizer = TfidfVectorizer()
        matrix = vectorizer.fit_transform(corpus)
        q_vec = vectorizer.transform([_normalize(query)])
        scores = np.asarray((q_vec @ matrix.T).todense()).flatten()
        top_idx = scores.argsort()[::-1][:k]
        return [(items[int(i)][0], items[int(i)][1]) for i in top_idx if scores[i] > 0]
