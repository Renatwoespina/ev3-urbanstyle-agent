import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from src.load_documents import load_all_documents


class LocalRetriever:
    """Recuperador TF-IDF local. Recuperación semántica del corpus organizacional sin costo de API."""

    def __init__(self, k: int = 4):
        self.k = k
        self.docs = []
        self.vectorizer = None
        self.vectors = None
        self.is_built = False

    def build(self):
        self.docs = load_all_documents()
        texts = [d.page_content for d in self.docs]
        self.vectorizer = TfidfVectorizer()
        self.vectors = self.vectorizer.fit_transform(texts)
        self.is_built = True

    def get_relevant_documents(self, query: str):
        if not self.is_built:
            self.build()
        q_vec = self.vectorizer.transform([query])
        scores = np.asarray((q_vec @ self.vectors.T).todense()).flatten()
        top_idx = scores.argsort()[-self.k:][::-1]
        results = [self.docs[int(i)] for i in top_idx if scores[i] > 0]
        return results if results else self.docs[: self.k]
