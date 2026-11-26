import csv
import os
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document


DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")


def load_faq() -> list[Document]:
    path = os.path.join(DATA_DIR, "faq.txt")
    return TextLoader(path, encoding="utf-8").load()


def load_policies() -> list[Document]:
    path = os.path.join(DATA_DIR, "politica_devoluciones.txt")
    return TextLoader(path, encoding="utf-8").load()


def load_size_guide() -> list[Document]:
    path = os.path.join(DATA_DIR, "guia_tallas.txt")
    return TextLoader(path, encoding="utf-8").load()


def load_catalog() -> list[Document]:
    path = os.path.join(DATA_DIR, "catalogo_productos.csv")
    docs = []
    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            text = (
                f"Producto: {row['nombre']}\n"
                f"SKU: {row['sku']}\n"
                f"Categoría: {row['categoria']}\n"
                f"Precio: ${row['precio_stock']}\n"
                f"Stock: {row['stock']} unidades\n"
                f"Tallas disponibles: {row['talla']}\n"
                f"Colores: {row['color']}\n"
                f"Descripción: {row['descripcion']}\n"
            )
            docs.append(Document(page_content=text, metadata={"source": "catalogo", "sku": row["sku"]}))
    return docs


def load_catalog_rows() -> list[dict]:
    path = os.path.join(DATA_DIR, "catalogo_productos.csv")
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def split_documents(docs: list[Document], chunk_size: int = 1000, chunk_overlap: int = 200) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )
    return splitter.split_documents(docs)


def load_all_documents() -> list[Document]:
    all_docs = []
    all_docs.extend(load_catalog())
    all_docs.extend(split_documents(load_faq()))
    all_docs.extend(split_documents(load_policies()))
    all_docs.extend(split_documents(load_size_guide()))
    return all_docs
