"""Chamada a um modelo local do Ollama com saída estruturada (validada por Pydantic)."""

import json
import os
from typing import TypeVar

import ollama
from pydantic import BaseModel, ValidationError

MODELO = os.getenv("OLLAMA_MODEL", "llama3.1:8b")
MODELO_VISAO = os.getenv("OLLAMA_MODEL_VISAO", "gemma3:4b")
T = TypeVar("T", bound=BaseModel)

_client = ollama.Client(host=os.getenv("OLLAMA_HOST", "http://localhost:11434"))


class ErroIA(Exception):
    pass


def extrair(texto: str, esquema: type[T], instrucoes: str, imagens: list[bytes] | None = None) -> T:
    """Envia o texto (e imagens, se houver) ao modelo e devolve um objeto do `esquema` já validado."""
    modelo = MODELO_VISAO if imagens else MODELO
    formato = esquema.model_json_schema()
    sistema = f"{instrucoes}\n\nResponda só com JSON neste esquema:\n{json.dumps(formato, ensure_ascii=False)}"
    mensagem = {"role": "user", "content": texto}
    if imagens:
        mensagem["images"] = imagens

    try:
        resposta = _client.chat(
            model=modelo,
            messages=[{"role": "system", "content": sistema}, mensagem],
            format=formato,  # o Ollama restringe a geração a este esquema JSON
            options={"temperature": 0, "num_ctx": 16384},
        )
    except ConnectionError as e:
        raise ErroIA("O Ollama não está rodando. Abra o app do Ollama ou rode `ollama serve`.") from e
    except ollama.ResponseError as e:
        if e.status_code == 404:
            raise ErroIA(f"Modelo `{modelo}` não encontrado. Rode `ollama pull {modelo}`.") from e
        raise ErroIA(f"Erro do Ollama: {e.error}") from e

    try:
        return esquema.model_validate_json(resposta.message.content)
    except ValidationError as e:
        raise ErroIA("O modelo respondeu fora do formato esperado. Tente de novo ou use um modelo maior.") from e
