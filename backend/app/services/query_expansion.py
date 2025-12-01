import logging
from typing import List, Optional

import google.generativeai as genai

from app.core.config import Settings

logger = logging.getLogger(__name__)

# Palavras típicas de manuais técnicos em PT-BR
TECH_KEYWORDS = [
    "transporte",
    "manuseio",
    "armazenagem",
    "instalação",
    "posição de trabalho",
    "lubrificação",
    "sistema de lubrificação",
    "operação",
    "partida",
    "parada",
    "manutenção",
    "inspeção",
    "defeitos",
    "falhas",
    "reparos",
    "dados de placa",
    "placa de identificação",
    "termo de garantia",
    "garantia",
]

QUERY_EXPANSION_PROMPT = """
Você recebe uma dúvida de um técnico sobre motores elétricos, redutores ou motorredutores.

Sua tarefa é REFORMULAR a pergunta em até 3 variações curtas, focadas em termos
que normalmente aparecem em manuais técnicos (transporte, manuseio, armazenagem,
instalação, posição de trabalho, lubrificação, operação, manutenção, defeitos,
reparos, dados de placa, termo de garantia etc.).

Regras:
- Mantenha o idioma original da pergunta.
- Cada linha da saída deve conter UMA reformulação.
- Não responda à pergunta, apenas gere variações para busca vetorial.

Pergunta original:
"{question}"

Variações:
"""


def _parse_variations(raw_text: str) -> List[str]:
    lines = []
    for line in raw_text.splitlines():
        cleaned = line.strip(" -•\t")
        if cleaned:
            lines.append(cleaned)
    # remove duplicados preservando ordem
    return list(dict.fromkeys(lines))[:3]


def _expand_with_llm(question: str, settings: Settings, llm_provider: Optional[str]) -> List[str]:
    try:
        from app.services.llm_orchestrator import LocalLLMClient, get_llm_client

        if not settings.QUERY_EXPANSION_USE_LLM:
            logger.info("[QUERY_EXPANSION] LLM expansion disabled by config")
            return []

        if settings.LLM_PROVIDER == "gemini" and not settings.GEMINI_API_KEY:
            logger.info("[QUERY_EXPANSION] Gemini selected but no API key available")
            return []

        client = get_llm_client(settings, override_provider=llm_provider)
        prompt = QUERY_EXPANSION_PROMPT.format(question=question)
        expanded_text = ""

        if hasattr(client, "model") and isinstance(client.model, genai.GenerativeModel):
            response = client.model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    max_output_tokens=150,
                    temperature=0.3,
                ),
            )
            expanded_text = response.text if response and response.text else ""
        elif isinstance(client, LocalLLMClient):
            client._load_model()  # relies on LocalLLMClient internals
            inputs = client._tokenizer(prompt, return_tensors="pt")
            inputs = {k: v.to(client._model.device) for k, v in inputs.items()}

            outputs = client._model.generate(
                **inputs,
                max_new_tokens=150,
                temperature=0.3,
                do_sample=True,
                pad_token_id=client._tokenizer.eos_token_id,
            )

            input_length = inputs["input_ids"].shape[1]
            generated_tokens = outputs[0][input_length:]
            expanded_text = client._tokenizer.decode(generated_tokens, skip_special_tokens=True)
        else:
            logger.info("[QUERY_EXPANSION] No supported LLM client available")
            return []

        variations = _parse_variations(expanded_text)

        if variations:
            logger.info(f"[QUERY_EXPANSION] LLM expanded query successfully: {variations}")
            return variations

        logger.warning("[QUERY_EXPANSION] LLM returned empty response or no variations")
        return []

    except Exception as exc:  # defensive: never quebrar pipeline
        logger.warning(f"[QUERY_EXPANSION] LLM expansion failed: {exc}")
        return []


def expand_query(question: str, settings: Settings, llm_provider: Optional[str] = None) -> List[str]:
    queries = []
    
    q_clean = question.strip()
    if q_clean:
        queries.append(q_clean)

    llm_variations = _expand_with_llm(q_clean, settings, llm_provider)
    queries.extend(llm_variations)

    return list(dict.fromkeys(queries))
