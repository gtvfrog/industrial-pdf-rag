import logging
from typing import Optional, List
from app.core.config import Settings
import google.generativeai as genai

logger = logging.getLogger(__name__)

TECH_KEYWORDS = [
    "transporte", "manuseio", "armazenagem",
    "instalação", "posição de trabalho",
    "lubrificação", "sistema de lubrificação",
    "operação", "partida", "parada",
    "manutenção", "inspeção",
    "defeitos", "falhas", "reparos",
    "dados de placa", "placa de identificação",
    "termo de garantia", "garantia",
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
- Não responda a pergunta, apenas gere variações para busca vetorial.

Pergunta original:
"{question}"

Variações:
    Usa LLM para reescrever a query de forma inteligente.
    Retorna lista vazia se LLM não estiver disponível ou falhar.
    Expande uma query em múltiplas versões para melhorar recall.
    
    Retorna lista com query original + variações do LLM.
