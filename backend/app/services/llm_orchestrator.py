import logging
from typing import Optional, List, Protocol, Dict
import google.generativeai as genai
from app.services.models import Chunk
from app.core.config import Settings

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """Você é um assistente técnico especializado em manuais industriais da WEG, WEG-CESTARI e Baldor.

SEU OBJETIVO:
- Responder de forma objetiva, técnica e clara usando APENAS o conteúdo dos manuais fornecidos no CONTEXTO.
- Evitar qualquer tipo de alucinação. Se algo não estiver nos textos, você deve dizer explicitamente que o manual não traz essa informação.

INSTRUÇÕES DE RESPOSTA:

1. Responda sempre em português (pt-BR), em tom técnico, mas acessível para engenheiros de manutenção.
2. Use SOMENTE as informações presentes nos trechos de manual em "Contexto". 
   - Se precisar fazer comentário geral de boas práticas, marque claramente como "orientação geral" e deixe claro que não está no manual.
3. Quando a informação estiver explícita nos manuais:
   - Comece com uma resposta direta de 1 a 3 parágrafos.
   - Depois, liste os pontos principais em bullet points, se fizer sentido (por exemplo: passos, condições, itens a verificar).
4. Quando a informação NÃO estiver nos manuais:
   - Responda algo como:
     "Os manuais fornecidos não trazem essa informação de forma explícita. Abaixo estão apenas orientações gerais."
   - Se der orientações gerais, deixe isso CLARAMENTE separado do que vem dos manuais.
5. Sempre traga as referências no final, neste formato:
   - Referências:
     - [1] NOME_ARQUIVO.pdf – página X: resumo curto do trecho usado
     - [2] ...
6. Se a resposta combinar informações de mais de um documento, explique rapidamente como cada um contribui (ex.: um define lubrificante, outro define condição de armazenagem).
7. Seja conciso: em geral, 2–5 parágrafos + a seção de referências já é suficiente.
"""

def build_rag_prompt(question: str, chunks: List[Chunk]) -> str:
    context_blocks: List[str] = []
    for idx, chunk in enumerate(chunks, start=1):
        filename = chunk.metadata.get("filename", "desconhecido")
        page = chunk.page if chunk.page is not None else chunk.metadata.get("page_number", "?")
        text = chunk.text.strip()
        context_blocks.append(
            f"[{idx}] arquivo={filename}, página={page}\n    {text}"
        )

    context_str = "\n\n".join(context_blocks)

    prompt = f"""{SYSTEM_PROMPT}

CONTEXTO DOS MANUAIS:
{context_str}

PERGUNTA DO USUÁRIO:
{question}

RESPOSTA (em português, técnico, objetivo, com referências):
"""
    return prompt

def build_rag_messages(question: str, chunks: List[Chunk]) -> List[Dict[str, str]]:
    context_blocks: List[str] = []
    for idx, chunk in enumerate(chunks, start=1):
        filename = chunk.metadata.get("filename", "desconhecido")
        page = chunk.page if chunk.page is not None else chunk.metadata.get("page_number", "?")
        text = chunk.text.strip()
        context_blocks.append(
            f"[{idx}] arquivo={filename}, página={page}\n    {text}"
        )

    context_str = "\n\n".join(context_blocks)

    user_content = f"""CONTEXTO DOS MANUAIS:
{context_str}

PERGUNTA DO USUÁRIO:
{question}

RESPOSTA (em português, técnico, objetivo, com referências):
"""

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]

class LLMProviderError(Exception):
    def __init__(self, provider: str, message: str, original_exception: Optional[Exception] = None):
        self.provider = provider
        self.original_exception = original_exception
        super().__init__(message)

class BaseLLMClient(Protocol):
    provider_name: str
    
    def answer(self, question: str, chunks: List[Chunk]) -> str:
        ...

class LocalLLMClient:
    provider_name = "local"
    
    _model = None
    _tokenizer = None
    _current_model_name = None
    
    def __init__(self, settings: Settings):
        self.settings = settings
        self.model_name = settings.LLM_LOCAL_MODEL_NAME
        self.max_new_tokens = settings.HF_LLM_MAX_NEW_TOKENS
        self.temperature = settings.HF_LLM_TEMPERATURE
        self.cache_dir = settings.HF_CACHE_DIR
    
    def _load_model(self):
        if LocalLLMClient._model is None or LocalLLMClient._current_model_name != self.model_name:
            try:
                import torch
                from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
                
                logger.info(f"Loading local LLM: {self.model_name}")
                
                device = "cuda" if torch.cuda.is_available() else "cpu"
                
                if device == "cuda":
                    quantization_config = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_compute_dtype=torch.float16,
                        bnb_4bit_use_double_quant=True,
                        bnb_4bit_quant_type="nf4"
                    )
                    
                    LocalLLMClient._model = AutoModelForCausalLM.from_pretrained(
                        self.model_name,
                        quantization_config=quantization_config,
                        device_map="auto",
                        cache_dir=self.cache_dir,
                        trust_remote_code=True
                    )
                else:
                    LocalLLMClient._model = AutoModelForCausalLM.from_pretrained(
                        self.model_name,
                        cache_dir=self.cache_dir,
                        trust_remote_code=True
                    )
                    LocalLLMClient._model.to(device)
                
                LocalLLMClient._tokenizer = AutoTokenizer.from_pretrained(
                    self.model_name,
                    cache_dir=self.cache_dir,
                    trust_remote_code=True
                )
                
                LocalLLMClient._current_model_name = self.model_name
                logger.info("Local LLM loaded successfully")
                
            except Exception as e:
                logger.exception("Failed to load local LLM")
                raise LLMProviderError(
                    provider=self.provider_name,
                    message=f"Failed to load local LLM: {str(e)}",
                    original_exception=e
                )
    
    def answer(self, question: str, chunks: List[Chunk]) -> str:
        import time
        from app.services.metrics import get_metrics_collector
        
        try:
            self._load_model()
            
            prompt = build_rag_prompt(question, chunks)
            
            inputs = LocalLLMClient._tokenizer(prompt, return_tensors="pt")
            inputs = {k: v.to(LocalLLMClient._model.device) for k, v in inputs.items()}
            
            start_time = time.time()
            outputs = LocalLLMClient._model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                do_sample=True,
                pad_token_id=LocalLLMClient._tokenizer.eos_token_id
            )
            duration = time.time() - start_time
            
            input_length = inputs["input_ids"].shape[1]
            generated_tokens = outputs[0][input_length:]
            answer = LocalLLMClient._tokenizer.decode(generated_tokens, skip_special_tokens=True)
            
            output_tokens = len(generated_tokens)
            
            get_metrics_collector().record_llm(
                question=question,
                duration=duration,
                input_tokens=input_length,
                output_tokens=output_tokens
            )
            
            return answer.strip()
            
        except Exception as e:
            logger.exception("Error in LocalLLMClient.answer")
            raise LLMProviderError(
                provider=self.provider_name,
                message=f"Local LLM inference failed: {str(e)}",
                original_exception=e
            )

class GeminiLLMClient:
    provider_name = "gemini"
    
    def __init__(self, settings: Settings):
        self.settings = settings
        self.api_key = settings.GEMINI_API_KEY
        self.model_name = settings.GEMINI_LLM_MODEL
        
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY is required")
        
        genai.configure(api_key=self.api_key)
        
        self.model = genai.GenerativeModel(
            model_name=self.model_name,
            system_instruction=SYSTEM_PROMPT
        )
    
    def answer(self, question: str, chunks: List[Chunk]) -> str:
        import time
        from app.services.metrics import get_metrics_collector
        
        try:
            prompt = build_rag_prompt(question, chunks)
            
            start_time = time.time()
            response = self.model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    max_output_tokens=1024,
                    temperature=0.1,
                )
            )
            duration = time.time() - start_time
            
            input_tokens = 0
            output_tokens = 0
            
            if hasattr(response, 'usage_metadata'):
                input_tokens = getattr(response.usage_metadata, 'prompt_token_count', 0)
                output_tokens = getattr(response.usage_metadata, 'candidates_token_count', 0)
            
            get_metrics_collector().record_llm(
                question=question,
                duration=duration,
                input_tokens=input_tokens,
                output_tokens=output_tokens
            )
            
            return response.text.strip() if response.text else ""
            
        except Exception as e:
            logger.exception("Error in GeminiLLMClient.answer")
            raise LLMProviderError(
                provider=self.provider_name,
                message=f"Gemini API failed: {str(e)}",
                original_exception=e
            )

def get_llm_client(settings: Settings, override_provider: Optional[str] = None) -> BaseLLMClient:
    provider = override_provider or settings.LLM_PROVIDER
    
    if provider == "gemini":
        if not settings.GEMINI_API_KEY:
            logger.warning("GEMINI_API_KEY not set; falling back to local LLM")
            return LocalLLMClient(settings)
        
        try:
            return GeminiLLMClient(settings)
        except Exception as e:
            logger.warning(f"Failed to initialize GeminiLLMClient: {e}; falling back to local")
            return LocalLLMClient(settings)
    
    return LocalLLMClient(settings)

def answer_with_fallback(
    question: str,
    chunks: List[Chunk],
    settings: Settings,
    requested_provider: Optional[str],
) -> tuple[str, str, Optional[str]]:
    providers_to_try = []
    
    if requested_provider == "gemini":
        if settings.GEMINI_API_KEY:
            providers_to_try.append("gemini")
        providers_to_try.append("local")
    else:
        providers_to_try.append("local")
        if settings.GEMINI_API_KEY:
            providers_to_try.append("gemini")
    
    last_error = None
    
    for provider in providers_to_try:
        try:
            logger.info(f"Attempting to answer with provider: {provider}")
            client = get_llm_client(settings, override_provider=provider)
            answer = client.answer(question, chunks)
            
            provider_used = client.provider_name
            fallback_from = None
            
            if requested_provider and provider_used != requested_provider:
                fallback_from = requested_provider
                logger.info(f"Used fallback: requested={requested_provider}, used={provider_used}")
            
            return answer, provider_used, fallback_from
            
        except LLMProviderError as e:
            logger.warning(f"Provider {provider} failed: {e}. Trying next provider if available...")
            last_error = e
            continue
    
    error_msg = f"All LLM providers failed. Last error: {last_error}"
    logger.error(error_msg)
    raise LLMProviderError(
        provider="all",
        message=error_msg,
        original_exception=last_error.original_exception if last_error else None
    )

def get_llm_orchestrator(settings: Settings):
    return get_llm_client(settings)
