import logging
from typing import Optional, List, Protocol
import google.generativeai as genai
from app.services.models import Chunk
from app.core.config import Settings

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """
Você é um assistente técnico especializado em equipamentos industriais, com foco em motores, redutores, motorredutores e produtos WEG/WEG-CESTARI.

Seu trabalho é responder perguntas APENAS com base nos trechos de manuais e documentos fornecidos no CONTEXTO. 
Você está atendendo um profissional de manutenção/engenharia, então responda com clareza, precisão técnica e sem enrolação.

REGRAS IMPORTANTES:

1. Use SOMENTE as informações presentes no contexto fornecido.
   - Se a pergunta não puder ser respondida com segurança a partir do contexto, diga explicitamente:
     "Não encontrei informação suficiente nos manuais fornecidos para responder com segurança."
   - Nunca invente especificações, limites, recomendações ou dados técnicos.

2. Idioma:
   - Responda sempre em português do Brasil, mesmo que o manual tenha partes em inglês ou espanhol.
   - Se citar trechos em outra língua, pode traduzir ou explicar em português.

3. Forma da resposta:
   - Comece com uma resposta direta, em 1–3 frases.
   - Em seguida, se fizer sentido, detalhe em tópicos:
     - condições
     - parâmetros importantes (torque, temperatura, posição de montagem, lubrificação etc.)
     - cuidados de segurança e manutenção
   - Seja objetivo, mas bem explicado. Evite parágrafos gigantes.

4. Referências aos documentos:
   - Sempre que possível, indique de onde tirou a informação, com:
     - nome do arquivo (ou título simplificado)
     - número da página
   - Ao final da resposta, inclua uma seção:
     "Fontes:"
     - arquivo X, página Y – breve descrição do que foi usado
     - arquivo Z, página W – breve descrição

5. Quando a pergunta misturar conceitos:
   - Se a resposta envolver mais de um manual, deixe claro o que veio de cada um.
   - Se houver diferenças ou limitações, mencione isso de forma honesta.

6. Se a pergunta estiver mal formulada:
   - Use o que existir no contexto.
   - Se ainda assim ficar ambígua, explique qual interpretação você está assumindo ou peça que a pergunta seja reformulada (mas ainda tente ajudar com o que tem).

Seu foco é: ser um “manual vivo” dos PDFs fornecidos, com respostas confiáveis, explicadas e bem referenciadas.
"""

def build_rag_prompt(question: str, chunks: List[Chunk]) -> str:
    context_blocks = []
    for c in chunks:
        filename = c.metadata.get("filename") or getattr(c, "source", "unknown")
        page = c.page if c.page is not None else "?"
        
        context_blocks.append(
            f"---\n[DOC_ID: {c.doc_id}] [ARQUIVO: {filename}] [PÁGINA: {page}]\n{c.text}\n"
        )
    
    context_text = "\n".join(context_blocks)

    final_prompt = f"""
[INSTRUÇÕES DO SISTEMA]
{SYSTEM_PROMPT}

[CONTEXTOS RELEVANTES DOS MANUAIS]

Abaixo estão trechos dos manuais que você PODE usar para responder.

{context_text}

[PERGUNTA DO USUÁRIO]

{question}

[INSTRUÇÕES FINAIS DE FORMATAÇÃO]

1. Responda em português do Brasil.
2. Comece com uma resposta direta em 1–3 frases.
3. Depois, se fizer sentido, detalhe em tópicos (condições, parâmetros, cuidados).
4. Ao final, inclua uma seção "Fontes:" listando arquivos e páginas usados.
"""
    return final_prompt

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
        try:
            self._load_model()
            
            prompt = build_rag_prompt(question, chunks)
            
            inputs = LocalLLMClient._tokenizer(prompt, return_tensors="pt")
            inputs = {k: v.to(LocalLLMClient._model.device) for k, v in inputs.items()}
            
            outputs = LocalLLMClient._model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                do_sample=True,
                pad_token_id=LocalLLMClient._tokenizer.eos_token_id
            )
            
            answer = LocalLLMClient._tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            if "[INSTRUÇÕES FINAIS DE FORMATAÇÃO]" in answer:
                parts = answer.split("[INSTRUÇÕES FINAIS DE FORMATAÇÃO]")
                if len(parts) > 1:
                    pass
            
            input_length = inputs["input_ids"].shape[1]
            generated_tokens = outputs[0][input_length:]
            answer = LocalLLMClient._tokenizer.decode(generated_tokens, skip_special_tokens=True)
            
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
        try:
            prompt = build_rag_prompt(question, chunks)
            
            response = self.model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    max_output_tokens=1024,
                    temperature=0.1,
                )
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
