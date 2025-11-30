import streamlit as st
import requests
import os

st.set_page_config(
    page_title="Chat - RAG System",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    .user-message {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white !important;
        padding: 1.2rem;
        border-radius: 15px;
        margin: 0.8rem 0;
        text-align: right;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    .assistant-message {
        background: white;
        color: #1e293b !important;
        padding: 1.2rem;
        border-radius: 15px;
        margin: 0.8rem 0;
        border-left: 4px solid #667eea;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    .assistant-message strong {
        color: #667eea !important;
    }
    
    .source-box {
        background: #f8fafc;
        border: 2px solid #667eea;
        border-radius: 10px;
        padding: 0.8rem;
        margin: 0.5rem 0;
        font-size: 0.9rem;
        color: #1e293b !important;
    }
    
    .source-box strong {
        color: #667eea !important;
    }
    
    .source-box em {
        color: #475569 !important;
    }
    
    section[data-testid="stSidebar"] {
        background: rgba(255, 255, 255, 0.95) !important;
    }
    
    section[data-testid="stSidebar"] * {
        color: #1e293b !important;
    }
    
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white !important;
        border: none;
        border-radius: 10px;
        padding: 0.5rem 2rem;
        font-weight: bold;
        transition: all 0.3s;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
    }
    
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    h1 {
        color: white !important;
        text-align: center;
        font-weight: 800;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
    }
    
    .subtitle {
        text-align: center;
        color: white !important;
        font-size: 1.2rem;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.2);
    }
    
    .streamlit-expanderHeader {
        background-color: #f8fafc !important;
        color: #1e293b !important;
        border-radius: 8px;
    }
    
    .stTextInput input {
        color: #1e293b !important;
    }
    
    [data-testid="stFileUploader"] section > div > div > span {
        display: none;
    }
    
    [data-testid="stFileUploader"] section > div > div::after {
        content: "Arraste e solte arquivos aqui";
        color: white;
        font-size: 1rem;
        display: block;
        margin-bottom: 10px;
    }
    
    [data-testid="stFileUploader"] small {
        display: none;
    }
    
    [data-testid="stFileUploader"] section > div > div > small::after {
        content: "Limite de 200MB por arquivo • PDF";
        color: rgba(255, 255, 255, 0.8);
        font-size: 0.8rem;
        display: block;
    }
    
    [data-testid="stFileUploader"] button {
        color: transparent !important;
        position: relative;
    }
    
    [data-testid="stFileUploader"] button::after {
        content: "Carregar";
        color: white;
        position: absolute;
        left: 50%;
        top: 50%;
        transform: translate(-50%, -50%);
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")

if "messages" not in st.session_state:
    st.session_state.messages = []
if "uploaded_docs" not in st.session_state:
    st.session_state.uploaded_docs = []
    try:
        response = requests.get(f"{API_BASE_URL}/documents", timeout=2)
        if response.status_code == 200:
            st.session_state.uploaded_docs = response.json()
    except:
        pass

st.title("🤖 TRACTIAN RAG - Assistente Técnico")
st.markdown("<p class='subtitle'>Faça perguntas sobre os manuais técnicos</p>", unsafe_allow_html=True)

with st.sidebar:
    st.header("📄 Gerenciar Documentos")
    
    st.subheader("Upload de PDFs")
    uploaded_files = st.file_uploader(
        "Arraste seus PDFs aqui",
        type=['pdf'],
        accept_multiple_files=True,
        help="Faça upload dos manuais técnicos em PDF"
    )
    
    if uploaded_files:
        new_files = [f for f in uploaded_files if f.name not in st.session_state.uploaded_docs]
        
        if new_files:
            with st.spinner(f"Processando {len(new_files)} arquivo(s)..."):
                try:
                    files = [("files", (f.name, f.getvalue(), "application/pdf")) for f in new_files]
                    response = requests.post(f"{API_BASE_URL}/documents", files=files)
                    
                    if response.status_code == 200:
                        data = response.json()
                        for f in new_files:
                            st.session_state.uploaded_docs.append(f.name)
                        st.success(f"✅ {data['documents_indexed']} documento(s) indexado(s) ({data['total_chunks']} chunks)")
                    else:
                        st.error(f"❌ Erro ao processar arquivos: {response.text}")
                except Exception as e:
                    st.error(f"❌ Erro: {str(e)}")
    
    if st.session_state.uploaded_docs:
        st.subheader("📚 Documentos Indexados")
        for doc in st.session_state.uploaded_docs:
            st.markdown(f"✓ {doc}")
    
    st.divider()
    st.subheader("⚙️ Modelo de Resposta")
    
    llm_option = st.radio(
        "Escolha o modelo:",
        options=["local", "gemini"],
        format_func=lambda x: "Local LLM (GPU/CPU)" if x == "local" else "Gemini (API Google)",
        index=0,
        help="Local: roda na máquina. Gemini: usa API da Google."
    )
    
    if llm_option == "local":
        st.caption("🚀 Mistral-7B com 4-bit quantization")
        st.caption("⚡ GPU detectada automaticamente")
    else:
        st.caption("✨ Gemini 2.0 Flash (rápido e barato)")
    
    if st.button("🗑️ Limpar Conversa"):
        st.session_state.messages = []
        st.rerun()

st.markdown("---")

for message in st.session_state.messages:
    role = message["role"]
    
    if role == "user":
        content = message.get("content", "")
        st.markdown(f"""
        <div class="user-message">
            <strong>Você:</strong><br>
            {content}
        </div>
        """, unsafe_allow_html=True)
    else:
        answer = message.get("answer", message.get("content", ""))
        references = message.get("references", [])
        provider_used = message.get("provider_used", "")
        fallback_from = message.get("fallback_from")
        
        st.markdown(f"""
        <div class="assistant-message">
            <strong>🤖 Assistente:</strong><br>
            {answer}
        </div>
        """, unsafe_allow_html=True)
        
        if provider_used:
            provider_label = "Local" if provider_used == "local" else "Gemini"
            if fallback_from:
                fallback_label = "Local" if fallback_from == "local" else "Gemini"
                st.caption(f"⚠️ Tentou usar {fallback_label}, mas usou fallback → {provider_label}")
            else:
                st.caption(f"🤖 Respondido por: **{provider_label}**")
        
        if references:
            with st.expander("📚 Ver Fontes"):
                for idx, ref in enumerate(references, 1):
                    if isinstance(ref, dict):
                        filename = ref.get('filename', 'Documento')
                        page = ref.get('page', '?')
                        snippet = ref.get('snippet', '')
                        st.markdown(f"""
                        <div class="source-box">
                            📄 <strong>{filename}</strong> - Página {page}<br>
                            <em>{snippet[:150]}...</em>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div class="source-box">
                            📄 <strong>Referência {idx}:</strong><br>
                            <em>{ref}</em>
                        </div>
                        """, unsafe_allow_html=True)

if not st.session_state.uploaded_docs:
    st.warning("⚠️ Por favor, faça upload de pelo menos um PDF na barra lateral para começar!")
else:
    question = st.chat_input("Digite sua pergunta sobre os documentos...")
    
    if question:
        st.session_state.messages.append({"role": "user", "content": question})
        
        st.markdown(f"""
        <div class="user-message">
            <strong>Você:</strong><br>
            {question}
        </div>
        """, unsafe_allow_html=True)
        
        with st.spinner("🤔 Pensando..."):
            try:
                response = requests.post(
                    f"{API_BASE_URL}/question",
                    json={
                        "question": question,
                        "llm_provider": llm_option
                    }
                )
                
                if response.status_code == 200:
                    data = response.json()
                    answer = data.get("answer", "Sem resposta")
                    references = data.get("references", [])
                    provider_used = data.get("provider_used", "unknown")
                    fallback_from = data.get("fallback_from")
                    
                    st.session_state.messages.append({
                        "role": "assistant",
                        "answer": answer,
                        "references": references,
                        "provider_used": provider_used,
                        "fallback_from": fallback_from
                    })
                    
                    st.rerun()
                else:
                    st.error(f"❌ Erro: {response.status_code} - {response.text}")
            except Exception as e:
                st.error(f"❌ Erro ao conectar com a API: {str(e)}")

st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #667eea; padding: 1rem;'>
    <p>💡 <strong>Dica:</strong> Faça perguntas específicas sobre os manuais técnicos</p>
</div>
""", unsafe_allow_html=True)
