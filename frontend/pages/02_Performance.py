import streamlit as st
import os
import pandas as pd
import requests
import plotly.express as px
import plotly.graph_objects as go
import time
from datetime import datetime

st.set_page_config(page_title="Performance Metrics", page_icon="📊", layout="wide")

st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    h1, h2, h3 {
        color: white !important;
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: rgba(255, 255, 255, 0.1);
        color: white;
        border-radius: 8px;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: white;
        color: #667eea;
    }
</style>
""", unsafe_allow_html=True)

API_URL = os.getenv("API_BASE_URL", "http://localhost:8000")

if 'metrics_history' not in st.session_state:
    st.session_state.metrics_history = []

@st.cache_data(ttl=2)
def get_metrics():
    try:
        response = requests.get(f"{API_URL}/metrics", timeout=1)
        if response.status_code == 200:
            return response.json()
    except requests.exceptions.ConnectionError:
        return {"error": "connection_refused"}
    except requests.exceptions.Timeout:
        return {"error": "timeout"}
    except Exception as e:
        return {"error": str(e)}
    return None

def reset_metrics():
    try:
        requests.post(f"{API_URL}/metrics/reset")
        st.success("✅ Métricas resetadas com sucesso!")
        time.sleep(1)
        st.rerun()
    except:
        st.error("❌ Falha ao resetar métricas")

st.title("📊 System Performance & Telemetry")

col_refresh1, col_refresh2 = st.columns([3, 1])
with col_refresh1:
    auto_refresh = st.checkbox("Auto-refresh (5s)", value=False)
with col_refresh2:
    if st.button("🔄 Refresh Now"):
        st.rerun()

if auto_refresh:
    st_autorefresh = st.empty()
    with st_autorefresh:
        st.caption("Auto-refreshing...")
    time.sleep(5)
    st.rerun()

metrics = get_metrics()

if metrics and metrics.get("error"):
    error_type = metrics.get("error")
    if error_type == "connection_refused":
        st.error("❌ **Backend API não está respondendo!**")
        st.info(f"💡 Certifique-se de que o backend está rodando em {API_URL}")
        st.code("python run_backend.py", language="bash")
    else:
        st.error(f"❌ Erro ao conectar com a API: {error_type}")
    st.stop()

if not metrics:
    st.error("❌ Não foi possível obter métricas do backend")
    st.stop()

current_sys = metrics.get("current_system", {})
if current_sys:
    if 'timestamp' not in current_sys:
        current_sys['timestamp'] = datetime.now().isoformat()
    st.session_state.metrics_history.append(current_sys)
    st.session_state.metrics_history = st.session_state.metrics_history[-50:]

col1, col2, col3, col4 = st.columns(4)

with col1:
    cpu = current_sys.get('cpu_percent', 0)
    st.metric("💻 CPU Usage", f"{cpu:.1f}%", 
              delta=None, delta_color="inverse")

with col2:
    ram = current_sys.get('ram_percent', 0)
    ram_gb = current_sys.get('ram_used_gb', 0)
    st.metric("🧠 RAM Usage", f"{ram:.1f}%", 
              delta=f"{ram_gb:.1f} GB")

with col3:
    gpu = current_sys.get("gpu", {})
    if gpu.get("available"):
        gpu_mem = gpu.get('memory_allocated_mb', 0)
        st.metric("🎮 GPU Memory", f"{gpu_mem:.0f} MB",
                  delta=gpu.get('name', 'GPU')[:15])
    else:
        st.metric("🎮 GPU", "Not Available", delta="CPU Mode")

with col4:
    if st.button("🗑️ Reset Metrics", width="stretch"):
        reset_metrics()

st.markdown("---")

tab1, tab2, tab3, tab4 = st.tabs(["📈 System Monitor", "🤖 LLM & Retrieval", "📄 Ingestion & Embeddings", "📊 Raw Data"])

with tab1:
    st.subheader("System Health Monitor")
    
    col_gauge1, col_gauge2 = st.columns(2)
    
    with col_gauge1:
        fig_cpu = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=cpu,
            title={'text': "CPU Usage (%)"},
            delta={'reference': 50},
            gauge={
                'axis': {'range': [None, 100]},
                'bar': {'color': "#667eea"},
                'steps': [
                    {'range': [0, 50], 'color': "lightgray"},
                    {'range': [50, 80], 'color': "lightyellow"},
                    {'range': [80, 100], 'color': "lightcoral"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 90
                }
            }
        ))
        fig_cpu.update_layout(height=300)
        st.plotly_chart(fig_cpu, width="stretch")
    
    with col_gauge2:
        fig_ram = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=ram,
            title={'text': "RAM Usage (%)"},
            delta={'reference': 50},
            gauge={
                'axis': {'range': [None, 100]},
                'bar': {'color': "#764ba2"},
                'steps': [
                    {'range': [0, 50], 'color': "lightgray"},
                    {'range': [50, 80], 'color': "lightyellow"},
                    {'range': [80, 100], 'color': "lightcoral"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 90
                }
            }
        ))
        fig_ram.update_layout(height=300)
        st.plotly_chart(fig_ram, width="stretch")
    
    if len(st.session_state.metrics_history) > 1:
        st.subheader("Historical Trends")
        df_history = pd.DataFrame(st.session_state.metrics_history)
        
        fig_trends = go.Figure()
        fig_trends.add_trace(go.Scatter(
            x=df_history.index,
            y=df_history['cpu_percent'],
            mode='lines+markers',
            name='CPU %',
            line=dict(color='#667eea', width=2)
        ))
        fig_trends.add_trace(go.Scatter(
            x=df_history.index,
            y=df_history['ram_percent'],
            mode='lines+markers',
            name='RAM %',
            line=dict(color='#764ba2', width=2)
        ))
        fig_trends.update_layout(
            title="Resource Usage Over Time",
            xaxis_title="Sample",
            yaxis_title="Usage (%)",
            hovermode='x unified',
            height=400
        )
        st.plotly_chart(fig_trends, width="stretch")
    else:
        st.info("📊 Ative o auto-refresh para ver tendências históricas")

with tab2:
    col_llm, col_ret = st.columns(2)
    
    with col_llm:
        st.subheader("🤖 LLM Performance")
        llm_data = metrics.get("llm", [])
        
        if llm_data and len(llm_data) > 0:
            df_llm = pd.DataFrame(llm_data)
            
            st.metric("Total Requests", len(df_llm))
            avg_duration = df_llm["duration_seconds"].mean()
            st.metric("Avg Response Time", f"{avg_duration:.2f}s")
            
            fig_lat = px.line(
                df_llm, 
                x=df_llm.index, 
                y="duration_seconds",
                title="LLM Response Time",
                labels={"index": "Request #", "duration_seconds": "Time (s)"}
            )
            fig_lat.update_traces(line_color='#667eea')
            st.plotly_chart(fig_lat, width="stretch")
            
            if "input_tokens" in df_llm.columns:
                fig_tok = go.Figure()
                fig_tok.add_trace(go.Bar(
                    x=df_llm.index,
                    y=df_llm["input_tokens"],
                    name="Input Tokens",
                    marker_color='lightskyblue'
                ))
                fig_tok.add_trace(go.Bar(
                    x=df_llm.index,
                    y=df_llm["output_tokens"],
                    name="Output Tokens",
                    marker_color='#764ba2'
                ))
                fig_tok.update_layout(
                    title="Token Usage Per Request",
                    xaxis_title="Request #",
                    yaxis_title="Tokens",
                    barmode='stack'
                )
                st.plotly_chart(fig_tok, width="stretch")
        else:
            st.metric("Total Requests", 0)
            st.metric("Avg Response Time", "--")
            st.info("📭 Nenhuma requisição LLM ainda. Faça uma pergunta no Chat para gerar dados!")
    
    with col_ret:
        st.subheader("🔍 Retrieval Quality")
        ret_data = metrics.get("retrieval", [])
        
        if ret_data and len(ret_data) > 0:
            df_ret = pd.DataFrame(ret_data)
            
            st.metric("Total Queries", len(df_ret))
            avg_score = df_ret["top_score"].mean()
            st.metric("Avg Top Score", f"{avg_score:.4f}")
            
            fig_score = px.histogram(
                df_ret, 
                x="top_score", 
                nbins=20,
                title="Similarity Score Distribution",
                labels={"top_score": "Score"}
            )
            fig_score.update_traces(marker_color='#667eea')
            st.plotly_chart(fig_score, width="stretch")
            
            fig_scores_time = px.scatter(
                df_ret,
                x=df_ret.index,
                y="top_score",
                title="Top Similarity Scores Over Time",
                labels={"index": "Query #", "top_score": "Score"}
            )
            fig_scores_time.update_traces(marker=dict(size=10, color='#764ba2'))
            st.plotly_chart(fig_scores_time, width="stretch")
        else:
            st.metric("Total Queries", 0)
            st.metric("Avg Top Score", "--")
            st.info("📭 Nenhuma busca realizada ainda. Faça perguntas no Chat!")

with tab3:
    st.subheader("📄 Ingestion & Embedding Pipeline")
    
    col_ing, col_emb = st.columns(2)
    
    with col_ing:
        st.markdown("**Document Ingestion**")
        ing_data = metrics.get("ingestion", [])
        
        if ing_data and len(ing_data) > 0:
            df_ing = pd.DataFrame(ing_data)
            
            st.metric("Documents Processed", len(df_ing))
            
            df_ing['filename_short'] = df_ing['filename'].apply(
                lambda x: x[:20] + '...' if len(x) > 20 else x
            )
            
            fig_ing = px.bar(
                df_ing,
                x="filename_short",
                y="duration_seconds",
                title="Processing Time per Document",
                labels={"duration_seconds": "Time (s)", "filename_short": "File"},
                hover_data={"filename": True, "filename_short": False}
            )
            fig_ing.update_traces(marker_color='#667eea')
            st.plotly_chart(fig_ing, width="stretch")
            
            if "chunks_count" in df_ing.columns:
                fig_chunks = px.bar(
                    df_ing,
                    x="filename_short",
                    y="chunks_count",
                    title="Chunks per Document",
                    labels={"filename_short": "File"},
                    hover_data={"filename": True, "filename_short": False}
                )
                fig_chunks.update_traces(marker_color='#764ba2')
                st.plotly_chart(fig_chunks, width="stretch")
        else:
            st.metric("Documents Processed", 0)
            st.info("📭 Nenhum documento processado ainda. Faça upload de PDFs!")
    
    with col_emb:
        st.markdown("**Embedding Generation**")
        emb_data = metrics.get("embeddings", [])
        
        if emb_data and len(emb_data) > 0:
            df_emb = pd.DataFrame(emb_data)
            
            st.metric("Embedding Batches", len(df_emb))
            avg_throughput = df_emb["throughput_items_per_sec"].mean()
            st.metric("Avg Throughput", f"{avg_throughput:.1f} items/s")
            
            fig_emb = px.scatter(
                df_emb,
                x="batch_size",
                y="duration_seconds",
                size="batch_size",
                title="Embedding Time vs Batch Size",
                labels={"duration_seconds": "Time (s)"}
            )
            fig_emb.update_traces(marker=dict(color='#764ba2'))
            st.plotly_chart(fig_emb, width="stretch")
            
            fig_throughput = px.line(
                df_emb,
                x=df_emb.index,
                y="throughput_items_per_sec",
                title="Embedding Throughput Over Time",
                labels={"index": "Batch #", "throughput_items_per_sec": "Items/sec"}
            )
            fig_throughput.update_traces(line_color='#667eea')
            st.plotly_chart(fig_throughput, width="stretch")
        else:
            st.metric("Embedding Batches", 0)
            st.metric("Avg Throughput", "--")
            st.info("📭 Nenhum embedding gerado ainda. Processe alguns documentos!")

with tab4:
    st.subheader("Raw Metrics Data")
    
    st.markdown("### Summary")
    col_s1, col_s2, col_s3, col_s4 = st.columns(4)
    with col_s1:
        st.metric("Ingestion Events", len(metrics.get("ingestion", [])))
    with col_s2:
        st.metric("Embedding Events", len(metrics.get("embeddings", [])))
    with col_s3:
        st.metric("Retrieval Events", len(metrics.get("retrieval", [])))
    with col_s4:
        st.metric("LLM Events", len(metrics.get("llm", [])))
    
    st.markdown("### Complete JSON Data")
    st.json(metrics)

st.markdown("---")
st.markdown("""
<div style='text-align: center; color: white; padding: 1rem;'>
    <p>🔄 Last updated: {}</p>
</div>
""".format(datetime.now().strftime("%H:%M:%S")), unsafe_allow_html=True)


