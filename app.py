

import sqlite3
from datetime import datetime
import streamlit as st
import spacy

# -----------------------------------------------------------------------------
# 1. Configurações da Página e Recursos
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Processamento e Análise de Sentimento",
    page_icon="🇧🇷",
    layout="wide"
)

@st.cache_resource
def carregar_spacy():
    """Carrega o modelo do spaCy para Português."""
    return spacy.load("pt_core_news_sm")

nlp = carregar_spacy()

# Dicionário léxico especializado em Português para polaridade de sentimentos
DICIONARIO_SENTIMENTO = {
    "feliz": 0.8, "ótimo": 0.9, "excelente": 1.0, "bom": 0.6, "maravilhoso": 0.9,
    "gostei": 0.7, "satisfeito": 0.7, "perfeito": 1.0, "adoro": 0.8, "alegre": 0.7,
    "fantástico": 0.9, "recomendo": 0.8, "rápido": 0.5, "eficiente": 0.7,
    "triste": -0.7, "ruim": -0.6, "péssimo": -1.0, "horroroso": -0.9, "odiei": -0.9,
    "insatisfeito": -0.7, "lento": -0.4, "problema": -0.5, "defeito": -0.6, "chato": -0.5,
    "atrasou": -0.6, "horrível": -0.9, "pior": -0.8, "falha": -0.6
}

# -----------------------------------------------------------------------------
# 2. Banco de Dados SQLite3
# -----------------------------------------------------------------------------
DB_NAME = "historico_ptbr.db"

def init_db():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS analises (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            data_hora TEXT NOT NULL,
            texto_original TEXT NOT NULL,
            sentimento TEXT NOT NULL,
            compound REAL NOT NULL
        )
    """)
    conn.commit()
    conn.close()

def salvar_analise(data_hora, texto, sentimento, compound):
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO analises (data_hora, texto_original, sentimento, compound)
        VALUES (?, ?, ?, ?)
    """, (data_hora, texto, sentimento, compound))
    conn.commit()
    conn.close()

def buscar_historico():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("SELECT data_hora, texto_original, sentimento, compound FROM analises ORDER BY id DESC")
    dados = cursor.fetchall()
    conn.close()
    return dados

init_db()

# -----------------------------------------------------------------------------
# 3. Processamento e Análise de Sentimento em Português
# -----------------------------------------------------------------------------
def processar_e_analisar(texto):
    """
    Fatia o texto com spaCy, remove ruídos e calcula a polaridade 
    utilizando o léxico nativo em Português.
    """
    doc = nlp(texto)
    
    # Tokens filtrados (sem stop words, pontuação, etc.)
    tokens_filtrados = [
        token.text for token in doc 
        if not token.is_stop and not token.is_punct and not token.like_num and not token.is_space
    ]
    
    # Cálculo do score compound com base nas lematizações e palavras detetadas
    pontuacoes = []
    for token in doc:
        lema = token.lemma_.lower()
        palavra = token.text.lower()
        if lema in DICIONARIO_SENTIMENTO:
            pontuacoes.append(DICIONARIO_SENTIMENTO[lema])
        elif palavra in DICIONARIO_SENTIMENTO:
            pontuacoes.append(DICIONARIO_SENTIMENTO[palavra])
            
    if not pontuacoes:
        compound = 0.0
    else:
        compound = sum(pontuacoes) / max(len(pontuacoes), 1)
        compound = max(min(compound, 1.0), -1.0) # Normaliza entre -1.0 e 1.0

    # Classificação final
    if compound >= 0.05:
        sentimento = "Positivo"
    elif compound <= -0.05:
        sentimento = "Negativo"
    else:
        sentimento = "Neutro"
        
    return doc, tokens_filtrados, sentimento, compound

# -----------------------------------------------------------------------------
# 4. Interface Gráfica (Streamlit)
# -----------------------------------------------------------------------------
st.title("🇧🇷 Processamento de Texto e Análise de Sentimento em PT-BR")
st.markdown("Aplicação com **spaCy**, **Léxico Nativo em Português** e **SQLite3**.")

aba1, aba2 = st.tabs(["📝 Nova Análise", "📊 Histórico SQLite3"])

with aba1:
    texto_input = st.text_area(
        "Digite seu texto em português:",
        height=130,
        placeholder="Ex: Eu estou muito feliz e satisfeito com este serviço!"
    )
    
    if st.button("Analisar e Salvar", type="primary"):
        if not texto_input.strip():
            st.warning("Por favor, introduza um texto válido.")
        else:
            # Executa o processamento e a análise de sentimento
            doc, tokens_limpos, sentimento, compound = processar_e_analisar(texto_input)
            
            # Registo de data e hora
            data_hora_atual = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            # Gravação no SQLite3
            salvar_analise(data_hora_atual, texto_input, sentimento, compound)
            st.success("Análise armazenada com sucesso na base de dados!")
            
            # Exibição dos resultados em colunas
            col_esq, col_dir = st.columns(2)
            
            with col_esq:
                st.subheader("✂️ Fatiamento (spaCy PT)")
                st.write(f"**Total de tokens originais:** {len(doc)}")
                st.write("**Tokens filtrados:**")
                st.json(tokens_limpos)
                
            with col_dir:
                st.subheader("📊 Métrica de Sentimento")
                st.metric(label="Classificação", value=sentimento)
                st.metric(label="Métrica Compound", value=f"{compound:.4f}")
            
            # Detalhes morfossintáticos
            with st.expander("🔍 Ver análise detalhada dos tokens (spaCy)"):
                tabela_tokens = [
                    {
                        "Token": token.text,
                        "Lema": token.lemma_,
                        "Classe (POS)": token.pos_,
                        "É Stop Word?": token.is_stop
                    }
                    for token in doc
                ]
                st.dataframe(tabela_tokens, use_container_width=True)

with aba2:
    st.subheader("📜 Histórico Armazenado")
    dados = buscar_historico()
    
    if dados:
        historico_df = [
            {
                "Data/Hora": row[0],
                "Texto Input": row[1],
                "Sentimento": row[2],
                "Compound": row[3]
            }
            for row in dados
        ]
        st.dataframe(historico_df, use_container_width=True)
    else:
        st.info("Ainda não existem registos na base de dados.")
