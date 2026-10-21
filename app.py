

from datetime import datetime
import sqlite3
import pandas as pd
import spacy
import streamlit as st
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Configuração da Página do Streamlit
st.set_page_config(
    page_title="Análise de Sentimentos NLP", page_icon="🧠", layout="centered"
)


# Carregar modelos (spaCy e VADER) com cache para otimizar performance
@st.cache_resource
def load_models():
  nlp = spacy.load("pt_core_news_sm")
  analyzer = SentimentIntensityAnalyzer()
  return nlp, analyzer


nlp, analyzer = load_models()


# Configuração e inicialização do Banco de Dados SQLite3
def init_db():
  conn = sqlite3.connect("sentimentos.db")
  cursor = conn.cursor()
  cursor.execute("""
        CREATE TABLE IF NOT EXISTS historico (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            data_hora TEXT,
            texto TEXT,
            sentimento TEXT,
            compound REAL
        )
    """)
  conn.commit()
  conn.close()


# Função para salvar no banco SQLite
def salvar_no_banco(texto, sentimento, compound):
  conn = sqlite3.connect("sentimentos.db")
  cursor = conn.cursor()
  data_hora = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
  cursor.execute(
      """
        INSERT INTO historico (data_hora, texto, sentimento, compound)
        VALUES (?, ?, ?, ?)
    """,
      (data_hora, texto, sentimento, compound),
  )
  conn.commit()
  conn.close()


# Inicializar o banco ao iniciar o app
init_db()

# --- Interface Gráfica (Streamlit) ---
st.title("🧠 Processamento de Linguagem Natural com spaCy")
st.write(
    "Insira um texto para realizar o fatiamento (remoção de *stopwords* e"
    " pontuação), análise de sentimento e salvamento automático no SQLite."
)

# Input do usuário
texto_input = st.text_area(
    "Digite seu texto aqui:",
    placeholder="Ex: Eu estou extremamente feliz com os resultados do projeto!",
)

if st.button("Analisar e Salvar", type="primary"):
  if texto_input.strip():
    # 1. Processamento com spaCy (Fatiamento e Limpeza)
    doc = nlp(texto_input)

    # Filtrando tokens: removendo stopwords, pontuação e espaços em branco
    tokens_filtrados = [
        token.text
        for token in doc
        if not token.is_stop and not token.is_punct and not token.is_space
    ]

    st.json(tokens_filtrados)
    texto_processado = " ".join(tokens_filtrados)



    # 2. Análise de Sentimento (VADER -> extração do Compound)
    scores = analyzer.polarity_scores(texto_input)
    compound = scores["compound"]

    # Classificação baseada no compound score
    if compound >= 0.05:
      sentimento = "Positivo"
    elif compound <= -0.05:
      sentimento = "Negativo"
    else:
      sentimento = "Neutro"

    # 3. Salvar no SQLite3
    salvar_no_banco(texto_input, sentimento, compound)

    # --- Exibição dos Resultados ---
    st.success("Análise concluída e salva no banco de dados com sucesso!")

    col1, col2 = st.columns(2)
    with col1:
      st.metric(label="Sentimento Predominante", value=sentimento)
    with col2:
      st.metric(label="Compound Score", value=f"{compound:.4f}")

    st.subheader("Texto Fatiado (Sem Palavras Desnecessárias):")
    st.write(
        texto_processado
        if texto_processado
        else "*Nenhum token relevante restante após a filtragem.*"
    )

  else:
    st.warning("Por favor, insira um texto válido antes de analisar.")

# --- Seção de Histórico do Banco de Dados ---
st.divider()
st.subheader("📊 Histórico de Análises Recentes (SQLite3)")

if st.button("Atualizar Histórico"):
  pass

# Carregar dados do SQLite para exibição
try:
  conn = sqlite3.connect("sentimentos.db")
  df = pd.read_sql(
      "SELECT data_hora, texto, sentimento, compound FROM historico ORDER BY id"
      " DESC LIMIT 10",
      conn,
  )
  conn.close()

  if not df.empty:
    st.dataframe(df, use_container_width=True)
  else:
    st.write("Nenhum registro encontrado no banco de dados.")
except Exception as e:
  st.error(f"Erro ao carregar o histórico: {e}")