# app.py

import streamlit as st
import pandas as pd
import io
import re
import time
import gc

# --- Importaciones de PNL (Machine Learning) ---
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding
from sentence_transformers import SentenceTransformer
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm

# ===============================================================
# CONFIGURACIÓN INICIAL DE LA PÁGINA Y MODELOS
# ===============================================================

st.set_page_config(
    page_title="Analizador de Tono y Tema",
    page_icon="🚀",
    layout="wide"
)

# --- Configuración de Modelos y Parámetros ---
# Usamos los mismos modelos y parámetros optimizados del notebook
MODELO_SENTIMIENTO = "cardiffnlp/twitter-xlm-roberta-base-sentiment-multilingual"
MODELO_EMBEDDING = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
BATCH_SENT = 256  # Reducido para ser más amigable con CPUs y VRAM más limitada
BATCH_EMB = 256   # Reducido
MAX_LEN_SENT = 128
MAX_CHARS_CLEAN = 400
EXACT_PREFIX_LEN = 80
MIN_PREFIX_LEN = 20
NUM_WORKERS = 0 # En Streamlit, es mejor usar 0 para evitar problemas de paralelismo

# --- Detección de Dispositivo (GPU o CPU) ---
@st.cache_data
def get_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        return device, "GPU"
    else:
        return torch.device("cpu"), "CPU"

device, device_name = get_device()

# ===============================================================
# CACHEO DE MODELOS (CRUCIAL PARA EL RENDIMIENTO)
# ===============================================================

@st.cache_resource
def load_sentiment_model():
    tokenizer = AutoTokenizer.from_pretrained(MODELO_SENTIMIENTO)
    model = AutoModelForSequenceClassification.from_pretrained(MODELO_SENTIMIENTO)
    model.to(device)
    if device_name == "GPU":
        model.half()  # Usar FP16 solo en GPU
    model.eval()
    return tokenizer, model

@st.cache_resource
def load_embedding_model():
    encoder = SentenceTransformer(MODELO_EMBEDDING, device=device)
    encoder.max_seq_length = 128
    return encoder

# ===============================================================
# DICCIONARIOS Y FUNCIONES DE PROCESAMIENTO
# (Adaptadas del notebook original)
# ===============================================================

TOPIC_KEYWORDS = {
    "Políticas Públicas y Gobierno": ["reforma","decreto","ley","plan nacional","conpes","congreso","gobierno","política"],
    "Gestión Presupuestaria y Recursos": ["presupuesto","licitación","contrato","inversión","vigencias","adición","recursos"],
    "Salud Pública y Sanidad": ["hospital","vacunación","salud pública","epidemia","covid","ips","eps","medicina"],
    "Educación y Desarrollo Académico": ["colegios","universidades","icfes","becas","educación","saber","estudiantes"],
    "Seguridad Ciudadana y Justicia": ["policía","delito","captura","homicidio","fiscalía","juzgado","crimen"],
    "Economía, Empleo y Desarrollo": ["pib","inflación","empleo","desempleo","pymes","exportaciones","economía"],
    "Infraestructura y Obras Públicas": ["vía","doble calzada","intercambiador","vivienda","energía","metro","obra"],
    "Relaciones Exteriores y Cooperación": ["cancillería","acuerdo bilateral","cooperación","embajada","internacional"],
    "Medio Ambiente y Sostenibilidad": ["deforestación","emisiones","renovables","biodiversidad","clima","ambiente"],
    "Transparencia y Participación Ciudadana": ["veeduría","rendición de cuentas","corrupción","participación","transparencia"],
}
TOPIC_LIST = list(TOPIC_KEYWORDS.keys())

# --- Funciones de Utilidad (Limpieza, Agrupamiento) ---
def clean_text_basic(text):
    if not isinstance(text, str): return ""
    t = text.strip().replace("\n", " ")
    t = re.sub(r"[\x00-\x1f\x7f]", " ", t)
    t = re.sub(r"\s+", " ", t)
    if len(t) > MAX_CHARS_CLEAN:
        t = t[:MAX_CHARS_CLEAN].rsplit(" ", 1)[0] if " " in t[:MAX_CHARS_CLEAN] else t[:MAX_CHARS_CLEAN]
    return t

def normalize_for_prefix(text):
    t = text.lower().strip()
    t = re.sub(r"https?://\S+|www\.\S+", " ", t)
    t = re.sub(r"[^\w\sáéíóúüñ]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t

def agrupar_por_prefijo(textos):
    grupos, mapa = [], np.empty(len(textos), dtype=np.int32)
    idx_por_pref = {}
    for i, txt in enumerate(textos):
        pref = normalize_for_prefix(txt)[:EXACT_PREFIX_LEN]
        if len(pref) < MIN_PREFIX_LEN: pref = f"_short_{i}"
        gid = idx_por_pref.get(pref)
        if gid is None:
            gid = len(grupos)
            idx_por_pref[pref] = gid
            grupos.append([i])
        else:
            grupos[gid].append(i)
        mapa[i] = gid
    reps = [max(miembros, key=lambda k: len(textos[k])) for miembros in grupos]
    return reps, mapa

class TextDatasetFast(Dataset):
    def __init__(self, textos, tokenizer, max_len):
        self.textos, self.tok, self.max_len = textos, tokenizer, max_len
    def __len__(self): return len(self.textos)
    def __getitem__(self, idx):
        return self.tok(self.textos[idx], truncation=True, max_length=self.max_len, return_tensors=None)

# --- Funciones de Inferencia (Corazón del Análisis) ---
@torch.inference_mode()
def inferir_sentimiento(reps, progress_bar):
    tokenizer, model = load_sentiment_model()
    dataset = TextDatasetFast(reps, tokenizer, MAX_LEN_SENT)
    collator = DataCollatorWithPadding(tokenizer=tokenizer, pad_to_multiple_of=8)
    dataloader = DataLoader(dataset, batch_size=BATCH_SENT, shuffle=False, num_workers=NUM_WORKERS, collate_fn=collator)
    
    all_preds = []
    total_batches = len(dataloader)
    for i, batch in enumerate(dataloader):
        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
        if device_name == "GPU":
            with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                logits = model(**batch).logits
        else: # CPU no soporta autocast con float16
            logits = model(**batch).logits
        
        preds = logits.argmax(dim=-1).cpu().numpy()
        all_preds.extend(preds)
        progress_bar.progress((i + 1) / total_batches, text="Analizando sentimiento...")

    label_map = {0: "Negativo", 1: "Neutro", 2: "Positivo"}
    return [label_map[int(p)] for p in all_preds]

@torch.inference_mode()
def inferir_temas(reps, progress_bar):
    encoder = load_embedding_model()
    frases = [f"passage: {tema}. {' '.join(kws[:5])}" for tema, kws in TOPIC_KEYWORDS.items()]
    centroides = encoder.encode(frases, batch_size=32, show_progress_bar=False, convert_to_tensor=True, normalize_embeddings=True, device=device)
    
    texts_prefixed = [f"passage: {t}" for t in reps]
    vecs = encoder.encode(texts_prefixed, batch_size=BATCH_EMB, show_progress_bar=False, convert_to_tensor=True, normalize_embeddings=True, device=device)
    progress_bar.progress(1.0, text="Clasificando temas...")
    
    similarities = torch.matmul(vecs, centroides.T)
    tema_ids = similarities.argmax(dim=1).cpu().numpy()
    return [TOPIC_LIST[int(idx)] for idx in tema_ids]

# --- Función para convertir DataFrame a Excel en memoria ---
def to_excel(df):
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name='Resultados')
    processed_data = output.getvalue()
    return processed_data

# ===============================================================
# INTERFAZ DE STREAMLIT
# ===============================================================

st.title("🔗📊 Analizador de Tono y Tema para Textos")
st.markdown("Una herramienta para concatenar columnas de un Excel y analizar el texto resultante.")

# --- Inicialización del estado de la sesión ---
if 'step' not in st.session_state:
    st.session_state.step = "upload"
    st.session_state.df_original = None
    st.session_state.df_processed = None
    st.session_state.df_analyzed = None

# --- PASO 1: Cargar y Concatenar ---
st.header("Paso 1: Cargar y preparar el archivo")

uploaded_file = st.file_uploader("Sube tu archivo Excel (.xlsx)", type=["xlsx"])

if uploaded_file is not None:
    try:
        df = pd.read_excel(uploaded_file)
        st.session_state.df_original = df
        st.success(f"✅ Archivo '{uploaded_file.name}' cargado con {df.shape[0]} filas.")
        
        with st.expander("Ver columnas y primeras filas del archivo original"):
            st.dataframe(df.head())

        st.subheader("Selecciona las columnas a unir")
        st.info("El texto de estas columnas se unirá en una nueva columna llamada 'resumen'.")
        
        options = st.multiselect(
            "Columnas para concatenar:",
            options=df.columns.tolist(),
            help="Usa Ctrl+Click o Cmd+Click para seleccionar varias."
        )

        if st.button("🔗 Concatenar Columnas"):
            if len(options) < 1:
                st.warning("⚠️ Debes seleccionar al menos una columna.")
            else:
                with st.spinner("Procesando..."):
                    df_copy = df.copy()
                    
                    def acortar_texto(texto, limite=80):
                        palabras = str(texto).split()
                        return ' '.join(palabras[:limite]) + ('...' if len(palabras) > limite else '')

                    df_copy['resumen'] = df_copy[options].fillna('').astype(str).apply(lambda row: ' '.join(row), axis=1)
                    df_copy['resumen'] = df_copy['resumen'].apply(acortar_texto)

                    st.session_state.df_processed = df_copy
                    st.session_state.step = "analyze"
                    st.success("¡Columnas concatenadas exitosamente en la columna 'resumen'!")

    except Exception as e:
        st.error(f"❌ Error al leer el archivo. Asegúrate de que es un .xlsx válido. Detalle: {e}")

# --- PASO 2: Analizar ---
if st.session_state.step == "analyze":
    st.header("Paso 2: Analizar Tono y Tema")
    
    st.dataframe(st.session_state.df_processed[['resumen']].head())
    
    st.info(f"Se analizarán {len(st.session_state.df_processed)} textos. Dispositivo detectado: **{device_name}**")
    if device_name == "CPU":
        st.warning("⚠️ **Advertencia:** Estás usando CPU. El análisis puede ser muy lento para archivos grandes. Para máxima velocidad, ejecuta esta aplicación en un entorno con GPU NVIDIA.")

    if st.button("🚀 Analizar Ahora", type="primary"):
        st.session_state.step = "processing"
        st.rerun() # Volver a ejecutar para entrar en el estado de procesamiento

if st.session_state.step == "processing":
    with st.spinner("Iniciando análisis... Esto puede tardar varios minutos."):
        df_to_analyze = st.session_state.df_processed
        textos = df_to_analyze['resumen'].fillna('').astype(str).map(clean_text_basic).tolist()
        
        st.write("Agrupando textos similares para optimizar...")
        reps, mapa = agrupar_por_prefijo(textos)
        st.write(f"Reducción: {len(textos)} textos originales -> {len(reps)} grupos únicos para procesar.")

        progress_bar = st.progress(0, text="Análisis en curso...")
        
        # Inferencia
        start_time = time.time()
        tonos_rep = inferir_sentimiento(reps, progress_bar)
        temas_rep = inferir_temas(reps, progress_bar)
        
        # Propagar resultados
        n = len(df_to_analyze)
        df_to_analyze['tono'] = [tonos_rep[int(mapa[i])] for i in range(n)]
        df_to_analyze['tema'] = [temas_rep[int(mapa[i])] for i in range(n)]
        
        end_time = time.time()
        
        st.session_state.df_analyzed = df_to_analyze
        st.session_state.processing_time = end_time - start_time
        st.session_state.step = "results"
        
        # Limpiar memoria de GPU/RAM
        gc.collect()
        if device_name == "GPU":
            torch.cuda.empty_cache()
            
        st.rerun()

# --- PASO 3: Resultados y Descarga ---
if st.session_state.step == "results":
    st.header("Paso 3: Resultados del Análisis")
    
    df_final = st.session_state.df_analyzed
    total_time = st.session_state.processing_time
    
    st.success(f"✅ ¡Análisis completado en {total_time:.2f} segundos! ({len(df_final)/total_time:.1f} textos/s)")
    
    st.subheader("Vista Previa de los Resultados")
    st.dataframe(df_final[['resumen', 'tono', 'tema']].head())
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Distribución de Tonos")
        tono_counts = df_final['tono'].value_counts()
        st.bar_chart(tono_counts)
        st.write(tono_counts)
    
    with col2:
        st.subheader("Distribución de Temas")
        tema_counts = df_final['tema'].value_counts()
        st.bar_chart(tema_counts)
        st.write(tema_counts)
        
    st.subheader("Descargar Resultados")
    excel_data = to_excel(df_final)
    st.download_button(
        label="📥 Descargar archivo Excel completo",
        data=excel_data,
        file_name="analisis_tono_tema.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
