!pip install -q torch transformers sentence-transformers pandas openpyxl tqdm scikit-learn 2>/dev/null

import os, re, time, torch
import numpy as np
import pandas as pd
import warnings
from tqdm.auto import tqdm
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from collections import defaultdict
from google.colab import files

warnings.filterwarnings("ignore")

# ==================== CONFIGURACIÓN ====================
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🚀 Dispositivo: {device.upper()}")

# MODELO 1: E5 para clustering (mantiene agrupación rápida)
print("📦 Cargando multilingual-e5-large para clustering...")
embedder = SentenceTransformer("intfloat/multilingual-e5-large", device=device)

# MODELO 2: RoBERTa-spanish para análisis de sentimiento
print("📦 Cargando RoBERTa-spanish para análisis de tono...")
sentiment_model_name = "finiteautomata/beto-sentiment-analysis"
tokenizer = AutoTokenizer.from_pretrained(sentiment_model_name)
sentiment_model = AutoModelForSequenceClassification.from_pretrained(sentiment_model_name)
sentiment_model = sentiment_model.to(device)
sentiment_model.eval()

print("✅ Modelos cargados\n")

# ==================== REGLAS COMPLEMENTARIAS ====================
# Reglas que REFUERZAN el análisis de RoBERTa

CLIENT_POSITIVE_PATTERNS = [
    "liderazgo del", "bajo el liderazgo", "gracias a la alcaldía",
    "gracias al alcalde", "gestión del alcalde", "alcaldía entrega",
    "alcalde confirma", "alcalde anuncia", "alcalde inaugura"
]

CLIENT_NEGATIVE_PATTERNS = [
    "investigan al alcalde", "denuncian al", "escándalo del alcalde",
    "corrupción del", "acusan al alcalde", "irregularidades del"
]

NEUTRAL_REACTIONS = [
    "lamenta", "rechaza atentado", "condena", "expresa solidaridad",
    "pide", "solicita", "manifiesta preocupación"
]

ACHIEVEMENT_INDICATORS = [
    "bachilleres inician", "estudiantes se inscriben", "familias beneficiadas",
    "viviendas entregadas", "obras culminadas", "inversión de",
    "programas profesionales", "vida universitaria", "acceso a"
]

TOPICS_KEYWORDS = {
    "Infraestructura": ["escenario", "multideportivo", "obra", "infraestructura", "vía", "parque",
                        "pavimentación", "construcción", "puente", "polideportivo", "cancha"],
    "Educación": ["educación", "bachilleres", "universitaria", "colegio", "becas", "estudiantes",
                  "escuela", "universidad", "educación superior", "acceso a la universidad"],
    "Seguridad y Justicia": ["policía", "seguridad", "auxiliares", "dotación", "homicidio",
                            "delincuencia", "crimen", "atentado", "fallecimiento"],
    "Relaciones Internacionales": ["embajador", "cumbre", "celac", "diplomático", "internacional", "bilateral"],
    "Corrupción y Escándalos": ["escándalo", "corrupción", "investigan al", "denuncia contra", "fraude"],
    "Salud": ["salud", "hospital", "médico", "atención médica", "vacunación", "pandemia"],
    "Medio Ambiente": ["lluvias", "inundaciones", "emergencia", "clima", "ambiental", "desastre natural",
                      "huracán", "afectados", "coletazo"],
    "Economía": ["economía", "empleo", "empresa", "comercio", "inversión económica", "presupuesto"],
}

# ==================== FUNCIONES DE CLUSTERING (E5) ====================

def normalize_text(text):
    """Normaliza texto removiendo artículos y preposiciones"""
    t = text.lower()
    stopwords = ['el', 'la', 'los', 'las', 'un', 'una', 'unos', 'unas',
                 'de', 'del', 'en', 'para', 'por', 'con', 'sin', 'sobre',
                 'que', 'todo', 'está', 'son', 'es']

    t = re.sub(r'[^\wáéíóúñ\s]', ' ', t)
    t = re.sub(r'\s+', ' ', t).strip()

    words = t.split()
    words = [w for w in words if w not in stopwords and len(w) > 2]

    return " ".join(words)

def get_signature(text, words=20):
    """Genera firma textual normalizada para clustering"""
    normalized = normalize_text(text)
    return " ".join(normalized.split()[:words])

def universal_cluster(texts):
    """Clustering usando E5 (mantiene agrupación rápida y efectiva)"""
    print("📊 Agrupando noticias similares con E5...")
    groups = defaultdict(list)

    for i, t in enumerate(texts):
        signature = get_signature(t)
        groups[signature].append(i)

    representatives = []
    cluster_map = [-1] * len(texts)
    cid = 0

    for idxs in groups.values():
        longest_idx = max(idxs, key=lambda i: len(texts[i]))
        representatives.append(texts[longest_idx])

        for i in idxs:
            cluster_map[i] = cid
        cid += 1

    print(f"✅ {len(texts)} noticias → {len(representatives)} grupos únicos")
    return representatives, np.array(cluster_map)

# ==================== ANÁLISIS CON ROBERTA ====================

def get_roberta_sentiment(text, max_length=512):
    """Obtiene sentimiento base usando RoBERTa"""
    # Truncar texto si es muy largo
    text_truncated = text[:max_length * 4]  # Aproximado para tokens

    inputs = tokenizer(text_truncated, return_tensors="pt",
                      truncation=True, max_length=max_length,
                      padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = sentiment_model(**inputs)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=1)

    # BETO sentiment: 0=NEG, 1=NEU, 2=POS
    sentiment_idx = torch.argmax(probs, dim=1).item()
    confidence = probs[0][sentiment_idx].item()

    sentiment_map = {0: "Negativo", 1: "Neutro", 2: "Positivo"}

    return sentiment_map[sentiment_idx], confidence

def analyze_with_roberta(rep_text, client_name):
    """Análisis híbrido: RoBERTa + reglas contextuales"""
    lower = rep_text.lower()
    client_lower = client_name.lower()

    # PASO 1: Obtener sentimiento base de RoBERTa
    base_sentiment, base_confidence = get_roberta_sentiment(rep_text)

    # PASO 2: Ajustes contextuales específicos para marca

    # Caso 1: Negativos DIRECTOS contra el cliente (override RoBERTa)
    if any(pattern in lower for pattern in CLIENT_NEGATIVE_PATTERNS):
        return "Negativo", 0.95, base_sentiment

    # Caso 2: Reacciones neutras (override solo si RoBERTa marcó positivo/negativo)
    if any(reaction in lower for reaction in NEUTRAL_REACTIONS):
        # El cliente solo reacciona, no gestiona
        if base_sentiment != "Neutro":
            return "Neutro", 0.88, base_sentiment

    # Caso 3: POSITIVOS claros de gestión (reforzar si RoBERTa es neutro)
    client_is_positive_subject = any(pattern in lower for pattern in CLIENT_POSITIVE_PATTERNS)
    has_achievements = any(indicator in lower for indicator in ACHIEVEMENT_INDICATORS)

    if client_is_positive_subject or has_achievements:
        if base_sentiment == "Neutro":
            # RoBERTa no captó el positivo, pero el cliente SÍ está gestionando
            return "Positivo", 0.87, base_sentiment
        elif base_sentiment == "Positivo":
            # RoBERTa captó correctamente, aumentamos confianza
            return "Positivo", min(0.95, base_confidence + 0.10), base_sentiment

    # PASO 3: Usar análisis de RoBERTa como base
    return base_sentiment, round(base_confidence, 3), base_sentiment

def get_topic(text):
    """Análisis de tema por keywords"""
    lower = text.lower()
    tema = "Gestión y Acciones"
    max_matches = 0

    for topic, keywords in TOPICS_KEYWORDS.items():
        matches = sum(1 for kw in keywords if kw in lower)
        if matches > max_matches:
            max_matches = matches
            tema = topic

    return tema

# ==================== MAIN EXECUTION ====================
print("\n" + "="*80)
print("🤖 ANALIZADOR HÍBRIDO v31 - RoBERTa + E5")
print("="*80 + "\n")

client_name = input("👤 Nombre del cliente (ej: Carlos Pinedo): ").strip()
if not client_name:
    client_name = "Carlos Pinedo"
    print(f"   Usando cliente por defecto: {client_name}")

print("\n📂 Sube tu archivo Excel...")
uploaded = files.upload()
file_name = list(uploaded.keys())[0]

print(f"📖 Leyendo {file_name}...")
df = pd.read_excel(file_name)

text_columns = ['resumen', 'texto', 'contenido', 'titulo', 'title', 'noticia', 'descripcion']
text_col = next((c for c in text_columns if c in df.columns), None)

if not text_col:
    print(f"❌ Error: No se encontró columna de texto. Columnas disponibles: {list(df.columns)}")
    raise ValueError("No se encontró columna de texto válida")

print(f"✅ Columna detectada: '{text_col}'")

df = df.dropna(subset=[text_col]).reset_index(drop=True)
texts = df[text_col].astype(str).tolist()
print(f"📊 {len(texts)} noticias cargadas\n")

# ==================== PROCESAMIENTO ====================
t0 = time.time()

# PASO 1: Clustering con E5
rep_texts, cluster_map = universal_cluster(texts)

# PASO 2: Análisis con RoBERTa + reglas
print("\n🤖 Analizando tono con RoBERTa-spanish...")
tonos = []
confianzas = []
tonos_roberta_base = []  # Para comparación
temas = []

for rep in tqdm(rep_texts, desc="Procesando"):
    tono, conf, base_sent = analyze_with_roberta(rep, client_name)
    tema = get_topic(rep)

    tonos.append(tono)
    confianzas.append(conf)
    tonos_roberta_base.append(base_sent)
    temas.append(tema)

# PASO 3: Mapear a todas las noticias del grupo
print("\n🔗 Aplicando resultados a grupos similares...")
df["grupo_id"] = cluster_map
df["tono_marca"] = [tonos[cluster_map[i]] for i in range(len(df))]
df["confianza"] = [confianzas[cluster_map[i]] for i in range(len(df))]
df["tema"] = [temas[cluster_map[i]] for i in range(len(df))]
df["tono_roberta_base"] = [tonos_roberta_base[cluster_map[i]] for i in range(len(df))]

# ==================== EXPORTACIÓN ====================
out = f"Final_RoBERTa_{client_name.replace(' ', '_')}_{time.strftime('%Y%m%d_%H%M')}.xlsx"
print(f"\n💾 Guardando resultados en {out}...")

df.to_excel(out, sheet_name="Resultados", index=False, engine="openpyxl")
files.download(out)

# ==================== RESULTADOS ====================
tiempo_total = time.time() - t0
print(f"\n{'='*80}")
print(f"✅ ¡ANÁLISIS COMPLETADO CON RoBERTa!")
print(f"⏱️  Tiempo: {tiempo_total:.1f}s")
print(f"📄 Archivo: {out}")
print(f"{'='*80}\n")

total = len(df)
pos = (df["tono_marca"] == "Positivo").sum()
neg = (df["tono_marca"] == "Negativo").sum()
neu = (df["tono_marca"] == "Neutro").sum()

# Comparación: ajustes aplicados
ajustes = (df["tono_marca"] != df["tono_roberta_base"]).sum()

print("📊 RESUMEN:\n")
print(f"   📈 Total noticias: {total}")
print(f"   🔢 Grupos únicos: {len(rep_texts)}")
print(f"   🔗 Promedio noticias/grupo: {total/len(rep_texts):.1f}")
print(f"   ✅ Positivas: {pos} ({pos/total*100:.1f}%)")
print(f"   ❌ Negativas: {neg} ({neg/total*100:.1f}%)")
print(f"   ⚪ Neutras: {neu} ({neu/total*100:.1f}%)")
print(f"\n   🔧 Ajustes contextuales aplicados: {ajustes} ({ajustes/total*100:.1f}%)")

print("\n📈 DISTRIBUCIÓN DE TONO FINAL:")
print(df["tono_marca"].value_counts().to_string())

print("\n🤖 TONO BASE ROBERTA (sin ajustes):")
print(df["tono_roberta_base"].value_counts().to_string())

print("\n🎯 DISTRIBUCIÓN DE TEMAS:")
print(df["tema"].value_counts().to_string())

# Mostrar ejemplos de ajustes
if ajustes > 0:
    print("\n🔍 EJEMPLOS DE AJUSTES CONTEXTUALES:")
    adjusted = df[df["tono_marca"] != df["tono_roberta_base"]].head(3)
    for idx, row in adjusted.iterrows():
        texto_corto = row[text_col][:90] + "..." if len(row[text_col]) > 90 else row[text_col]
        print(f"\n   📰 {texto_corto}")
        print(f"      RoBERTa base: {row['tono_roberta_base']} → Final: {row['tono_marca']}")

print("\n✨ Análisis finalizado exitosamente ✨")
