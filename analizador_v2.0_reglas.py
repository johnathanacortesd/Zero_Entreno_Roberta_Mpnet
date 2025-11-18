!pip install -q torch transformers sentence-transformers pandas openpyxl tqdm scikit-learn 2>/dev/null

import os, re, time, torch
import numpy as np
import pandas as pd
import warnings
from tqdm.auto import tqdm
from sentence_transformers import SentenceTransformer
from collections import defaultdict
from google.colab import files

warnings.filterwarnings("ignore")

# ==================== CONFIGURACIÓN ====================
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🚀 Dispositivo: {device.upper()}")

# Modelo multilingüe con licencia Apache 2.0
embedder = SentenceTransformer("intfloat/multilingual-e5-large", device=device)

# ==================== REGLAS MEJORADAS ====================
POSITIVE_VERBS_CLIENT = [
    "confirma", "anuncia", "inaugura", "lanza", "lidera", "impulsa", "acompaña","continua",
    "gestiona", "acompaña", "otorga", "destina", "facilita", "apoya", "confirma","recibe",
    "reconoce", "homenajea", "firma", "inicia", "invierte", "entrega", "adelanta","demuestra",
    "presenta", "logra", "avanza", "atiende", "visita", "verifica","continúa","intensifica",
    "construye", "desarrolla", "fortalece", "promueve", "garantiza","expresa",
]

# Contextos que indican gestión POSITIVA del cliente
POSITIVE_CONTEXTS = [
    "todo está listo", "obras de", "inversión en","recibe distinción",
    "mejoras en", "avance en", "beneficio", "articulación","expresa solidaridad",
    "liderazgo del", "bajo el liderazgo", "gracias a la alcaldía",
    "gracias al alcalde", "proyecto es posible gracias",
    "gestión de", "gestión del alcalde"
]

# Logros de programas/proyectos que implican gestión positiva
ACHIEVEMENT_INDICATORS = [
    "bachilleres inician", "bachilleres se han inscrito", "estudiantes inician",
    "familias beneficiadas", "viviendas entregadas", "obras culminadas",
    "proyectos en marcha", "inversión de", "beneficiarios",
    "acceso a", "programas profesionales", "vida universitaria"
]

# Palabras que indican acciones REACTIVAS (neutras, no gestión)
NEUTRAL_REACTIONS = [
    "lamenta", "rechaza", "condena", "expresa solidaridad",
    "pide", "solicita", "exige", "manifesta preocupación"
]

# Contextos claramente negativos
NEGATIVE_TRIGGERS = [
    "escándalo", "corrupción", "investigan al", "denuncian al",
    "irregularidad", "desfalco", "acusan", "protesta contra",
    "demanda contra", "sancionan", "fraude", "malversación"
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

# ==================== FUNCIONES CORREGIDAS ====================

def normalize_text(text):
    """Normaliza texto removiendo artículos y preposiciones para mejor clustering"""
    # Convertir a minúsculas
    t = text.lower()

    # Remover artículos y preposiciones comunes
    stopwords = ['el', 'la', 'los', 'las', 'un', 'una', 'unos', 'unas',
                 'de', 'del', 'en', 'para', 'por', 'con', 'sin', 'sobre',
                 'que', 'todo', 'está', 'son', 'es']

    # Mantener solo caracteres alfanuméricos y espacios
    t = re.sub(r'[^\wáéíóúñ\s]', ' ', t)
    t = re.sub(r'\s+', ' ', t).strip()

    # Remover stopwords
    words = t.split()
    words = [w for w in words if w not in stopwords and len(w) > 2]

    return " ".join(words)

def get_signature(text, words=20):
    """Genera firma textual normalizada para clustering"""
    normalized = normalize_text(text)
    # Tomar más palabras para capturar mejor la similitud
    return " ".join(normalized.split()[:words])

def universal_cluster(texts):
    """Clustering REAL - noticias similares tendrán mismo tema/tono"""
    print("📊 Agrupando noticias similares...")
    groups = defaultdict(list)

    for i, t in enumerate(texts):
        signature = get_signature(t)
        groups[signature].append(i)

    representatives = []
    cluster_map = [-1] * len(texts)
    cid = 0

    for idxs in groups.values():
        # El representante es el texto más largo del grupo (más completo)
        longest_idx = max(idxs, key=lambda i: len(texts[i]))
        representatives.append(texts[longest_idx])

        for i in idxs:
            cluster_map[i] = cid
        cid += 1

    print(f"✅ {len(texts)} noticias → {len(representatives)} grupos únicos")
    return representatives, np.array(cluster_map)

def analyze_universal(rep_text, client_name):
    """Análisis MEJORADO con lógica de contexto"""
    lower = rep_text.lower()
    client_lower = client_name.lower()

    # === ANÁLISIS DE TONO CON PRIORIDADES ===

    # PRIORIDAD 1: Negativos directos contra el cliente
    negative_about_client = any(
        f"{neg} {client_lower}" in lower or f"{client_lower} {neg}" in lower
        for neg in ["investigan al", "denuncian al", "acusan", "escándalo", "corrupción"]
    )

    if negative_about_client:
        tono = "Negativo"
        confianza = 0.95

    # PRIORIDAD 2: Reacciones neutras (lamenta, rechaza eventos externos)
    elif any(reaction in lower for reaction in NEUTRAL_REACTIONS):
        # Si solo reacciona a eventos externos = Neutro
        # (no es gestión propia, solo declaraciones)
        tono = "Neutro"
        confianza = 0.88

    # PRIORIDAD 3: Gestión positiva del cliente
    else:
        # Buscar si el cliente ES EL GESTOR (sujeto activo)
        is_active_subject = False

        # Contexto 1: "liderazgo del alcalde X", "gracias a X"
        leadership_patterns = [
            f"liderazgo del {client_lower}",
            f"liderazgo de {client_lower}",
            f"gracias a {client_lower}",
            f"gracias al {client_lower}",
            f"gestión del {client_lower}",
            f"gestión de {client_lower}"
        ]

        if any(pattern in lower for pattern in leadership_patterns):
            is_active_subject = True

        # Contexto 2: "alcalde X + verbo de gestión activa"
        if not is_active_subject:
            active_verbs = ["confirma", "anuncia", "inaugura", "entrega", "invierte",
                          "construye", "lidera", "impulsa", "gestiona", "inicia"]

            for verb in active_verbs:
                # Patrón: "alcalde X verbo"
                if f"alcalde {client_lower} {verb}" in lower or \
                   f"alcaldía {verb}" in lower:
                    is_active_subject = True
                    break

        # Contexto 3: Logros/resultados de programas (implican gestión)
        # "500 bachilleres inician...", "familias beneficiadas..."
        has_achievement = any(indicator in lower for indicator in ACHIEVEMENT_INDICATORS)

        # Contexto 4: Presencia de indicadores positivos generales
        has_positive_context = any(ctx in lower for ctx in POSITIVE_CONTEXTS)

        # Si hay logros concretos en áreas de gestión pública = Positivo
        if is_active_subject or has_positive_context or has_achievement:
            tono = "Positivo"
            confianza = 0.92 if is_active_subject else 0.85
        else:
            tono = "Neutro"
            confianza = 0.70

    # === ANÁLISIS DE TEMA ===
    tema = "Gestión y Acciones"  # Por defecto
    max_matches = 0

    for topic, keywords in TOPICS_KEYWORDS.items():
        matches = sum(1 for kw in keywords if kw in lower)
        if matches > max_matches:
            max_matches = matches
            tema = topic

    return tono, round(confianza, 3), tema

# ==================== MAIN EXECUTION ====================
print("\n" + "="*80)
print("📰 ANALIZADOR DE NOTICIAS v30 - CLUSTERING REAL")
print("="*80 + "\n")

# Solicitar nombre del cliente
client_name = input("👤 Nombre del cliente (ej: Carlos Pinedo): ").strip()
if not client_name:
    client_name = "Carlos Pinedo"
    print(f"   Usando cliente por defecto: {client_name}")

# Cargar archivo
print("\n📂 Sube tu archivo Excel...")
uploaded = files.upload()
file_name = list(uploaded.keys())[0]

# Leer datos
print(f"📖 Leyendo {file_name}...")
df = pd.read_excel(file_name)

# Detectar columna de texto
text_columns = ['resumen', 'texto', 'contenido', 'titulo', 'title', 'noticia', 'descripcion']
text_col = next((c for c in text_columns if c in df.columns), None)

if not text_col:
    print(f"❌ Error: No se encontró columna de texto. Columnas disponibles: {list(df.columns)}")
    raise ValueError("No se encontró columna de texto válida")

print(f"✅ Columna detectada: '{text_col}'")

# Limpiar datos
df = df.dropna(subset=[text_col]).reset_index(drop=True)
texts = df[text_col].astype(str).tolist()
print(f"📊 {len(texts)} noticias cargadas\n")

# ==================== PROCESAMIENTO ====================
t0 = time.time()

# Clustering REAL
rep_texts, cluster_map = universal_cluster(texts)

# Análisis - CADA GRUPO SE ANALIZA UNA SOLA VEZ
print("\n🔍 Analizando tono y tema de grupos únicos...")
tonos = []
confianzas = []
temas = []

for rep in tqdm(rep_texts, desc="Procesando"):
    tono, conf, tema = analyze_universal(rep, client_name)
    tonos.append(tono)
    confianzas.append(conf)
    temas.append(tema)

# Mapear resultados - TODAS las noticias del mismo grupo tendrán EXACTAMENTE el mismo tema/tono
print("\n🔗 Aplicando tema/tono a grupos similares...")
df["grupo_id"] = cluster_map
df["tono_marca"] = [tonos[cluster_map[i]] for i in range(len(df))]
df["confianza"] = [confianzas[cluster_map[i]] for i in range(len(df))]
df["tema"] = [temas[cluster_map[i]] for i in range(len(df))]

# Verificación: Mostrar algunos grupos para debugging
print("\n🔍 Verificando grupos (muestra):")
for gid in sorted(df["grupo_id"].unique())[:3]:
    grupo = df[df["grupo_id"] == gid]
    if len(grupo) > 1:
        print(f"\n   Grupo {gid} ({len(grupo)} noticias) - Tono: {tonos[gid]} - Tema: {temas[gid]}")
        for idx, row in grupo.head(2).iterrows():
            texto_corto = row[text_col][:80] + "..." if len(row[text_col]) > 80 else row[text_col]
            print(f"      • {texto_corto}")

# ==================== EXPORTACIÓN (SOLO 1 HOJA) ====================
out = f"Final_{client_name.replace(' ', '_')}_{time.strftime('%Y%m%d_%H%M')}.xlsx"
print(f"\n💾 Guardando resultados en {out}...")

# SOLO UNA HOJA con todos los resultados
df.to_excel(out, sheet_name="Resultados", index=False, engine="openpyxl")

files.download(out)

# ==================== RESULTADOS ====================
tiempo_total = time.time() - t0
print(f"\n{'='*80}")
print(f"✅ ¡ANÁLISIS COMPLETADO!")
print(f"⏱️  Tiempo: {tiempo_total:.1f}s")
print(f"📄 Archivo: {out}")
print(f"{'='*80}\n")

# Estadísticas
total = len(df)
pos = (df["tono_marca"] == "Positivo").sum()
neg = (df["tono_marca"] == "Negativo").sum()
neu = (df["tono_marca"] == "Neutro").sum()

print("📊 RESUMEN:\n")
print(f"   📈 Total noticias: {total}")
print(f"   🔢 Grupos únicos: {len(rep_texts)}")
print(f"   🔗 Promedio noticias/grupo: {total/len(rep_texts):.1f}")
print(f"   ✅ Positivas: {pos} ({pos/total*100:.1f}%)")
print(f"   ❌ Negativas: {neg} ({neg/total*100:.1f}%)")
print(f"   ⚪ Neutras: {neu} ({neu/total*100:.1f}%)")

# Mostrar algunos ejemplos de agrupación
print("\n📋 EJEMPLOS DE AGRUPACIÓN:")
grupos_multi = df[df.duplicated(subset=["grupo_id"], keep=False)].sort_values("grupo_id")
if len(grupos_multi) > 0:
    for gid in grupos_multi["grupo_id"].unique()[:2]:
        grupo = df[df["grupo_id"] == gid]
        print(f"\n   🔗 Grupo {gid} → {len(grupo)} noticias similares")
        print(f"      Tono asignado: {grupo.iloc[0]['tono_marca']}")
        print(f"      Tema asignado: {grupo.iloc[0]['tema']}")
        for idx, row in grupo.head(2).iterrows():
            texto_corto = row[text_col][:70] + "..." if len(row[text_col]) > 70 else row[text_col]
            print(f"      • {texto_corto}")
else:
    print("   ℹ️  No se encontraron noticias duplicadas en esta muestra")

print("\n📈 DISTRIBUCIÓN DE TONO:")
print(df["tono_marca"].value_counts().to_string())

print("\n🎯 DISTRIBUCIÓN DE TEMAS:")
print(df["tema"].value_counts().to_string())

print("\n✨ Análisis finalizado exitosamente ✨")
