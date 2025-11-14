# 🚀 Analizador de Tono y Tema para Noticias (v15)

Un pipeline avanzado de NLP para clasificar el tono (sentimiento) y el tema de grandes volúmenes de texto, optimizado para el contexto de noticias en español.

[![Abrir en Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1vFgoU5bkl3OLJ7PIjQZlSEznJdfDANnS?usp=sharing#scrollTo=f9f36a86)

Este script toma un archivo Excel, procesa una columna de texto (ej. resumen), y devuelve un reporte detallado con:

- **Tono:** Positivo, Negativo o Neutro.

- **Tema:** Una categoría fija (ej. "Política y Gobierno", "Economía y Finanzas").

- **Agrupamiento:** Identifica noticias duplicadas o semánticamente idénticas para optimizar el análisis y la visualización.

## 🔬 Metodología: El Pipeline de Análisis

El script opera en un pipeline de varias fases, diseñado para maximizar tanto la velocidad como la precisión contextual.

1.  **Carga y Limpieza:**

    - Lee un archivo .xlsx subido por el usuario.

    - Normaliza y limpia los textos de la columna resumen (clean\_text\_enhanced), preparándolos para el análisis.

2.  **Fase 1: Agrupamiento Inteligente (Clustering)**

    - **Propósito:** Evitar analizar miles de noticias idénticas o muy similares. Si 500 artículos son un re-post de la misma noticia, solo se analiza *una vez*.

    - **Nivel 1 (Firma):** Agrupa textos por sus primeras 8 palabras. Es un filtro rápido para encontrar duplicados exactos.

    - **Nivel 2 (Semántico):** Dentro de cada grupo de "firma", un modelo de embeddings (paraphrase-multilingual-mpnet-base-v2) vectoriza los textos. Un clustering jerárquico (AgglomerativeClustering o DBSCAN) agrupa textos que *significan* lo mismo aunque estén escritos de forma diferente.

    - **Nivel 3 (Fusión):** Compara los textos "representantes" de cada cluster y fusiona micro-clusters que son semánticamente idénticos.

3.  **Fase 2: Análisis de Tono (Híbrido)**

    - Se procesan únicamente los textos "representantes" de cada cluster.

    - **Lógica Base:** El modelo clapAI/roberta-large-multilingual-sentiment da una clasificación inicial (Positivo, Negativo, Neutro).

    - **Lógica de Patrones:** El texto también se compara con un diccionario de patrones (SENTIMENT\_PATTERNS) que contienen palabras con alta carga sentimental (ej. tragedia, crisis grave, logro histórico).

    - **Decisión Final:** Se implementa una lógica de votación. Para noticias individuales, se da **prioridad al resultado de los patrones** (si no es neutro), ya que estos son más sensibles al contexto de noticias que el modelo general.

4.  **Fase 3: Clasificación de Tema (Jerárquica)**

    - **Nivel 1 (Keywords):** El texto se compara primero con el diccionario TOPIC\_KEYWORDS. Si encuentra coincidencias fuertes (ej. "presidente petro" o "reforma pensional"), asigna el tema "Política y Gobierno" con alta confianza. Esto es rápido y muy preciso.

    - **Nivel 2 (Semántico):** Si no hay un match claro por keywords, se usan los embeddings del modelo paraphrase-multilingual-mpnet-base-v2. Se compara la similitud semántica del texto con "centroides" (frases clave) de cada tema y se asigna el más cercano.

5.  **Fase 4: Post-procesamiento y Consistencia**

    - El script revisa los resultados en busca de inconsistencias lógicas.

    - **Ejemplo:** Si un artículo fue clasificado como "Positivo" pero el tema es "Seguridad y Justicia" (y contiene palabras como masacre o asesinato), el script lo corrige automáticamente a "Neutro" o "Negativo".

6.  **Fase 5: Exportación**

    - Los resultados (tono y tema) del cluster representante se propagan a todos los textos que pertenecen a ese cluster.

    - Se genera un archivo Excel (analisis\_avanzado\_v15\_...xlsx) con los resultados detallados, estadísticas y métricas de rendimiento.

## 🧠 Modelos Utilizados y Ventajas

La elección de los modelos es crucial para el contexto de noticias en español.

### Para Tono: clapAI/roberta-large-multilingual-sentiment

- **¿Qué es?** Es un modelo RoBERTa-large entrenado en 8 idiomas (incluyendo español) y afinado (fine-tuned) específicamente para la tarea de clasificación de sentimientos.

- **Ventajas para Noticias en Español:**

    1.  **Multilingüe Nativo:** A diferencia de modelos solo en inglés traducidos, este fue entrenado con texto en español desde el inicio, capturando mejor las sutilezas, modismos y estructuras gramaticales del idioma.

    2.  **Arquitectura "Large":** Al ser un modelo large, tiene una comprensión contextual más profunda que los modelos base, permitiéndole entender mejor la ironía o el sentimiento complejo en frases largas de noticias.

    3.  **Híbrido con Patrones:** El script no confía ciegamente en el modelo. Al combinarlo con los SENTIMENT\_PATTERNS (que fueron ajustados para ser más sensibles), se obtiene lo mejor de ambos mundos: la comprensión contextual del modelo y la precisión explícita de las palabras clave.

### Para Temas (Clustering y Semántica): sentence-transformers/paraphrase-multilingual-mpnet-base-v2

- **¿Qué es?** Es un modelo MPNet (una evolución de BERT) parte de la familia SentenceTransformers. Está diseñado para una tarea específica: crear "embeddings" (vectores numéricos) que representan el significado de una oración.

- **Ventajas para Noticias en Español:**

    1.  **Especialista en Paráfrasis:** Este modelo fue entrenado específicamente para identificar **paráfrasis** (frases que significan lo mismo). Esto es *perfecto* para las noticias, donde "El gobierno anuncia nueva reforma" y "Ejecutivo presenta proyecto de ley" son semánticamente idénticos. El modelo los agrupará correctamente.

    2.  **Eficiencia Semántica:** Genera vectores de alta calidad que son excelentes para comparar similitud (clustering). Esto permite al script encontrar textos temáticamente similares incluso si no comparten *ninguna* palabra clave.

    3.  **Multilingüe:** Al igual que RoBERTa, su naturaleza multilingüe garantiza un alto rendimiento en español.

## 🎯 La Necesidad de Temas Fijos y Palabras Clave

Este script utiliza un enfoque de **clasificación supervisada (temas fijos)** en lugar de un enfoque no supervisado (como *Topic Modeling* tipo LDA, que "descubre" temas).

**¿Por qué es esto una ventaja?**

1.  **Consistencia del Negocio:** Para el análisis de medios, la consistencia es clave. Los analistas necesitan "contenedores" estables. Si un modelo "descubre" el tema "Fútbol" un día y "Deportes de Balón" al siguiente, los reportes no son comparables. Los temas fijos (Política, Economía, Deportes) garantizan que los datos siempre estén organizados de la misma manera.

2.  **Contextualización (Dominio Específico):** Un modelo de IA general no sabe que "ELN", "disidencias FARC" o "paz total" son temas de "Seguridad y Justicia" en Colombia. Tampoco sabe que "presidente petro" o "reforma pensional" son "Política y Gobierno".

    - El diccionario TOPIC\_KEYWORDS **inyecta este conocimiento de dominio específico** (el contexto colombiano) directamente en el pipeline.

3.  **Precisión y Velocidad (Enfoque Jerárquico):** El sistema de palabras clave (primary y secondary) es extremadamente rápido. El script lo usa como un "filtro de alta confianza".

    - Si un texto contiene "inflación" y "banco república", se clasifica instantáneamente como "Economía".

    - Esto permite que el modelo de embeddings (que es más lento) solo se utilice en los textos ambiguos que no activaron ninguna palabra clave, optimizando el rendimiento general.

## 🚀 Cómo Usar

1.  **Entorno:** Asegúrese de estar en un entorno con GPU (como Google Colab) para un rendimiento óptimo.

2.  **Instalación:** Instale las dependencias necesarias:

    ```bash

    pip install torch pandas numpy tqdm transformers sentence-transformers scikit-learn openpyxl

    ```

3.  **Ejecución:** Corra el script `analizador_v15.py` en su entorno.

4.  **Carga:** Cuando se le solicite, suba su archivo Excel. Este archivo **debe** contener una columna llamada `resumen`.

5.  **Descarga:** El script procesará todos los textos y automáticamente iniciará la descarga del archivo de resultados (ej. `analisis_avanzado_v15_...xlsx`).

## ⚖️ Licencias

### Licencia del Script

Este script (`analizador_v15.py`) se distribuye bajo la **Licencia MIT**.

Copyright (c) 2025 [Johnathan Cortés]

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

### Licencias de los Modelos Utilizados

Los modelos de Hugging Face utilizados en este proyecto se distribuyen bajo la **Licencia Apache 2.0**:

- clapAI/roberta-large-multilingual-sentiment: [Apache 2.0 License](https://huggingface.co/clapAI/roberta-large-multilingual-sentiment/blob/main/LICENSE)

- sentence-transformers/paraphrase-multilingual-mpnet-base-v2: [Apache 2.0 License](https://huggingface.co/sentence-transformers/paraphrase-multilingual-mpnet-base-v2/blob/master/LICENSE)
