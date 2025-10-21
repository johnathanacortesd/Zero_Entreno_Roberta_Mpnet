# 🚀 Analizador de Tono y Tema para Textos en Español (GPU Optimizado)

Este proyecto de Google Colab proporciona un flujo de trabajo completo en dos pasos para analizar grandes volúmenes de texto en español. Combina una interfaz visual para la preparación de datos con un potente script de análisis de NLP optimizado para ejecutarse en GPUs.

## ✨ Características Principales

-   **Interfaz Gráfica Sencilla**: Una herramienta interactiva para cargar un archivo Excel y concatenar múltiples columnas de texto en una sola, preparándola para el análisis.
-   **Análisis de Tono (Sentimiento)**: Clasifica el texto en `Positivo`, `Neutro` o `Negativo` utilizando el modelo `cardiffnlp/twitter-xlm-roberta-base-sentiment-multilingual`.
-   **Clasificación de Temas**: Asigna a cada texto una categoría temática (p. ej., "Políticas Públicas", "Salud", "Economía") basada en similitud semántica.
-   **Alta Velocidad**: El script de análisis está optimizado para GPUs (NVIDIA T4 o superior), procesando miles de textos por segundo mediante técnicas como el procesamiento por lotes (batching), precisión mixta (FP16) y agrupación de textos similares.
-   **Reporte Detallado**: Genera un archivo Excel de salida con los resultados, estadísticas del proceso y gráficos de distribución de tonos y temas.

---

##  workflow Flujo de Trabajo (2 Pasos)

El proceso está diseñado para ser secuencial. Primero preparas los datos y luego los analizas.

### **Paso 1: Preparar los Datos con el Concatenador Visual**

Esta herramienta toma tu archivo Excel con varias columnas de texto (ej: `título`, `descripción`, `contenido`) y las une en una única columna llamada `resumen`.

1.  **Ejecuta la primera celda** titulada `📊 Interfaz para Concatenar Columnas`.
2.  Usa el botón **"Subir Archivo"** para cargar tu archivo `.xlsx`.
3.  Una vez cargado, aparecerá un selector de columnas. **Selecciona dos o más columnas** que desees unir (usa `Ctrl+Click` o `Cmd+Click`).
4.  Haz clic en el botón verde **"🔗 Procesar y Descargar Resultado"**.
5.  Automáticamente se descargará un nuevo archivo llamado `resumen_concatenado.xlsx`. **Este es el archivo que usarás en el siguiente paso.**

### **Paso 2: Analizar Tono y Tema con el Script GPU**

Esta parte toma el archivo generado en el paso anterior y realiza el análisis de NLP.

1.  **Asegúrate de tener un entorno de ejecución con GPU** en Colab (`Entorno de ejecución` > `Cambiar tipo de entorno de ejecución` > `T4 GPU`).
2.  **Ejecuta la segunda celda** (`🔧 Instalaciones`) para instalar las dependencias necesarias.
3.  **Ejecuta la tercera celda** (`🚀 Analizador de Tono y Tema...`).
4.  Aparecerá un botón para subir archivos. **Sube el archivo `resumen_concatenado.xlsx`** que descargaste en el Paso 1.
5.  El script procesará los datos, mostrando el progreso y la velocidad.
6.  Al finalizar, se descargará automáticamente un archivo de resultados detallado (ej: `analisis_fast_resumen_concatenado_...xlsx`).

---

## 🛠️ Requisitos

-   **Google Colab**: El notebook está diseñado para ejecutarse en este entorno.
-   **GPU**: Se requiere un entorno con GPU (como T4) para el script de análisis. El código fallará intencionadamente si no detecta una GPU.
-   **Formato de Archivo**: El archivo de entrada debe ser `.xlsx`.

---

## 🧠 Modelos Utilizados

-   **Análisis de Sentimiento**: [`cardiffnlp/twitter-xlm-roberta-base-sentiment-multilingual`](https://huggingface.co/cardiffnlp/twitter-xlm-roberta-base-sentiment-multilingual) - Un modelo robusto y multilingüe para clasificación de sentimientos.
-   **Clasificación de Temas (Embeddings)**: [`sentence-transformers/paraphrase-multilingual-mpnet-base-v2`](https://huggingface.co/sentence-transformers/paraphrase-multilingual-mpnet-base-v2) - Un modelo de alto rendimiento para generar representaciones vectoriales de texto, usadas para calcular la similitud con los temas predefinidos.
