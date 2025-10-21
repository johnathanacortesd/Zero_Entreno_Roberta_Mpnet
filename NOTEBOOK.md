# Documentación del Notebook: Analizador de Textos

Este documento describe el código y el funcionamiento del notebook de Google Colab, que consta de dos herramientas principales diseñadas para funcionar en secuencia.

## Herramienta 1: Concatenador Visual de Columnas

Esta primera parte del notebook ofrece una interfaz gráfica para que los usuarios puedan preparar fácilmente sus datos para el análisis. El objetivo es tomar un archivo Excel con múltiples columnas de texto y unirlas en una sola.

### Código de la Interfaz

```python
# @title 📊 Interfaz para Concatenar Columnas (Ejecutar esta celda)
# --- Importaciones necesarias ---
import pandas as pd
import ipywidgets as widgets
from google.colab import files
from IPython.display import display, clear_output
import io

# --- Instalación silenciosa de la dependencia para .xlsx ---
!pip install -q openpyxl

# --- Clase principal de la aplicación ---
class VisualConcatenator:
    def __init__(self):
        """Inicializa la interfaz y sus componentes."""
        self.df = None
        self.output_df = None

        # --- Definición de los Widgets (Componentes de la UI) ---
        self.title = widgets.HTML("<h2>🔗 Concatenador Visual de Columnas</h2><hr>")
        self.instructions = widgets.HTML("<h4>Paso 1: Sube tu archivo Excel para empezar.</h4>")

        self.uploader = widgets.FileUpload(
            accept='.xlsx',
            description='Subir Archivo',
            button_style='primary'
        )

        # Contenedor para la segunda parte de la UI (que aparece después de subir)
        self.step2_container = widgets.VBox([])

        # Área de salida para mensajes y resultados
        self.output_area = widgets.Output()

        # --- Observador de eventos ---
        # Llama a la función _handle_upload cuando se sube un archivo
        self.uploader.observe(self._handle_upload, names='value')

    def _handle_upload(self, change):
        """Se activa al subir un archivo. Lee el archivo y crea la UI de selección."""
        # ... (Lógica para leer el archivo y crear la UI del paso 2)

    def _create_column_selector_ui(self):
        """Crea los widgets para seleccionar columnas y procesar."""
        # ... (Creación de widgets de selección y botón de procesado)

    def _process_and_download(self, b):
        """Realiza la concatenación, acortado y descarga del archivo."""
        with self.output_area:
            clear_output()

            selected_cols = self.column_selector.value
            if len(selected_cols) < 2:
                print("❌ Error: Debes seleccionar al menos dos columnas para poder unirlas.")
                return

            print("🚀 Procesando los datos...")
            # --- Lógica principal ---
            # 1. Función para acortar el texto a 80 palabras
            def acortar_texto(texto, limite=80):
                palabras = str(texto).split()
                if len(palabras) > limite:
                    return ' '.join(palabras[:limite]) + '...'
                return ' '.join(palabras)

            # 2. Unir, rellenar vacíos y acortar
            texto_unido = self.df[list(selected_cols)].fillna('').astype(str).apply(lambda fila: ' '.join(fila), axis=1)
            texto_acortado = texto_unido.apply(acortar_texto)

            # 3. Crear DataFrame final y descargar
            self.output_df = pd.DataFrame({'resumen': texto_acortado})
            output_filename = 'resumen_concatenado.xlsx'
            self.output_df.to_excel(output_filename, index=False, engine='openpyxl')
            files.download(output_filename)

    def display_app(self):
        """Muestra la aplicación completa en la celda."""
        display(self.title, self.instructions, self.uploader, self.step2_container, self.output_area)

# --- Punto de entrada: Crear y mostrar la aplicación ---
app = VisualConcatenator()
app.display_app()
