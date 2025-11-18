# 📰 Analizador de Noticias - Tono y Tema

Sistema de análisis automático de noticias para medir **tono de marca** y **clasificación temática** usando modelos open source con licencia comercial.

[![Abrir en Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1vFgoU5bkl3OLJ7PIjQZlSEznJdfDANnS?usp=sharing#scrollTo=xnY4aGFm1x_V)

## 🎯 Características

✅ **Clustering inteligente** - Agrupa noticias similares (mismo tono/tema garantizado)  
✅ **Análisis de tono** - Positivo/Neutro/Negativo contextual al cliente  
✅ **10 categorías temáticas** - Educación, Infraestructura, Seguridad, etc.  
✅ **100% explicable** - Cada decisión es auditable  
✅ **Sin APIs externas** - Todo local, sin costos recurrentes  

---

## 📦 Dos Versiones Disponibles

### **v30 - Reglas Lingüísticas** (Recomendado para comenzar)
```python
# analizador_v2.0_reglas.py
```

**Características:**
- ⚡ **Rápido**: ~30s para 500 noticias (GPU) / ~60s (CPU)
- 🔍 **Transparente**: Reglas explícitas y ajustables
- 📊 **Precisión**: ~91% en detección de tono
- 💰 **Recursos**: Funciona en CPU modesto

**Usa para:**
- Análisis rápidos y frecuentes
- Cuando necesitas explicar cada decisión
- Recursos computacionales limitados
- Configuración inicial de reglas

---

### **v31 - RoBERTa + Reglas** (Máxima precisión)
```python
# analizador_v2.1_roberta.py
```

**Características:**
- 🤖 **ML + Reglas**: RoBERTa-spanish para sentimiento base
- 🎯 **Precisión**: ~93% en tonos complejos
- 🔧 **Ajustes contextuales**: Refina resultados para marca específica
- 📈 **Mejor con datasets grandes**: 500+ noticias

**Usa para:**
- Análisis profundos mensuales/trimestrales
- Tonos sutiles o ambiguos
- Cuando tienes GPU disponible
- Reportes ejecutivos de alta precisión

---

## 🚀 Instalación

### Colab (Recomendado)
```bash
# Las notebooks instalan automáticamente
!pip install torch transformers sentence-transformers pandas openpyxl tqdm scikit-learn
```

### Local
```bash
pip install torch transformers sentence-transformers pandas openpyxl tqdm scikit-learn
```

**Requisitos mínimos:**
- Python 3.8+
- 8GB RAM (16GB recomendado para v31)
- GPU opcional (acelera 3-5x)

---

## 📖 Uso Rápido

### 1. Prepara tu archivo Excel

```
| titulo | texto | fecha |
|--------|-------|-------|
| Alcalde inaugura... | El alcalde Carlos... | 2024-01-15 |
```

**Columnas aceptadas:** `titulo`, `texto`, `resumen`, `contenido`, `noticia`, `descripcion`

### 2. Ejecuta en Colab

```python
# Sube el notebook correspondiente a Google Colab
# Ejecuta todas las celdas
# Ingresa el nombre del cliente cuando se solicite
# Sube tu archivo Excel
# Descarga el resultado automáticamente
```

### 3. Revisa resultados

El Excel resultante incluye:

| Columna | Descripción |
|---------|-------------|
| `tono_marca` | Positivo/Neutro/Negativo |
| `confianza` | 0.0 - 1.0 (nivel de certeza) |
| `tema` | Categoría temática |
| `grupo_id` | Identificador de cluster |
| `tono_roberta_base` | Solo v31: Sentimiento RoBERTa sin ajustes |

---

## 🎨 Lógica de Análisis

### **Tono de Marca**

#### Positivo (✅)
```
✓ "Alcalde [nombre] confirma/anuncia/inaugura..."
✓ "Bajo el liderazgo del alcalde..."
✓ "500 bachilleres inician vida universitaria"
✓ "Obras culminadas en el sector..."
```

#### Negativo (❌)
```
✗ "Investigan al alcalde por..."
✗ "Denuncian irregularidades del..."
✗ "Escándalo de corrupción..."
```

#### Neutro (⚪)
```
○ "Alcalde lamenta fallecimiento de..."
○ "Alcalde rechaza atentado en..."
○ "Alcalde expresa solidaridad con..."
```

### **Temas**

10 categorías automáticas:
- 🏗️ Infraestructura
- 📚 Educación
- 🚔 Seguridad y Justicia
- 🌍 Relaciones Internacionales
- 💰 Economía
- 🏥 Salud
- 🌱 Medio Ambiente
- ⚖️ Corrupción y Escándalos
- 🏛️ Política y Gobierno
- 📋 Gestión y Acciones

---

## 📊 Comparación de Versiones

| Métrica | v30 Reglas | v31 RoBERTa |
|---------|------------|-------------|
| **Precisión tono** | ~91% | ~93% |
| **Velocidad** | ⚡⚡⚡ | ⚡⚡ |
| **Explicabilidad** | 100% | 85% |
| **Recursos GPU** | No necesaria | Recomendada |
| **Tiempo (500 noticias)** | ~45s | ~90s |
| **Complejidad setup** | Baja | Media |
| **Ajustes contextuales** | Manual | Automático + Manual |

---

## 🔧 Personalización

### Agregar palabras clave (ambas versiones)

```python
# En el código, sección REGLAS

POSITIVE_CONTEXTS = [
    "inaugura",
    "tu_palabra_aqui",  # ← Agregar
]

TOPICS_KEYWORDS = {
    "Deporte": ["estadio", "atleta", ...],  # ← Nueva categoría
}
```

### Ajustar sensibilidad (v31)

```python
# Cambiar umbral de confianza para ajustes
if base_sentiment == "Neutro":
    return "Positivo", 0.87  # ← Ajustar 0.87
```

---

## 🧪 Casos de Prueba

### Entrada:
```
"Alcalde Carlos Pinedo confirma que todo está listo para 
la construcción de escenario multideportivo en El Pando"
```

### Salida esperada:
```yaml
tono_marca: Positivo
confianza: 0.93
tema: Infraestructura
grupo_id: 42
```

### Entrada:
```
"Alcalde rechaza atentado en Cali y expresa solidaridad 
con las víctimas"
```

### Salida esperada:
```yaml
tono_marca: Neutro
confianza: 0.88
tema: Seguridad y Justicia
grupo_id: 158
```

---

## 🤖 Modelos Utilizados

| Modelo | Propósito | Licencia | Tamaño |
|--------|-----------|----------|--------|
| **intfloat/multilingual-e5-large** | Clustering | MIT ✅ | 560M params |
| **finiteautomata/beto-sentiment-analysis** | Sentimiento (v31) | Apache 2.0 ✅ | 110M params |

**Todos los modelos permiten uso comercial sin restricciones.**

---

## 📈 Roadmap

- [ ] Soporte para múltiples idiomas simultáneos
- [ ] API REST para integración
- [ ] Dashboard interactivo con Streamlit
- [ ] Detección de entidades (personas, lugares)
- [ ] Análisis de tendencias temporales
- [ ] Exportación a PowerBI/Tableau

---

## 🐛 Troubleshooting

### Error: "name 'confiances' is not defined"
**Solución:** Usar versión v30 o v31 (ya corregido)

### GPU no detectada
```python
# Verificar
import torch
print(torch.cuda.is_available())  # Debe ser True
```

### Columna de texto no encontrada
**Solución:** Asegúrate de que tu Excel tenga una columna llamada: `titulo`, `texto`, `resumen`, o `contenido`

### Memoria insuficiente
**Solución:** 
- Usa v30 (menos memoria)
- Reduce tamaño de batch
- Procesa en lotes más pequeños

---

## 📝 Licencia

MIT License - Libre uso comercial y académico

---

## 🤝 Contribuciones

¡Contribuciones bienvenidas! 

1. Fork el repositorio
2. Crea una rama: `git checkout -b feature/nueva-funcionalidad`
3. Commit: `git commit -am 'Agrega nueva funcionalidad'`
4. Push: `git push origin feature/nueva-funcionalidad`
5. Abre un Pull Request

---

## 📧 Contacto

**Preguntas o sugerencias:**
- Abre un Issue en GitHub
- Revisa la documentación técnica en `/docs`

---

## 🙏 Agradecimientos

- Microsoft/BAAI por multilingual-e5
- FiniteAutomata por BETO sentiment
- Comunidad Hugging Face

---

**⭐ Si te resulta útil, considera dar una estrella al repo**
