# Notebooks

## analizador_v30_reglas.ipynb
Análisis con reglas lingüísticas (rápido, explicable)

## analizador_v31_roberta.ipynb
Análisis híbrido RoBERTa + reglas (máxima precisión)

### Uso en Colab:
1. Abrir notebook en Google Colab
2. Ejecutar todas las celdas
3. Subir archivo Excel cuando se solicite
```

---

#### 2.3 Crear carpeta `src/`
```
src/
└── news_analyzer/
    ├── __init__.py
    ├── clustering.py
    ├── sentiment_rules.py
    ├── sentiment_roberta.py
    ├── topic_classifier.py
    └── utils.py
