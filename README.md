# Pipeline Base: clasificador en PyTorch

Primer checkpoint del proyecto integrador: infraestructura mínima y ciclo de entrenamiento/validación funcional.

## Estructura

```
data/          datasets (iris.csv)
src/           lógica del modelo (train.py)
results/       curva de pérdida y métricas generadas al entrenar
requirements.txt
README.md
```

## Cómo correrlo

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python src/train.py
```

El script detecta solo el dispositivo (CUDA, MPS o CPU), entrena un MLP con Adam y guarda `results/loss_curve.png` y `results/metrics.json`.

## Detalles técnicos

- **Dataset:** Iris (150 muestras, 4 features, 3 clases), split 80/20 estratificado, estandarización ajustada solo con train.
- **Modelo:** MLP de 2 capas ocultas (16 neuronas, ReLU).
- **Pérdida / optimizador:** CrossEntropyLoss y Adam.
- **Versión de PyTorch:** `COMPLETAR` (la imprime el script en la primera línea y queda en `metrics.json`).
- **Learning rate:** `0.01`. Con Adam y un dataset chico, es un valor que converge rápido sin oscilar.

## Resultados

| Métrica | Valor |
|---|---|
| Pérdida de entrenamiento (final) | `COMPLETAR` |
| Pérdida de validación (final) | `COMPLETAR` |
| Accuracy de validación (final) | `COMPLETAR` |

![Curva de pérdida](results/loss_curve.png)

## Interpretación de la curva de pérdida

`COMPLETAR` tras correr el script: ¿bajó la pérdida de entrenamiento y de validación a lo largo de las épocas? ¿Se mantienen cercanas (sin sobreajuste) o la de validación se despega?
