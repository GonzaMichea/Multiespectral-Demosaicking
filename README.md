# Demosaicking multiespectral agnóstico al patrón

Proyecto del curso Fotografía Computacional 2026-2 (UTFSM). Pregunta: ¿puede una
sola red reconstruir un cubo de 16 bandas a partir de cualquier MSFA, incluso
uno que nunca vio, sin reentrenar?

## Instalación

```bash
uv venv --python 3.12 .venv
uv pip install -p .venv/bin/python -r requirements.txt -e .
.venv/bin/pytest            # 42 tests
```

## Datos

```bash
.venv/bin/python scripts/download_cave.py --data-root data   # CAVE, ~800 MB
```

- `--data-mode cave-split`: solo CAVE, partición fija 21 / 3 / 8 (entrenar / validar / probar).
- `--data-mode arad`: entrena con ARAD-1K (`data/ARAD_1K/Train_Spec/*.mat`, 220 + 30)
  y prueba con las 32 escenas de CAVE.

## Uso

```bash
# 1. Entrenar (≈ 2 h por red en un Mac M5 con MPS, 40 000 iteraciones)
.venv/bin/python -m msdemosaic.train --pattern imec   --data-mode cave-split --out checkpoints/fixed_imec.pt
.venv/bin/python -m msdemosaic.train --pattern osp    --data-mode cave-split --out checkpoints/fixed_osp.pt
.venv/bin/python -m msdemosaic.train --pattern random --data-mode cave-split --out checkpoints/agnostic.pt

# 2. Evaluar métodos × patrones × escenas → results/{per_scene,per_band,summary}.csv, table.md, table.tex, hypotheses.json
.venv/bin/python -m msdemosaic.evaluate_all --data-mode cave-split

# 3. Figuras → results/figs/
.venv/bin/python -m msdemosaic.figures --data-mode cave-split

# 4. Póster (lee results/)
typst compile --root . Poster/poster.typ
```

Para entrenar en Kaggle: `notebooks/kaggle_train.ipynb`.

## Estructura

| Archivo | Contenido |
|---|---|
| `msdemosaic/patterns.py` | Patrones de prueba (IMEC, OSP, BTES, aleatorio) y muestreador de patrones de entrenamiento |
| `msdemosaic/simulate.py` | `simulate(cube, pattern) -> (mosaic, mask)` |
| `msdemosaic/wb.py` | Baseline weighted bilinear, `wb(mosaic, mask) -> recon` |
| `msdemosaic/model.py` | U-Net (≈ 1,06 M parámetros) y `load_method(ckpt)` |
| `msdemosaic/train.py` | Entrenamiento de redes fijas y agnóstica |
| `msdemosaic/metrics.py` | PSNR, SSIM, SAM, `evaluate(recon, cube)` |
| `msdemosaic/evaluate_all.py` | Tabla completa e hipótesis H1–H3 |
| `msdemosaic/figures.py` | Figuras de resultados |

## Resultados

Ver `results/table.md` y `results/hypotheses.json` (se regeneran con el paso 2).
