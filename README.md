# Demosaicking multiespectral agnóstico al patrón

Proyecto del curso Fotografía Computacional 2026-2 (UTFSM). Pregunta: ¿puede una
sola red reconstruir un cubo de 16 bandas a partir de cualquier MSFA, incluso
uno que nunca vio, sin reentrenar?

## Instalación

```bash
uv venv --python 3.12 .venv
uv pip install -p .venv/bin/python -r requirements.txt -e .
.venv/bin/pytest            # 44 tests
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

Todo en un comando, por sesiones (se puede pausar y continuar):

```bash
caffeinate -i scripts/run_all.sh 20000 cave-split 60   # entrena ~60 min y se pausa
caffeinate -i scripts/run_all.sh 20000 cave-split 60   # el mismo comando continúa
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

## Material de referencia (no incluido en este repositorio)

Este proyecto parte de una limitación que reconoce el siguiente trabajo, pero no
incluye ni su artículo ni su código:

- **Artículo**: N. Diaz, A. Alvarado, P. Meza, F. Guzmán y E. Vera, "Multispectral
  Filter Array Design by Optimal Sphere Packing", *IEEE Transactions on Image
  Processing*, vol. 32, pp. 3634–3649, 2023. El PDF tiene copyright de IEEE, por
  lo que no se redistribuye aquí.
- **Código MATLAB del artículo**:
  <https://github.com/nelson10/DemosaickingMultispectral3DSpherePacking>. Su
  licencia lo limita a uso en investigación y prohíbe republicarlo, por lo que
  solo se enlaza.

### Diferencias con ese trabajo

| | Diaz et al. (2023) | Este proyecto |
|---|---|---|
| Objetivo | Diseñar un MSFA óptimo (empaquetamiento de esferas, OSP) | Reconstruir con cualquier MSFA sin reentrenar |
| Red | TRevSCI, reentrenada para cada MSFA | U-Net pequeña (≈ 1,06 M parámetros) que recibe la máscara del patrón como entrada |
| Patrones | Propone OSP y lo compara con otros | Usa IMEC, OSP, BTES y uno aleatorio solo como prueba; no diseña patrones |
| Código | MATLAB | Python / PyTorch, escrito desde cero |
| Datos | Simulados a partir de CAVE (256 × 256, 8 bits); filtros ideales y respuestas espectrales reales de la cámara IMEC | Simulados a partir de CAVE (512 × 512, 16 bandas, filtros ideales, sin ruido) |

Del artículo solo se toma la fórmula del patrón OSP, `(7·i + 4·j) mod 16 + 1` (con `i`, `j` desde 1), para
usarlo como patrón de prueba; no se reutiliza código.

## Resultados

Ver `results/table.md` y `results/hypotheses.json` (se regeneran con el paso 2).
