# Demosaicking multiespectral agnóstico al patrón — Plan de implementación

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Construir en 13 días (12–25 nov 2026) el pipeline Python que simula 4 MSFA de 16 bandas, reconstruye con WB y 3 U-Nets (fija-IMEC, fija-OSP, agnóstica) y produce la tabla H1–H3, las figuras, el póster (18 nov) y el informe ICIP (25 nov).

**Architecture:** Paquete `msdemosaic` con tres piezas desacopladas por la interfaz común `simulate(cube, pattern) -> (mosaic, mask)`, `method(mosaic, mask) -> recon`, `evaluate(recon, cube) -> {psnr, ssim, sam}`. Todo método (WB o red) se expone como un callable NumPy con esa firma; un único script de evaluación recorre métodos × patrones × escenas y escribe CSV, del cual salen tabla y figuras.

**Tech Stack:** Python 3.11+, NumPy, SciPy, scikit-image, imageio, h5py, PyTorch (MPS en Mac, CUDA en Kaggle), matplotlib, pytest. Póster en Typst, informe en LaTeX (tectonic).

**Spec:** `CLAUDE.md` (decisión, hipótesis, método, alcance) y `Presentacion_Plan/plan_proyecto.typ` (cronograma, hitos, riesgos, kernels de los patrones en líneas 51–55). El ejecutor lee ambos.

## Actualización del 16 nov 2026 (manda sobre el resto del documento)

Tras la revisión (`docs/revision-2026-11-16.md`) cambió el diseño de las redes.
Donde este plan diga otra cosa, vale lo siguiente:

- **Entrada de todas las redes:** `[WB(Y, M), Y⊙M, M]` (48 canales), salida
  directa, pérdida L1. Reemplaza a `[Y, M]` (agnóstica) y a "solo Y" (fijas).
  Motivo: con `[Y, M]` la agnóstica quedaba 4 dB bajo WB a las 2 000 iteraciones;
  con WB como entrada queda 2 dB sobre WB.
- **Redes fijas con la misma entrada que la agnóstica** (incluida la máscara):
  la única diferencia es el patrón de entrenamiento. La red sin máscara de la
  ppt pasa a ser una ablación opcional (`--input mosaic`). Motivo: la fija-OSP
  sin máscara perdía 1,45 dB porque OSP tiene período 16 en filas y la U-Net
  solo distingue posiciones módulo 4.
- **Interfaz:** `UNet(input_mode="wb" | "mosaic_mask" | "mosaic", base=48)`;
  `train(..., input_mode="wb")`; CLI `--input`; el checkpoint guarda
  `"input_mode"` (ya no `"use_mask"`). `weighted_bilinear(sparse, mask)` en
  `model.py` es la versión torch de `wb`.
- **Parámetros:** 1 068 640 (sigue bajo 2 M).
- **Evaluación:** desviación estándar muestral; `paired_differences(rows)`
  entrega las diferencias por escena de H2 y H3 en `hypotheses.json["paired"]`.
- **No usar residuo sobre WB**: con L1 o L1 suavizada las ReLU mueren.
- **Expectativa realista:** H1 será una caída moderada (~3 dB, la fija vuelve
  al nivel de WB), no un colapso; H2 está en 1,5 dB de brecha a las 2 000
  iteraciones y puede no bajar de 1 dB; H3 se cumple con +1,4 a +2,5 dB.

### Tareas nuevas

- [x] **T15: entrada `wb` y modos de entrada** en `model.py`/`train.py` (hecho, 66 tests).
- [x] **T16: desviación muestral y diferencias pareadas** en `evaluate_all.py` (hecho).
- [ ] **T17: probar `scripts/run_all.sh` de punta a punta** con pocas iteraciones
  (p. ej. `scripts/run_all.sh 30 cave-split`), revisar que genera los 3
  checkpoints, `results/` y el póster; después borrar `checkpoints/` y
  restaurar `results/` y `Poster/poster.pdf` con `git checkout`.
- [ ] **T18: póster** — describir la entrada nueva, cambiar la línea base (ya no
  "sin máscara"), parámetros 1,07 M, iteraciones reales.
- [ ] **T19 (opcional): red convencional sin máscara** como cuarta fila
  (`conventional_imec`): agregar a `METHOD_ORDER`, `hypotheses`
  (`h1_drop_db_conventional`), figuras y póster; entrenar al final.
- [ ] **T12 (entrenamiento real):** `caffeinate -i scripts/run_all.sh 20000 cave-split`
  (~2,5 h) o 40 000 (~5 h, de noche). Orden: fija-IMEC → agnóstica → fija-OSP.

### Calendario vigente

| Cuándo | Qué |
|---|---|
| Lun 16 nov (noche) o mar 17 (mañana) | T17, T18 y lanzar T12 |
| Mar 17 nov | Evaluación, figuras, póster final; práctica de la presentación |
| **Mié 18 nov, 14:40** | **Póster** |
| Jue 19 – dom 22 nov | Reentrenar a 40 000 si hace falta, T19, nivel C si sobra tiempo; experimentos cerrados el dom 22 |
| Lun 23 – mar 24 nov | Informe ICIP y README |
| **Mié 25 nov** | **Entrega final** |

## Calendario original (reemplazado por el de arriba)

| Fecha | Fase ppt | Tareas | Hito |
|---|---|---|---|
| jue 12 – vie 13 nov | S1 | T1, T2, T3, T4 | Humanos: descargar CAVE y pedir ARAD-1K **hoy** |
| sáb 14 nov | S2 | T5, T6 | **Pipeline funcionando** (fila WB de la tabla). Corte ARAD-1K |
| dom 15 nov | S2 | T7, T8 | Lanzar entrenamiento fija-IMEC |
| lun 16 nov | S3 | T9 (parcial) | **Punto de decisión**: ¿fija-IMEC supera a WB en IMEC y H1 está medida? Sí → nivel B. No → cerrar en nivel A |
| mar 17 nov | — | T10, T11 | Póster compilado |
| **mié 18 nov** | — | — | **Presentación de avance (póster)**: nivel A + agnóstica preliminar si existe |
| jue 19 – vie 20 nov | S3–S4 | T12 | Entrenar fija-OSP y agnóstica final |
| sáb 21 – dom 22 nov | S5 | T9 (completo), T10 | **Experimentos cerrados** (dom 22): no se agregan experimentos |
| lun 23 – mar 24 nov | S6 | T13 | Informe, README |
| **mié 25 nov** | — | — | **Entrega final** |

Roles: el LLM escribe código y tests. Persona A: descargas, registro NTIRE, revisar T2–T6, correr evaluación. Persona B: revisar T7–T8, correr entrenamientos en Kaggle (T12). Nivel C (T14) solo si la tabla está cerrada el vie 20.

## Global Constraints

- 16 bandas, filtros ideales, sin ruido (nivel B): `Y = Σ_k X_k ⊙ C_k`.
- Cubos: `np.ndarray` `float32`, forma `(H, W, 16)`, rango `[0, 1]`. Patrón: `np.ndarray` `int64`, forma `(H, W)`, valores `0..15` (**0-indexado**; los kernels del `.typ` están 1-indexados). Máscara: `float32` one-hot `(H, W, 16)`. Tensores PyTorch: `(B, C, H, W)`.
- Bandas: `BAND_IDX = [0, 2, 4, ..., 30]` de las 31 (400–700 nm, paso 20 nm).
- U-Net < 2 000 000 parámetros, pérdida L1, parches 64 × 64.
- La red agnóstica nunca se entrena con IMEC, OSP, BTES ni el kernel "aleatorio" de prueba.
- Las redes fijas no reciben máscara (entrada 1 canal); la agnóstica recibe `[Y, M]` (17 canales).
- Prueba: 32 escenas CAVE (modo `arad`) u 8 escenas (modo `cave-split`, plan B de la ppt).
- Datos, checkpoints y cachés fuera de git (`data/`, `checkpoints/`).
- Identificadores y comentarios de código en inglés; documentos en español; informe en inglés (ICIP).
- No tocar `LaTeX_template/Template.tex`, `proposal.tex` ni `Presentacion_Plan/`.
- Fuera de alcance: cámara real, TRevSCI, diseño de patrones, > 16 bandas, PPID (nivel C).

## Review Focus

1. **Tamaño no múltiplo de 4** (ARAD-1K es 482 × 512): la U-Net debe aceptar cualquier `H, W` y devolver la misma forma → test en T7.
2. **Píxeles negros** (fondos de CAVE con espectro ≈ 0): SAM no debe dar NaN → test en T3.
3. **PNG irregulares de CAVE** (canales extra, 8 bits, carpeta anidada `x_ms/x_ms/`, archivos `.bmp` y `Thumbs.db`): el cargador debe entregar siempre `(512, 512, 31)` en `[0, 1]` → test en T4.
4. **Fase del patrón**: el patrón se genera desde el origen `(0, 0)` del parche y de la imagen completa; un desfase entre entrenamiento y prueba arruina las redes fijas → test en T2 (periodicidad) y T8 (el parche usa `make_pattern(name, 64, 64)`).
5. **Dispositivo**: checkpoint entrenado en CUDA debe cargar en MPS/CPU (`map_location`), y la salida de la red puede salir de `[0, 1]` → `clip` antes de métricas → tests en T7 y T3.

## Estructura de archivos

```
pyproject.toml, requirements.txt, .gitignore, README.md
msdemosaic/
  __init__.py
  patterns.py      # 4 patrones de prueba + muestreador de entrenamiento
  simulate.py      # simulate(cube, pattern)
  metrics.py       # psnr, ssim, sam, evaluate, psnr_per_band
  data.py          # CAVE, ARAD-1K, selección de bandas, splits, caché, parches
  wb.py            # weighted bilinear (convolución normalizada)
  model.py         # UNet + load_method
  train.py         # CLI de entrenamiento
  evaluate_all.py  # CLI: métodos × patrones × escenas → CSV + tabla
  figures.py       # CLI: figuras desde los CSV
tests/test_{patterns,simulate,metrics,data,wb,model,train,evaluate_all}.py
scripts/download_cave.py
notebooks/kaggle_train.ipynb
results/            # CSV, tablas, figuras (sí van a git)
Poster/poster.typ
LaTeX_template/report.tex
```

---

### Task 1: Andamiaje del repositorio

**Files:** Create `pyproject.toml`, `requirements.txt`, `.gitignore`, `msdemosaic/__init__.py`, `tests/__init__.py`.

- [ ] **Step 1:** `git init` en la raíz del proyecto. `.gitignore`: `data/`, `checkpoints/`, `.venv/`, `__pycache__/`, `.DS_Store`, `*.pt`, `DemosaickingMultispectral3DSpherePacking/`.
- [ ] **Step 2:** `requirements.txt`: `numpy scipy scikit-image imageio h5py torch matplotlib pytest tqdm`. `pyproject.toml` mínimo que declare el paquete `msdemosaic` y `[tool.pytest.ini_options] testpaths = ["tests"]`.
- [ ] **Step 3:** `python3 -m venv .venv && .venv/bin/pip install -r requirements.txt && .venv/bin/pip install -e .`
- [ ] **Step 4:** Run `.venv/bin/python -c "import torch; print(torch.backends.mps.is_available())"` → Expected `True`. Run `.venv/bin/pytest` → Expected "no tests ran".
- [ ] **Step 5:** Commit `chore: project scaffold`.

### Task 2: Patrones y simulador

**Files:** Create `msdemosaic/patterns.py`, `msdemosaic/simulate.py`, `tests/test_patterns.py`, `tests/test_simulate.py`.

**Interfaces — Produces:**
- `TEST_PATTERNS = ("imec", "osp", "btes", "random")`
- `make_pattern(name: str, H: int, W: int) -> np.ndarray` — `(H, W)` int64 en `0..15`, anclado en `(0, 0)`.
- `sample_training_pattern(rng: np.random.Generator, H: int, W: int) -> np.ndarray` — misma forma.
- `pattern_to_mask(pattern) -> np.ndarray` — `(H, W, 16)` float32 one-hot.
- `simulate(cube, pattern) -> (mosaic, mask)` — `mosaic` `(H, W)` float32, `mask` `(H, W, 16)` float32.

Definiciones (kernels 4 × 4 en teselado, copiados de `plan_proyecto.typ:52-54` **restando 1**):
- `imec`: `[[6,7,5,4],[14,15,13,12],[10,11,9,8],[2,3,1,0]]`
- `btes`: `[[0,12,3,14],[8,4,10,6],[2,15,1,13],[11,7,9,5]]`
- `random`: `[[10,2,13,5],[0,8,15,3],[12,6,1,11],[4,14,7,9]]`
- `osp`: `P[i, j] = (7*(i+1) + 4*(j+1)) % 16` (no es teselado 4 × 4: período 16 en filas, 4 en columnas).
- `sample_training_pattern`: con probabilidad 0,5 modo *tile* (una permutación aleatoria de `0..15` como kernel 4 × 4 teselado) y 0,5 modo *block* (cada bloque 4 × 4 alineado recibe una permutación independiente). Requiere `H, W` múltiplos de 4. En modo *tile*, re-sortear si el kernel coincide con `imec`, `btes` o `random`.

- [ ] **Step 1: Tests que fallan** (`tests/test_patterns.py`, `tests/test_simulate.py`)

```python
def test_imec_kernel():
    assert make_pattern("imec", 8, 8)[:4, :4].tolist() == [[6,7,5,4],[14,15,13,12],[10,11,9,8],[2,3,1,0]]
    assert (make_pattern("imec", 8, 8)[4:, 4:] == make_pattern("imec", 8, 8)[:4, :4]).all()

def test_osp_formula():
    P = make_pattern("osp", 32, 32)
    assert P[0, 0] == 11 and P[1, 0] == 2 and P[0, 1] == 15
    assert (P[16:, :] == P[:16, :]).all()

@pytest.mark.parametrize("name", TEST_PATTERNS)
def test_balanced(name):
    P = make_pattern(name, 64, 64)
    assert P.dtype == np.int64 and np.bincount(P.ravel(), minlength=16).tolist() == [256] * 16

def test_training_sampler_balanced_and_disjoint():
    rng = np.random.default_rng(0)
    forbidden = [make_pattern(n, 4, 4).tolist() for n in ("imec", "btes", "random")]
    tiled = 0
    for _ in range(200):
        P = sample_training_pattern(rng, 64, 64)
        assert np.bincount(P.ravel(), minlength=16).tolist() == [256] * 16
        is_tile = (P == np.tile(P[:4, :4], (16, 16))).all()
        tiled += is_tile
        if is_tile:
            assert P[:4, :4].tolist() not in forbidden
    assert 60 < tiled < 140   # both modes are used

def test_unknown_pattern_raises():
    with pytest.raises(ValueError):
        make_pattern("bayer", 8, 8)

def test_simulate():
    rng = np.random.default_rng(0)
    cube = rng.random((16, 16, 16), dtype=np.float32)
    P = make_pattern("osp", 16, 16)
    mosaic, mask = simulate(cube, P)
    assert mosaic.shape == (16, 16) and mask.shape == (16, 16, 16)
    assert mosaic.dtype == np.float32 and mask.dtype == np.float32
    assert (mask.sum(-1) == 1).all()
    assert mosaic[3, 5] == cube[3, 5, P[3, 5]]
    assert np.allclose((cube * mask).sum(-1), mosaic)
```

- [ ] **Step 2:** Run `.venv/bin/pytest tests/test_patterns.py tests/test_simulate.py -v` → FAIL (ImportError).
- [ ] **Step 3:** Implementar `patterns.py` y `simulate.py` con las firmas y definiciones de arriba.
- [ ] **Step 4:** Run mismos tests → PASS.
- [ ] **Step 5:** Commit `feat: MSFA patterns and capture simulator`.

### Task 3: Métricas

**Files:** Create `msdemosaic/metrics.py`, `tests/test_metrics.py`.

**Interfaces — Produces:**
- `psnr(recon, cube) -> float` — pico 1, MSE sobre el cubo entero, piso de MSE `1e-10` (máximo 100 dB).
- `psnr_per_band(recon, cube) -> np.ndarray` — `(16,)`.
- `ssim(recon, cube) -> float` — media sobre bandas de `skimage.metrics.structural_similarity(..., data_range=1.0)`.
- `sam(recon, cube) -> float` — grados, media sobre píxeles, `eps = 1e-8` en las normas, coseno recortado a `[-1, 1]`.
- `evaluate(recon, cube) -> dict` — claves exactas `"psnr"`, `"ssim"`, `"sam"`; recorta `recon` a `[0, 1]` antes de medir.

- [ ] **Step 1: Tests que fallan**

```python
def test_psnr_known_value():
    cube = np.full((32, 32, 16), 0.5, np.float32)
    assert psnr(cube + 0.1, cube) == pytest.approx(20.0, abs=1e-3)

def test_identical():
    cube = np.random.default_rng(0).random((32, 32, 16), dtype=np.float32)
    m = evaluate(cube, cube)
    assert set(m) == {"psnr", "ssim", "sam"}
    assert m["psnr"] == pytest.approx(100.0) and m["ssim"] == pytest.approx(1.0) and m["sam"] < 0.1

def test_sam_scale_invariant_and_orthogonal():
    cube = np.random.default_rng(0).random((8, 8, 16), dtype=np.float32) + 0.1
    assert sam(0.5 * cube, cube) < 0.1
    a = np.zeros((8, 8, 16), np.float32); a[..., 0] = 1
    b = np.zeros((8, 8, 16), np.float32); b[..., 1] = 1
    assert sam(a, b) == pytest.approx(90.0, abs=1e-3)

def test_sam_black_pixels_no_nan():
    assert np.isfinite(sam(np.zeros((8, 8, 16), np.float32), np.zeros((8, 8, 16), np.float32)))

def test_evaluate_clips():
    cube = np.ones((32, 32, 16), np.float32)
    assert evaluate(cube + 5.0, cube)["psnr"] == pytest.approx(100.0)

def test_psnr_per_band_shape():
    cube = np.random.default_rng(0).random((32, 32, 16), dtype=np.float32)
    assert psnr_per_band(cube * 0.9, cube).shape == (16,)
```

- [ ] **Step 2:** Run `.venv/bin/pytest tests/test_metrics.py -v` → FAIL.
- [ ] **Step 3:** Implementar `metrics.py`.
- [ ] **Step 4:** Run → PASS.
- [ ] **Step 5:** Commit `feat: PSNR, SSIM, SAM metrics`.

### Task 4: Datos (CAVE, ARAD-1K, splits, caché, parches)

**Files:** Create `msdemosaic/data.py`, `scripts/download_cave.py`, `tests/test_data.py`.

**Interfaces — Produces:**
- `BAND_IDX: list[int]` = `list(range(0, 31, 2))`.
- `select_bands(cube31) -> np.ndarray` — `(H, W, 16)`.
- `load_cave_scene(scene_dir: Path) -> np.ndarray` — `(512, 512, 31)` float32 `[0, 1]`. Busca recursivamente `*_ms_NN.png` (NN = 01..31), ordena por NN, ignora otros archivos; si el PNG tiene 3 dimensiones toma el canal 0; divide por 65535 si `uint16`, por 255 si `uint8`; `ValueError` si no hay exactamente 31.
- `load_arad_scene(mat_path: Path) -> np.ndarray` — `(482, 512, 31)` float32; lee con `h5py` la clave `"cube"` y transpone a `(H, W, 31)` (verificar eje con la primera escena real: la forma final debe ser `(482, 512, 31)`).
- `list_scenes(data_root: Path, data_mode: str) -> dict[str, list[Path]]` — claves `"train"`, `"val"`, `"test"`.
  - `"arad"`: `data_root/ARAD_1K/Train_Spec/*.mat` ordenados, primeros 250 → 220 train + 30 val (12 %); test = las 32 carpetas de `data_root/CAVE/`.
  - `"cave-split"`: 32 carpetas CAVE ordenadas, barajadas con `np.random.default_rng(0)` → 21 train, 3 val, 8 test.
- `load_cube16(path: Path, cache_dir: Path) -> np.ndarray` — carga (CAVE si es carpeta, ARAD si es `.mat`), selecciona bandas, guarda/lee `cache_dir/<stem>.npy` en `float16`, devuelve `float32`.
- `sample_patch(cubes: list[np.ndarray], rng, patch: int = 64) -> np.ndarray` — recorte aleatorio de una escena aleatoria + flip horizontal/vertical y `rot90` aleatorios; devuelve `(patch, patch, 16)` float32 contiguo.

- [ ] **Step 1 (humanos, en paralelo, hoy):** (a) descargar CAVE a `data/CAVE/` con `scripts/download_cave.py` (URL a verificar: `https://www.cs.columbia.edu/CAVE/databases/multispectral/zip/complete_ms_data.zip`; si cambió, tomarla de la página del dataset); (b) obtener `Train_Spec` de ARAD-1K a `data/ARAD_1K/Train_Spec/` — registrarse en NTIRE 2022 Spectral Recovery (CodaLab) y revisar también los enlaces de datos del README del repo `caiyuanhao1998/MST-plus-plus`. **Corte: sáb 14 nov 23:59.** Si ARAD-1K no está en disco, todo el proyecto usa `--data-mode cave-split` (plan B de la ppt) y no se vuelve atrás.
- [ ] **Step 2: Tests que fallan** — usan fixtures sintéticos en `tmp_path` (no requieren datos reales):

```python
def _fake_cave(root, name, dtype=np.uint16, rgba=False, nested=True):
    d = root / f"{name}_ms" / f"{name}_ms" if nested else root / f"{name}_ms"
    d.mkdir(parents=True)
    for b in range(1, 32):
        img = np.full((512, 512), b * 1000 if dtype == np.uint16 else b, dtype)
        if rgba: img = np.stack([img] * 4, -1)
        imageio.imwrite(d / f"{name}_ms_{b:02d}.png", img)
    (d / "Thumbs.db").write_bytes(b"x"); imageio.imwrite(d / f"{name}_RGB.bmp", np.zeros((4, 4, 3), np.uint8))
    return root / f"{name}_ms"

def test_cave_loader_16bit_nested(tmp_path):
    cube = load_cave_scene(_fake_cave(tmp_path, "toy"))
    assert cube.shape == (512, 512, 31) and cube.dtype == np.float32
    assert cube[0, 0, 0] == pytest.approx(1000 / 65535) and cube[0, 0, 30] == pytest.approx(31000 / 65535)

def test_cave_loader_8bit_rgba(tmp_path):
    cube = load_cave_scene(_fake_cave(tmp_path, "toy", dtype=np.uint8, rgba=True, nested=False))
    assert cube.shape == (512, 512, 31) and cube[0, 0, 30] == pytest.approx(31 / 255)

def test_cave_loader_missing_band(tmp_path):
    d = _fake_cave(tmp_path, "toy"); next(d.rglob("*_ms_07.png")).unlink()
    with pytest.raises(ValueError):
        load_cave_scene(d)

def test_select_bands():
    cube = np.arange(31, dtype=np.float32)[None, None, :] * np.ones((2, 2, 1), np.float32)
    assert select_bands(cube)[0, 0].tolist() == list(range(0, 31, 2))

def test_cave_split_is_deterministic_and_disjoint(tmp_path):
    for i in range(32): (tmp_path / "CAVE" / f"s{i:02d}_ms").mkdir(parents=True)
    a, b = list_scenes(tmp_path, "cave-split"), list_scenes(tmp_path, "cave-split")
    assert a == b and [len(a[k]) for k in ("train", "val", "test")] == [21, 3, 8]
    assert len(set(a["train"]) | set(a["val"]) | set(a["test"])) == 32

def test_sample_patch():
    cubes = [np.random.default_rng(0).random((100, 120, 16), dtype=np.float32)]
    p = sample_patch(cubes, np.random.default_rng(1), 64)
    assert p.shape == (64, 64, 16) and p.dtype == np.float32 and p.flags["C_CONTIGUOUS"]
```

- [ ] **Step 3:** Run `.venv/bin/pytest tests/test_data.py -v` → FAIL.
- [ ] **Step 4:** Implementar `data.py` y `scripts/download_cave.py` (descarga + unzip a `data/CAVE/`, idempotente).
- [ ] **Step 5:** Run → PASS. Con CAVE real en disco: `.venv/bin/python -c "from pathlib import Path; from msdemosaic.data import *; s=list_scenes(Path('data'),'cave-split'); c=load_cube16(s['test'][0], Path('data/cache')); print(c.shape, c.min(), c.max())"` → Expected `(512, 512, 16)` y valores en `[0, 1]`.
- [ ] **Step 6:** Commit `feat: CAVE/ARAD loaders, splits, patch sampling`.

### Task 5: Baseline WB

**Files:** Create `msdemosaic/wb.py`, `tests/test_wb.py`.

**Interfaces — Consumes:** `simulate`, `make_pattern`. **Produces:** `wb(mosaic, mask) -> np.ndarray` `(H, W, 16)` float32.

Algoritmo (Brauers y Aach generalizado a cualquier patrón por convolución normalizada): `F = outer(f, f)` con `f = [1, 2, 3, 4, 3, 2, 1]`; por banda `k`: `recon_k = conv(mosaic * mask_k, F) / max(conv(mask_k, F), 1e-8)` con `scipy.ndimage.convolve(mode="constant")`; al final los píxeles medidos conservan su valor (`recon[mask == 1] = mosaic` en su banda).

- [ ] **Step 1: Tests que fallan**

```python
@pytest.mark.parametrize("name", TEST_PATTERNS)
def test_wb_reproduces_flat_spectra(name):
    cube = np.ones((32, 32, 16), np.float32) * np.linspace(0.1, 0.9, 16, dtype=np.float32)
    recon = wb(*simulate(cube, make_pattern(name, 32, 32)))
    assert recon.shape == cube.shape and recon.dtype == np.float32
    assert np.allclose(recon, cube, atol=1e-5)

def test_wb_keeps_measured_pixels_and_beats_zero_fill():
    rng = np.random.default_rng(0)
    yy, xx = np.mgrid[0:64, 0:64] / 64
    cube = np.stack([0.5 + 0.4 * np.sin(2 * np.pi * (yy + xx) + k / 3) for k in range(16)], -1).astype(np.float32)
    mosaic, mask = simulate(cube, make_pattern("imec", 64, 64))
    recon = wb(mosaic, mask)
    assert np.allclose(recon[mask == 1], cube[mask == 1])
    assert psnr(recon, cube) > 30
```

- [ ] **Step 2:** Run `.venv/bin/pytest tests/test_wb.py -v` → FAIL.
- [ ] **Step 3:** Implementar `wb.py`.
- [ ] **Step 4:** Run → PASS.
- [ ] **Step 5:** Commit `feat: weighted bilinear baseline`.

### Task 6: Evaluación métodos × patrones × escenas

**Files:** Create `msdemosaic/evaluate_all.py`, `tests/test_evaluate_all.py`.

**Interfaces — Consumes:** `list_scenes`, `load_cube16`, `make_pattern`, `simulate`, `evaluate`, `psnr_per_band`, `wb`, y desde T7 `load_method`.
**Produces:**
- `run_grid(methods: dict[str, Callable], cubes: dict[str, np.ndarray], patterns=TEST_PATTERNS) -> tuple[list[dict], list[dict]]` — filas por escena `{"method", "pattern", "scene", "psnr", "ssim", "sam"}` y filas por banda `{"method", "pattern", "scene", "band", "psnr"}`.
- `summarize(rows) -> list[dict]` — `{"method", "pattern", "psnr_mean", "psnr_std", "ssim_mean", "ssim_std", "sam_mean", "sam_std"}`.
- CLI: `python -m msdemosaic.evaluate_all --data-root data --data-mode {arad,cave-split} --checkpoints checkpoints --out results` → escribe `results/per_scene.csv`, `results/per_band.csv`, `results/table.md`, `results/table.tex` (celdas `media ± desv.`, filas en orden `wb, fixed_imec, fixed_osp, agnostic`, columnas `imec, osp, btes, random`). Métodos: siempre `wb`; además cada `checkpoints/{fixed_imec,fixed_osp,agnostic}.pt` que exista (los ausentes se omiten con un aviso, no con error).

- [ ] **Step 1: Test que falla**

```python
def test_run_grid_and_summarize():
    rng = np.random.default_rng(0)
    cubes = {f"s{i}": rng.random((32, 32, 16), dtype=np.float32) for i in range(3)}
    rows, band_rows = run_grid({"wb": wb, "oracle_free_zero": lambda y, m: np.zeros(m.shape, np.float32)}, cubes)
    assert len(rows) == 2 * 4 * 3 and len(band_rows) == 2 * 4 * 3 * 16
    s = summarize(rows)
    assert len(s) == 8
    get = lambda meth, pat: next(r for r in s if r["method"] == meth and r["pattern"] == pat)
    assert get("wb", "imec")["psnr_mean"] > get("oracle_free_zero", "imec")["psnr_mean"]
    assert get("wb", "imec")["psnr_std"] >= 0
```

- [ ] **Step 2:** Run `.venv/bin/pytest tests/test_evaluate_all.py -v` → FAIL.
- [ ] **Step 3:** Implementar `evaluate_all.py`.
- [ ] **Step 4:** Run → PASS. Con datos reales: `.venv/bin/python -m msdemosaic.evaluate_all --data-root data --data-mode cave-split --checkpoints checkpoints --out results` → Expected: `results/table.md` con la fila `wb` en los 4 patrones, PSNR finito y del mismo orden entre patrones. **Hito "pipeline funcionando".**
- [ ] **Step 5:** Commit `feat: evaluation grid and WB reference row`.

### Task 7: U-Net y adaptador de método

**Files:** Create `msdemosaic/model.py`, `tests/test_model.py`.

**Interfaces — Produces:**
- `class UNet(nn.Module)`: `__init__(self, use_mask: bool, base: int = 48)`; `forward(y, m=None)` con `y` `(B, 1, H, W)`, `m` `(B, 16, H, W)` → `(B, 16, H, W)`. Entrada `cat([y, m])` (17 canales) si `use_mask`, si no `y` (1 canal).
- Arquitectura: 2 niveles de submuestreo. Bloque = 2 × (Conv 3 × 3 + ReLU), sin normalización. Canales `base → 2·base → 4·base`; `MaxPool2d(2)` al bajar; `ConvTranspose2d(k=2, s=2)` al subir; concatenación de skips; `Conv 1 × 1` final a 16, sin activación. `forward` rellena con `F.pad(mode="reflect")` hasta múltiplo de 4 y recorta la salida a `(H, W)`.
- `pick_device() -> torch.device` — `cuda` > `mps` > `cpu`.
- `save_checkpoint(path, model, *, train_pattern: str, iters: int, val_psnr: float)` — dict con claves `"state_dict"`, `"use_mask"`, `"base"`, `"train_pattern"`, `"iters"`, `"val_psnr"`.
- `load_method(ckpt_path, device=None) -> Callable[[np.ndarray, np.ndarray], np.ndarray]` — carga con `map_location`, `eval()`, y devuelve un callable `(mosaic (H, W), mask (H, W, 16)) -> recon (H, W, 16)` float32 (bajo `torch.no_grad()`; ignora la máscara si `use_mask` es falso).

- [ ] **Step 1: Tests que fallan**

```python
@pytest.mark.parametrize("use_mask", [True, False])
def test_param_budget_and_shape(use_mask):
    net = UNet(use_mask=use_mask)
    assert sum(p.numel() for p in net.parameters()) < 2_000_000
    y, m = torch.rand(2, 1, 64, 64), torch.rand(2, 16, 64, 64)
    assert net(y, m if use_mask else None).shape == (2, 16, 64, 64)

def test_arbitrary_size():
    net = UNet(use_mask=True)
    assert net(torch.rand(1, 1, 482, 510), torch.rand(1, 16, 482, 510)).shape == (1, 16, 482, 510)

def test_checkpoint_roundtrip_as_method(tmp_path):
    net = UNet(use_mask=True)
    save_checkpoint(tmp_path / "a.pt", net, train_pattern="random", iters=1, val_psnr=0.0)
    method = load_method(tmp_path / "a.pt", device=torch.device("cpu"))
    cube = np.random.default_rng(0).random((32, 32, 16), dtype=np.float32)
    recon = method(*simulate(cube, make_pattern("btes", 32, 32)))
    assert recon.shape == (32, 32, 16) and recon.dtype == np.float32 and np.isfinite(recon).all()
```

- [ ] **Step 2:** Run `.venv/bin/pytest tests/test_model.py -v` → FAIL.
- [ ] **Step 3:** Implementar `model.py`.
- [ ] **Step 4:** Run → PASS.
- [ ] **Step 5:** Commit `feat: small U-Net and method adapter`.

### Task 8: Entrenamiento

**Files:** Create `msdemosaic/train.py`, `tests/test_train.py`, `notebooks/kaggle_train.ipynb`.

**Interfaces — Consumes:** `UNet`, `save_checkpoint`, `pick_device`, `sample_patch`, `make_pattern`, `sample_training_pattern`, `pattern_to_mask`, `list_scenes`, `load_cube16`.
**Produces:**
- `make_batch(cubes, rng, train_pattern: str, batch: int = 32, patch: int = 64) -> tuple[Tensor, Tensor, Tensor]` — `(y (B,1,64,64), m (B,16,64,64), x (B,16,64,64))`. **Un patrón por lote**: `make_pattern(train_pattern, 64, 64)` si `train_pattern ∈ {"imec", "osp"}`; `sample_training_pattern(rng, 64, 64)` si `"random"`.
- `train(cubes_train, cubes_val, *, train_pattern, iters, out, batch=32, lr=1e-3, val_every=1000, seed=0, device=None) -> float` — devuelve el mejor PSNR de validación. `use_mask = (train_pattern == "random")`. Adam, `CosineAnnealingLR(T_max=iters, eta_min=1e-5)`, pérdida L1. Validación: 256 parches fijos (`rng` semilla 0) de las escenas de validación; para `"random"`, cada parche de validación usa uno de 8 patrones fijos sorteados con semilla 1234. Guarda en `out` solo cuando mejora el PSNR de validación; escribe `out.with_suffix(".log.csv")` con `iter, train_l1, val_psnr`.
- CLI: `python -m msdemosaic.train --pattern {imec,osp,random} --data-root data --data-mode {arad,cave-split} --iters 40000 --out checkpoints/<name>.pt`. Nombres de salida fijos: `fixed_imec.pt`, `fixed_osp.pt`, `agnostic.pt`.
- `notebooks/kaggle_train.ipynb`: clona/sube el repo, `pip install -e .`, apunta `--data-root` al dataset de Kaggle, corre los 3 entrenamientos y deja los `.pt` y `.log.csv` en `/kaggle/working/`.

- [ ] **Step 1: Tests que fallan**

```python
def _smooth_cubes(n=4):
    yy, xx = np.mgrid[0:96, 0:96] / 96
    return [np.stack([0.5 + 0.4 * np.sin(2 * np.pi * (yy * (i + 1) + xx) + k / 3) for k in range(16)], -1).astype(np.float32) for i in range(n)]

def test_make_batch_fixed_uses_anchored_pattern():
    y, m, x = make_batch(_smooth_cubes(), np.random.default_rng(0), "imec", batch=4)
    assert y.shape == (4, 1, 64, 64) and m.shape == (4, 16, 64, 64) and x.shape == (4, 16, 64, 64)
    assert (m[0].argmax(0).numpy() == make_pattern("imec", 64, 64)).all()
    assert torch.allclose((x * m).sum(1, keepdim=True), y)

def test_make_batch_random_changes_between_batches():
    rng = np.random.default_rng(0)
    m1 = make_batch(_smooth_cubes(), rng, "random", batch=2)[1]
    m2 = make_batch(_smooth_cubes(), rng, "random", batch=2)[1]
    assert (m1[0] == m1[1]).all() and not (m1[0] == m2[0]).all()

@pytest.mark.parametrize("pattern,use_mask", [("imec", False), ("random", True)])
def test_train_smoke(tmp_path, pattern, use_mask):
    cubes = _smooth_cubes()
    out = tmp_path / "m.pt"
    best = train(cubes[:3], cubes[3:], train_pattern=pattern, iters=150, out=out, batch=8, val_every=50, device=torch.device("cpu"))
    ck = torch.load(out, map_location="cpu")
    assert ck["use_mask"] is use_mask and ck["train_pattern"] == pattern and best > 15
    log = np.genfromtxt(out.with_suffix(".log.csv"), delimiter=",", names=True)
    assert log["train_l1"][-1] < log["train_l1"][0]
```

- [ ] **Step 2:** Run `.venv/bin/pytest tests/test_train.py -v` → FAIL.
- [ ] **Step 3:** Implementar `train.py` y el notebook.
- [ ] **Step 4:** Run → PASS (< 2 min en CPU).
- [ ] **Step 5:** Medir velocidad: `.venv/bin/python -m msdemosaic.train --pattern imec --data-root data --data-mode cave-split --iters 300 --out checkpoints/smoke.pt`. Si MPS da < 4 it/s, los entrenamientos de 40 000 iteraciones van a Kaggle (plan B de la ppt); si sigue lento, bajar a 20 000 iteraciones antes que cambiar el modelo.
- [ ] **Step 6:** Lanzar `fixed_imec` (40 000 iteraciones) con el modo de datos decidido en T4. Commit `feat: training loop and Kaggle notebook`.

### Task 9: Tabla H1–H3 y punto de decisión

**Files:** Modify `msdemosaic/evaluate_all.py` (añadir `hypotheses`), `tests/test_evaluate_all.py`. Output `results/`.

**Interfaces — Produces:** `hypotheses(summary: list[dict]) -> dict` con claves:
- `"h1_drop_db"`: PSNR de `fixed_imec` en `imec` menos su PSNR medio en `osp, btes, random` (y análogo `"h1_drop_db_osp"` si existe `fixed_osp`).
- `"h2_gap_db"`: `{"imec": psnr(fixed_imec, imec) − psnr(agnostic, imec), "osp": psnr(fixed_osp, osp) − psnr(agnostic, osp)}`; `"h2_holds"`: todos los gaps < 1,0.
- `"h3_margin_db"`: por cada par (red fija, patrón no visto por ella), `psnr(agnostic) − psnr(fija)`; `"h3_holds"`: todos > 0.
- Claves cuyos métodos falten → `None`. La CLI escribe `results/hypotheses.json`.

- [ ] **Step 1: Test que falla**

```python
def test_hypotheses_from_synthetic_summary():
    def row(m, p, v): return {"method": m, "pattern": p, "psnr_mean": v}
    s = [row("fixed_imec", "imec", 40), row("fixed_imec", "osp", 25), row("fixed_imec", "btes", 26), row("fixed_imec", "random", 24),
         row("fixed_osp", "osp", 41), row("fixed_osp", "imec", 25), row("fixed_osp", "btes", 25), row("fixed_osp", "random", 25),
         *[row("agnostic", p, 39.5) for p in ("imec", "osp", "btes", "random")]]
    h = hypotheses(s)
    assert h["h1_drop_db"] == pytest.approx(15.0)
    assert h["h2_gap_db"] == {"imec": pytest.approx(0.5), "osp": pytest.approx(1.5)} and h["h2_holds"] is False
    assert h["h3_holds"] is True and len(h["h3_margin_db"]) == 6
    assert hypotheses([r for r in s if r["method"] != "agnostic"])["h2_gap_db"] is None
```

- [ ] **Step 2:** Run → FAIL. **Step 3:** Implementar. **Step 4:** Run → PASS.
- [ ] **Step 5 (lun 16 nov — punto de decisión):** con `checkpoints/fixed_imec.pt`, correr la CLI de T6. Criterio: `fixed_imec` supera a `wb` en la columna `imec` y `h1_drop_db` está medido. **Sí →** lanzar `agnostic` y seguir a nivel B. **No →** depurar como máximo 1 día (sobreajustar 1 escena debe dar > 40 dB; revisar fase del patrón y rango de datos); si sigue fallando, cerrar en nivel A y saltar T12.
- [ ] **Step 6 (sáb 21 nov):** repetir con los 3 checkpoints → tabla completa. Commit `feat: hypothesis summary` y luego `results: full table`.

### Task 10: Figuras

**Files:** Create `msdemosaic/figures.py`. Output `results/figs/*.pdf` y `*.png`.

**Interfaces — Consumes:** `results/per_scene.csv`, `results/per_band.csv`, checkpoints, datos de prueba.
CLI: `python -m msdemosaic.figures --data-root data --data-mode <modo> --checkpoints checkpoints --results results`. Genera (omitiendo métodos sin checkpoint):
1. `psnr_bars` — barras agrupadas, método × patrón, con barras de error (desv.).
2. `psnr_per_band` — PSNR por banda (eje x en nm: `400 + 20·k`), una línea por método, un panel por patrón.
3. `visual_<pattern>` para `imec` y `osp` — recorte 128 × 128 de una escena de prueba en falso color (bandas `13, 7, 3` ≈ 660/540/460 nm como RGB): referencia + cada método + mapa de error absoluto (para ver zipper y falso color).
4. `spectra` — firmas espectrales en 2 píxeles fijos de esa escena: referencia vs. cada método (patrón `osp`).
Colores: `wb` gris, `fixed_imec` azul `(42, 85, 150)`, `fixed_osp` naranja, `agnostic` rojo ACCENT `(155, 24, 34)`.

- [ ] **Step 1:** Implementar `figures.py` (sin test unitario; verificación visual).
- [ ] **Step 2:** Run la CLI → Expected: los archivos existen. Abrir cada PNG y comprobar: ejes rotulados, leyenda, nada recortado, orden de métodos igual al de la tabla.
- [ ] **Step 3:** Commit `feat: result figures`.

### Task 11: Póster de avance (mié 18 nov)

**Files:** Create `Poster/poster.typ` (+ PDF).

Formato asumido: A1 vertical (confirmar con el profesor; es un parámetro `paper:` al inicio del archivo). Idioma español. Color ACCENT `rgb(155, 24, 34)`; reutilizar de `plan_proyecto.typ` las funciones `msfa`, `BAND` y los 4 patrones (copiarlas, no importar el archivo). Autores: Benjamín Cisternas y `[Integrante 2]` (pendiente de reemplazar).

Secciones: (1) Problema y cita de Diaz et al. sec. IV-I; (2) Pregunta + H1–H3; (3) Método: diagrama `[Y, M] → U-Net → X̂` y los 4 patrones dibujados; (4) Montaje experimental (datos, métricas, modo de datos real usado); (5) **Resultados a la fecha**: tabla leída de `results/table.md` con las filas disponibles, `psnr_bars`, `visual_imec`; celdas sin resultado marcadas "en curso", nunca inventadas; (6) Próximos pasos hasta el 25 nov; (7) Referencias.

- [ ] **Step 1:** Escribir `poster.typ` (respetar las notas Typst de `CLAUDE.md`).
- [ ] **Step 2:** `typst compile Poster/poster.typ` → sin errores; revisar el PDF renderizado: una página, sin desbordes, números idénticos a `results/table.md`.
- [ ] **Step 3:** Commit `docs: progress poster`.

### Task 12: Redes del nivel B (jue 19 – vie 20 nov)

**Interfaces — Consumes:** CLI de T8, notebook de Kaggle.

- [ ] **Step 1:** Entrenar `fixed_osp` y `agnostic` (40 000 iteraciones cada una; `agnostic` puede subir a 80 000 si su curva de validación sigue subiendo al terminar).
- [ ] **Step 2:** Revisar los `.log.csv`: el PSNR de validación se estabiliza; si `agnostic` queda > 3 dB bajo `fixed_imec` en validación, un único ajuste permitido: duplicar iteraciones. No cambiar arquitectura (queda como hallazgo, según la ppt: "se reporta y se analiza").
- [ ] **Step 3:** Copiar los `.pt` y `.log.csv` a `checkpoints/`; ejecutar T9 Step 6 y T10. **Dom 22 nov: experimentos cerrados.**

### Task 13: Informe y entrega (lun 23 – mié 25 nov)

**Files:** Create `LaTeX_template/report.tex` (a partir de `proposal.tex`, mismas `refs.bib` y `spconf.sty`), `README.md`. Modify `CLAUDE.md` (fechas nuevas, estado).

- [ ] **Step 1:** `report.tex` en inglés, formato ICIP: Introduction, Related Work, Method, Experimental Setup, Results (tabla desde `results/table.tex`, figuras de `results/figs/`), Discussion (veredicto de H1–H3 con los números de `results/hypotheses.json`, limitaciones: filtros ideales, sin ruido, modo de datos usado), Conclusion. Toda cifra del texto debe salir de `results/`.
- [ ] **Step 2:** `cd LaTeX_template && tectonic report.tex` → compila; ≤ 5 páginas + referencias; proteger `\\[` según nota de `CLAUDE.md`.
- [ ] **Step 3:** `README.md`: instalación, descarga de datos, los 3 comandos (train, evaluate_all, figures), tabla final.
- [ ] **Step 4:** `.venv/bin/pytest` → todo PASS. Commit `docs: final report and README`.

### Task 14 (nivel C, solo si la tabla está cerrada el vie 20 nov): robustez al ruido

**Files:** Modify `msdemosaic/simulate.py`, `msdemosaic/evaluate_all.py`, tests correspondientes.

- [ ] **Step 1:** Añadir `simulate(cube, pattern, noise_sigma: float = 0.0, rng=None)` (ruido gaussiano sobre el mosaico; por defecto idéntico al comportamiento actual). Test: `noise_sigma=0` da el mismo mosaico; con `0.01` la desviación de `mosaic − limpio` es `≈ 0.01` (±10 %).
- [ ] **Step 2:** `evaluate_all --noise-sigma 0.01 --out results/noise` (sin reentrenar) y una fila extra en el informe. PPID, respuestas IMEC reales y `Y ⊙ M` quedan como trabajo futuro.
