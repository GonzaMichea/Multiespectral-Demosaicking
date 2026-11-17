# Proyecto: Demosaicking multiespectral agnóstico al patrón

Curso: Fotografía Computacional 2026-2, Universidad Técnica Federico Santa María.
Equipo: Benjamín Cisternas y Gonzalo Michea. Idioma de trabajo: español
(los documentos tipo paper van en inglés porque el template ICIP lo exige).

## Estado actual (actualizado el lun 16 nov 2026, 23:15)

**Fechas vigentes (hubo prórroga; el 11 nov ya no aplica):**
- **Mié 18 nov, 14:40**: presentación de avance con **póster**. Benjamín necesita
  tiempo para practicarla.
- **Mié 25 nov**: entrega final (informe ICIP + repositorio).

**Hecho:**
- Código completo en `msdemosaic/`; 69 tests pasan (`.venv/bin/pytest`, ~1,8 min).
- CAVE descargado en `data/CAVE/` (32 escenas, todas cargan bien).
- **Las 4 redes entrenadas (20 000 iteraciones) y evaluadas el 16 nov.**
  `results/` y `results/figs/` tienen los resultados reales (revisados a ojo).
- **Póster en formato WACCI 2026** (17 nov): `WACCI_2026/poster_demosaicking.tex`
  (+ PDF), una página. Guion de la presentación: `Poster/guion_presentacion.md`.
- README, `scripts/run_all.sh` y notebook de Kaggle (este último sin probar y
  anterior al cambio de entrada).

**Revisión del 16 nov (detalle y números en `docs/revision-2026-11-16.md`):**
- Patrones, bandas, WB, simulador y PSNR verificados contra el paper y su código
  (el WB reproduce la Tabla I con 0,13 dB de diferencia).
- Pruebas cortas (2 000 iteraciones, escenas de validación) mostraron que el
  diseño original no alcanzaba: la agnóstica con entrada [Y, M] quedaba en
  28,3 dB, bajo WB (32,6) y 7 dB bajo la fija-IMEC.
- **Cambio de diseño (implementado, 66 tests pasan):** todas las redes reciben
  la entrada [WB(Y, M), Y⊙M, M] (`input_mode="wb"`, 48 canales, 1 068 640
  parámetros). Con eso la agnóstica llega a 34,6–35,1 dB y la fija-IMEC a
  36,1 dB en IMEC (y 32,5–33,7 en patrones no vistos).
- **Las redes fijas ahora reciben la máscara**: fija y agnóstica difieren solo
  en los patrones de entrenamiento. Esto cambia la línea base de la presentación
  ("U-Net sin máscara"); esa red queda como ablación (`--input mosaic`).
- Vista previa: H1 −3,2 dB, H2 1,5 dB de brecha en IMEC (aún no < 1 dB),
  H3 +1,4 a +2,5 dB. El residuo sobre WB no funciona (las ReLU mueren).
- `scripts/pilot.py` corre las pruebas cortas.

**Resultados finales (16 nov; 8 escenas de prueba, PSNR medio en dB; la fuente
es `results/`, no esta tabla):**

| Método | IMEC | OSP | BTES | Aleatorio |
|---|---|---|---|---|
| WB | 31,99 | 32,13 | 31,98 | 31,99 |
| U-Net sin máscara (IMEC) | 35,70 | 20,75 | 19,76 | 20,24 |
| U-Net fija-IMEC | 36,64 | 29,25 | 28,77 | 29,74 |
| U-Net fija-OSP | 25,55 | 37,73 | 28,90 | 28,95 |
| U-Net agnóstica | 35,43 | 36,33 | 36,14 | 36,07 |

- **H1 se cumple:** fuera de su patrón la fija-IMEC pierde 7,38 dB, la fija-OSP
  9,94 dB y la red sin máscara 15,45 dB; las tres quedan bajo WB.
- **H2 no se cumple:** la agnóstica queda a 1,21 dB (IMEC) y 1,40 dB (OSP) de la
  red fija de ese patrón (umbral: 1 dB). Por escena queda bajo 1 dB en 2 de 8
  (IMEC) y 3 de 8 (OSP). Con 8 escenas el IC 95 % de la brecha incluye 1 dB
  (IMEC 0,66–1,75; OSP 0,88–1,93): no alcanza para afirmar que la supere ni que
  quede bajo ella. Frente a la red sin máscara, la diferencia en IMEC es 0,27 dB
  en promedio, pero varía mucho por escena (−2,85 a +1,53 dB). El póster dice
  "no se cumple por poco" (pedido del usuario el 17 nov: mostrar que estuvo cerca).
- **H3 se cumple:** en patrones no vistos la agnóstica supera a las fijas por
  6,33 a 9,89 dB, en las 8 escenas de los 6 casos.
- La agnóstica supera a WB por 3,4 a 4,2 dB en promedio, pero no en todas las
  escenas (stuffed_toys en IMEC: 35,22 contra 35,37 de WB).
- Entrenamiento: `caffeinate -i scripts/run_all.sh 20000 cave-split`, de 19:50 a
  23:00 (~45 min por red). Existe `checkpoints/ALL_DONE`; no hay nada corriendo.
- **Antes de lanzar otro entrenamiento, preguntar al usuario** (ha detenido
  entrenamientos dos veces por batería o por tener que cerrar el Mac).

**Falta, en orden:**
1. Benjamín: practicar el póster (mar 17) con el guion e imprimir el póster WACCI.
2. Decidir qué hacer con H2 para el informe: reportarla como no cumplida (es un
   resultado válido) o probar más capacidad (`base` 64) y 40 000 iteraciones
   (más de 1 h por red; preguntar antes de lanzar).
3. `LaTeX_template/report.tex` (informe ICIP en inglés, a partir de
   `proposal.tex`; toda cifra debe salir de `results/`).
4. Revisión final del código (nadie más lo ha revisado).
5. Nivel C (ruido) no se ha tocado; solo si sobra tiempo.
6. Hay commits locales sin subir a GitHub: push solo si el usuario lo pide.

## Entrenamiento

- Velocidad medida en el Mac M5 con MPS, lote 32: ~8 it/s con la entrada nueva
  (20 000 iteraciones ≈ 43 min por red; 40 000 ≈ 85 min).
- Curvas de validación del entrenamiento real (`checkpoints/<red>.log.csv`,
  `results/figs/training_curves.png`): fija-IMEC 35,5 dB desde ~8 000
  iteraciones, agnóstica 34,8 desde ~13 000, fija-OSP 37,3 desde ~15 000. El L1
  de entrenamiento sigue bajando mientras la validación no sube, así que más
  iteraciones solas probablemente no cierran la brecha de H2 (no se probó).
- **Hay pausa y reanudación** (desde el 16 nov): el estado completo se guarda en
  `checkpoints/_running_<nombre>.state.pt` en cada validación (cada 1 000
  iteraciones). Repetir el mismo comando continúa donde quedó; las redes ya
  terminadas se saltan. Si se mata el proceso se pierden a lo más 1 000
  iteraciones (~2–3 min). Hay que repetir con las mismas iteraciones; si se
  cambian, borrar antes el `.state.pt`. Probado en MPS con datos reales.
- Tercer argumento = presupuesto en minutos: `scripts/run_all.sh 20000 cave-split 60`
  entrena ~1 h y se pausa solo en la siguiente validación.
- Todo en un comando (entrena fija-IMEC → agnóstica → fija-OSP → convencional sin
  máscara y, tras cada red, reevalúa, regenera figuras y recompila el póster):

  ```bash
  caffeinate -i scripts/run_all.sh 20000 cave-split   # o 40000
  ```

  Entrena a `checkpoints/_running_<nombre>.pt` y renombra al terminar, para que
  un entrenamiento a medias no se evalúe como final. Progreso:
  `tail -f checkpoints/fixed_imec.out`. Al final crea `checkpoints/ALL_DONE`.
- No necesita internet. **No cerrar la tapa** (el Mac entra en reposo y el
  entrenamiento se pausa); bloquear pantalla sí se puede. **Enchufar el Mac**:
  el 16 nov estaba en batería al 41 %.
- Correr las tres en paralelo no ayuda (misma GPU). Alternativas para acortar:
  entrenar solo fija-IMEC + agnóstica antes del póster, lote 16, o Kaggle
  (`notebooks/kaggle_train.ipynb`, sin probar, requiere subir datos).

## Comandos

```bash
.venv/bin/pytest                                              # tests
.venv/bin/python -m msdemosaic.train --pattern {imec,osp,random} --data-mode cave-split --iters N --out checkpoints/{fixed_imec,fixed_osp,agnostic}.pt
.venv/bin/python -m msdemosaic.evaluate_all --data-mode cave-split   # → results/
.venv/bin/python -m msdemosaic.figures --data-mode cave-split        # → results/figs/
(cd WACCI_2026 && tectonic -X compile poster_demosaicking.tex)       # póster WACCI
typst compile --root . Poster/poster.typ                             # póster Typst anterior
```

Entorno: `.venv` creado con `uv venv --python 3.12` (el Python del sistema es
3.14), torch 2.15 con MPS. `evaluate_all` usa todo `.pt` final que encuentre en
`checkpoints/` y omite con aviso los que falten.

## Decisión: dirección 3 (red agnóstica al patrón)
- Pregunta: ¿puede una sola red reconstruir con cualquier MSFA, incluso uno no
  visto, sin reentrenar?
- Motivación: Diaz et al., "Multispectral Filter Array Design by Optimal Sphere
  Packing", IEEE TIP 2023 (sec. IV-I) reconoce como limitación tener que
  reentrenar la red (TRevSCI) para cada MSFA.
- Hipótesis: H1 una red de patrón fijo cae en patrones no vistos; H2 la red
  agnóstica queda a < 1 dB de cada red fija en su propio patrón; H3 la agnóstica
  supera a las fijas en patrones no vistos.
- Modelo de captura: Y = Σ_k X_k ⊙ C_k (filtros ideales, 16 bandas, sin ruido).
- Patrones de prueba (ninguno se usa para entrenar la red agnóstica): IMEC 4×4,
  OSP ((a·I + b·J) mod 16 + 1, a=7, b=4), BTES y uno aleatorio.
- Métricas: PSNR, SSIM, SAM (media ± desv.); PSNR por banda; firmas espectrales;
  inspección visual de zipper y falso color.

## Decisiones de implementación (tomadas el 12 nov)
- **Datos: solo CAVE**, modo `cave-split`: partición fija (semilla 0) de
  21 entrenar / 3 validar / 8 probar. ARAD-1K no está en disco (requiere
  registro NTIRE 2022); el cargador ARAD existe pero no se probó con datos
  reales. Consecuencia: la prueba es con 8 escenas, no 32. Escenas de prueba:
  sponges, paints, fake_and_real_food, stuffed_toys, pompoms, beads,
  fake_and_real_tomatoes, watercolors.
- Bandas: 16 de 31 (índices 0, 2, …, 30; 400–700 nm cada 20 nm).
- Patrones en código **0-indexados** (0..15); los kernels de la presentación
  Typst están 1-indexados. Todo patrón se ancla en (0, 0) del parche o imagen.
- Patrones de entrenamiento de la agnóstica: 50 % kernel 4×4 permutado en
  teselado (rechazando los 3 kernels de prueba), 50 % permutación independiente
  por bloque 4×4 (para acercarse a OSP, que no es teselado 4×4). Un patrón por
  lote.
- WB: convolución normalizada con kernel triangular 7×7 (sirve para cualquier
  patrón); los píxeles medidos conservan su valor.
- U-Net: base 48, 2 submuestreos, sin normalización, salida directa (sin
  residuo). Entrada por defecto `wb` = [WB(Y, M), Y⊙M, M], 48 canales,
  1 068 640 parámetros. Ablaciones: `mosaic_mask` = [Y, M] (1 055 248) y
  `mosaic` = solo Y (1 048 336). Pérdida L1, Adam 1e-3 con coseno, parches
  64×64, lote 32.
- Todas las redes reciben la misma entrada (con máscara); cambia solo el
  patrón de entrenamiento (decisión del 16 nov, ver revisión).
- El test de humo del entrenamiento no exige un PSNR mínimo: la red tarda
  200–500 iteraciones en salir de la solución constante.
- Git: todo en `main`; remoto `origin` en GitHub (público). Fuera de git: `data/`,
  `checkpoints/`, `.venv/`, el PDF del paper, el código MATLAB y, desde el 17 nov
  (pedido del usuario), `Poster/`, `Presentacion_Plan/` y `WACCI_2026/`
  (pósteres, presentaciones y guion; versiones antiguas siguen en el historial).

## Alcance
- Nivel A (mínimo): simulador con 4 patrones, WB, U-Net fija-IMEC, H1.
- Nivel B (objetivo): A + U-Net agnóstica + U-Net fija-OSP + tabla completa
  (H1–H3) + figuras.
- Nivel C (extra): ruido gaussiano (único extra previsto); PPID, respuestas
  reales IMEC y [Y, M] vs Y ⊙ M quedan como trabajo futuro.
- Fuera de alcance: cámara real/hardware, reproducir TRevSCI o el paper completo,
  diseñar patrones nuevos, más de 16 bandas.

## Restricciones
- Tema asignado por el profesor: "Multispectral demosaicking".
- Sin GPU propia: Mac M5 16 GB (PyTorch con MPS) + Kaggle (GPU gratis).
- Template LaTeX formato ICIP 2026 en `LaTeX_template/`. No tocar `Template.tex`
  ni `proposal.tex`.

## Archivos
- `msdemosaic/`: `patterns.py`, `simulate.py`, `metrics.py`, `data.py`, `wb.py`,
  `model.py`, `train.py`, `evaluate_all.py`, `figures.py`. Interfaz común:
  `simulate(cube, pattern) -> (mosaic, mask)`, `method(mosaic, mask) -> recon`,
  `evaluate(recon, cube) -> {psnr, ssim, sam}`.
- `tests/`: un archivo por módulo.
- `scripts/download_cave.py`, `scripts/run_all.sh`, `notebooks/kaggle_train.ipynb`.
- `results/`: `per_scene.csv`, `per_band.csv`, `summary.csv`, `table.md`,
  `table.tex`, `hypotheses.json`, `figs/` (sí van a git).
- `WACCI_2026/poster_demosaicking.tex` (+ PDF): **póster vigente**, formato de la
  plantilla WACCI (`Main.tex`, clase `baposter`, 800 × 1000 mm, no tocar
  `Main.tex` ni `baposter.cls`). Compilar desde esa carpeta con
  `tectonic -X compile poster_demosaicking.tex`. Figuras de `../results/figs/`;
  las cifras están copiadas a mano de `results/` (actualizar si se reentrena).
  Con tectonic los degradados de `baposter` no se dibujan: el `.tex` usa
  `headershade=plain` y emula el degradado con franjas. `baposter` escala las
  longitudes (~3×): los tamaños en cm dentro del póster salen más grandes.
- `Poster/poster.typ` (+ PDF): versión anterior en Typst (A1), reemplazada por la
  WACCI; lee `results/` directamente. `Poster/guion_presentacion.md`: guion.
- `docs/superpowers/plans/2026-11-12-demosaicking-agnostico.md`: plan técnico
  por tareas (T1–T14). Su calendario quedó atrasado; manda este archivo.
- `Presentacion_Plan/plan_proyecto.typ` (+ PDF): presentación de tema y alcance
  (ya entregada), tema `touying-simpl-swufe`, ACCENT rgb(155,24,34).
- `LaTeX_template/proposal.tex` (+ PDF): propuesta en formato ICIP. Referencias
  en `refs.bib`. Compilar con `tectonic`.
- `Multispectral_Filter_Array_Design_by_Optimal_Sphere_Packing.pdf` y
  `DemosaickingMultispectral3DSpherePacking/` (MATLAB del paper): solo referencia.
- `data/complete_ms_data.zip` (406 MB): zip original de CAVE, ya descomprimido;
  se puede borrar.

## Notas técnicas
- Typst/touying: no usar `#set text(...)` suelto después de un `==`, porque se
  acumula entre diapositivas. Envolver el cuerpo en `#[ ... ]` o en `#grid(...)`.
  `*negrita*` toma el color del tema; sobre fondo ACCENT usar
  `text(weight: "bold")`.
- El póster se compila desde la raíz con `--root .` (lee archivos de `results/`).
- LaTeX con spconf: en `\name`/`\address`, un `\\` seguido de `[` se interpreta
  como espaciado opcional; proteger con `{[...]}`.
- Figuras visuales: escena `beads_ms` por defecto, recorte central 128×128,
  falso color con bandas 13/7/3.

## Preferencias del usuario
- Respuestas breves; pide que se le avise antes de lanzar procesos largos.
- Las presentaciones HTML publicadas antes en claude.ai quedaron desactualizadas;
  no usarlas.
