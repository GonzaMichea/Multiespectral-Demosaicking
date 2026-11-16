# Proyecto: Demosaicking multiespectral agnóstico al patrón

Curso: Fotografía Computacional 2026-2, Universidad Técnica Federico Santa María.
Equipo: Benjamín Cisternas y Gonzalo Michea. Idioma de trabajo: español
(los documentos tipo paper van en inglés porque el template ICIP lo exige).

## Estado actual (actualizado el lun 16 nov 2026, 14:20)

**Fechas vigentes (hubo prórroga; el 11 nov ya no aplica):**
- **Mié 18 nov, 14:40**: presentación de avance con **póster**. Benjamín necesita
  tiempo para practicarla.
- **Mié 25 nov**: entrega final (informe ICIP + repositorio).

**Hecho:**
- Código completo en `msdemosaic/`; 44 tests pasan (`.venv/bin/pytest`, ~40 s).
- CAVE descargado en `data/CAVE/` (32 escenas, todas cargan bien).
- Fila WB medida en las 8 escenas de prueba: ~32,0 dB PSNR, SSIM ~0,927,
  SAM ~4,96°, casi igual en los 4 patrones (`results/table.md`).
- `results/`, `results/figs/` y `Poster/poster.pdf` regenerados el 16 nov **solo
  con WB**; las filas de las redes dicen "en curso".
- README, `scripts/run_all.sh` y notebook de Kaggle (este último sin probar).

**NO hecho — ninguna red está entrenada:**
- `checkpoints/` está vacío. Hubo dos intentos, ambos detenidos por el usuario
  (12 nov a 6 000/40 000 iteraciones; 16 nov a ~1 000/20 000). Los checkpoints
  parciales se borraron.
- **No lanzar entrenamientos sin que el usuario lo pida explícitamente.** La
  última instrucción fue: "Deja de correr, no corras aún".

**Falta, en orden:**
1. Entrenar las 3 redes (ver "Entrenamiento").
2. `evaluate_all` + `figures` → tabla H1–H3 y figuras finales. Revisar las
   figuras a ojo.
3. Punto de decisión: ¿fija-IMEC supera a WB (~32,0 dB) en IMEC? Si no,
   depurar (sobreajustar 1 escena debe dar > 40 dB; revisar fase del patrón).
4. Recompilar el póster y revisarlo; confirmar formato (asumido A1 vertical).
5. `LaTeX_template/report.tex` (informe ICIP en inglés, a partir de
   `proposal.tex`; toda cifra debe salir de `results/`).
6. Revisión final del código (nadie más lo ha revisado).
7. Nivel C (ruido) no se ha tocado; solo si sobra tiempo.

## Entrenamiento

- Velocidad medida en el Mac M5 con MPS, lote 32: 6–8 it/s.
  40 000 iteraciones ≈ 2 h por red (~6 h las tres); 20 000 ≈ 1 h por red (~3 h).
- Curva conocida de fija-IMEC (validación): 32,2–32,6 dB a 1 000 iteraciones,
  34,8 dB a 6 000, aún subiendo. No se sabe cuánto se pierde con 20 000.
- **Hay pausa y reanudación** (desde el 16 nov): el estado completo se guarda en
  `checkpoints/_running_<nombre>.state.pt` en cada validación (cada 1 000
  iteraciones). Repetir el mismo comando continúa donde quedó; las redes ya
  terminadas se saltan. Si se mata el proceso se pierden a lo más 1 000
  iteraciones (~2–3 min). Hay que repetir con las mismas iteraciones; si se
  cambian, borrar antes el `.state.pt`. Probado en MPS con datos reales.
- Tercer argumento = presupuesto en minutos: `scripts/run_all.sh 20000 cave-split 60`
  entrena ~1 h y se pausa solo en la siguiente validación.
- Todo en un comando (entrena imec → agnóstica → osp y, tras cada red,
  reevalúa, regenera figuras y recompila el póster):

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
typst compile --root . Poster/poster.typ                             # lee results/
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
- U-Net: base 48, 2 submuestreos, sin normalización, 1 055 248 parámetros
  (agnóstica, entrada [Y, M] de 17 canales) y 1 048 336 (fija, solo Y).
  Pérdida L1, Adam 1e-3 con coseno, parches 64×64, lote 32.
- Las redes fijas no reciben máscara; la agnóstica sí.
- El test de humo del entrenamiento no exige un PSNR mínimo: la red tarda
  200–500 iteraciones en salir de la solución constante.
- Git: repo creado el 12 nov, todo en `main`, sin remoto. Fuera de git: `data/`,
  `checkpoints/`, `.venv/`, el PDF del paper y el código MATLAB.

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
- `Poster/poster.typ` (+ PDF): póster A1 en español; lee `results/summary.csv`,
  `results/hypotheses.json` y `results/figs/`, y muestra "en curso" donde falta.
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
