# Proyecto: Demosaicking multiespectral agnóstico al patrón

Curso: Fotografía Computacional 2026-2, Universidad Técnica Federico Santa María.
Equipo: Benjamín Cisternas y [Integrante 2]. Idioma de trabajo: español
(los documentos tipo paper van en inglés porque el template ICIP lo exige).

## Restricciones
- Tema asignado por el profesor: "Multispectral demosaicking" (sin más indicaciones).
- Entrega final: **miércoles 11 de noviembre de 2026**. Inicio: 5 de octubre.
- Entrega actual: presentación de **tema y alcance** (solo eso se pide ahora).
- Dedicación: ~6 h/semana por persona (~65 horas-persona en total).
- Sin GPU propia: Mac M5 16 GB (PyTorch con MPS) + Kaggle (GPU gratis).
- El profesor entregó un template LaTeX formato ICIP 2026 (`LaTeX_template/`).

## Decisión: dirección 3 (red agnóstica al patrón)
- Pregunta: ¿puede una sola red reconstruir con cualquier MSFA, incluso uno no
  visto, sin reentrenar?
- Motivación: Diaz et al., "Multispectral Filter Array Design by Optimal Sphere
  Packing", IEEE TIP 2023 (sec. IV-I) reconoce como limitación tener que
  reentrenar la red (TRevSCI) para cada MSFA.
- Hipótesis: H1 una red de patrón fijo cae en patrones no vistos; H2 la red
  agnóstica queda a < 1 dB de cada red fija en su propio patrón; H3 la agnóstica
  supera a las fijas en patrones no vistos.
- Método: U-Net pequeña (< 2 M parámetros, pérdida L1) con entrada [Y, M]
  (mosaico + máscara one-hot de 16 bandas). En cada lote se sortea un patrón
  aleatorio válido (cada banda aparece igual número de veces).
- Modelo de captura: Y = Σ_k X_k ⊙ C_k + N. Filtros ideales, 16 bandas.
- Baselines: WB (Brauers y Aach 2006); U-Net sin máscara entrenada solo con IMEC
  y solo con OSP.
- Patrones de prueba (ninguno se usa para entrenar la red agnóstica): IMEC 4×4,
  OSP ((a·I + b·J) mod 16 + 1, a=7, b=4), BTES y uno aleatorio.
- Datos: entrenamiento ARAD-1K (200–300 escenas; 10–15 % para validación),
  prueba CAVE (32 escenas), 16 de 31 bandas, parches 64×64.
- Métricas: PSNR, SSIM, SAM (media ± desv.); PSNR por banda; firmas espectrales;
  inspección visual de zipper y falso color.

## Alcance
- Nivel A (mínimo, semana 3): simulador con 4 patrones, WB, U-Net fija-IMEC, H1.
- Nivel B (objetivo, semana 5): A + U-Net agnóstica + U-Net fija-OSP + tabla
  completa (H1–H3) + figuras. Estimado ~45–50 h.
- Nivel C (extra): PPID (Mihoubi 2017; no está en el repo del paper, hay que
  implementarlo), ruido Poisson + Gaussiano, respuestas reales IMEC, comparar
  entradas [Y, M] vs Y ⊙ M.
- Fuera de alcance: cámara real/hardware, reproducir TRevSCI o el paper completo,
  diseñar patrones nuevos, más de 16 bandas.

## Cronograma y roles
Persona A: datos, simulador, WB, evaluación. Persona B: redes.
Interfaz común: `simulate(cube, pattern) -> (mosaic, mask)`, `method(mosaic, mask) -> recon`,
`evaluate(recon, cube) -> {psnr, ssim, sam}`.

| Semana | Fechas | A | B |
|---|---|---|---|
| S1 | 5–11 oct | datos, simulador, métricas | PyTorch, U-Net base |
| S2 | 12–18 oct | WB en CAVE | U-Net fija-IMEC |
| S3 | 19–25 oct | evaluación en 4 patrones (H1) | U-Net agnóstica |
| S4 | 26 oct–1 nov | U-Net fija-OSP | U-Net agnóstica (ajuste) |
| S5 | 2–8 nov | tabla completa y figuras (ambos) | |
| S6 | 9–11 nov | informe y presentación (ambos) | |

Hitos: 16 oct pipeline funcionando · **23 oct punto de decisión** (si la red
fija-IMEC no funciona, cerrar en nivel A) · 6 nov experimentos cerrados ·
11 nov entrega final.

## Archivos
- `Multispectral_Filter_Array_Design_by_Optimal_Sphere_Packing.pdf`: paper de referencia.
- `DemosaickingMultispectral3DSpherePacking/`: código MATLAB del paper (solo
  referencia; MATLAB no está instalado). `Dataset/`, `Test/` y `Test1/` vacíos.
- `Presentacion_Plan/plan_proyecto.typ` (+ PDF): **presentación actual**, 12
  diapositivas, tema `touying-simpl-swufe` con color ACCENT rgb(155,24,34).
  Compilar con `typst compile plan_proyecto.typ`.
- `LaTeX_template/proposal.tex` (+ PDF): contexto completo + propuesta en formato
  ICIP (6 páginas, la última solo referencias). Referencias en `refs.bib`.
  Compilar con `tectonic proposal.tex`. `Template.tex` es el original, sin tocar.

## Notas técnicas
- Typst/touying: no usar `#set text(...)` suelto después de un `==`, porque se
  acumula entre diapositivas. Envolver el cuerpo en `#[ ... ]` o en `#grid(...)`.
  `*negrita*` toma el color del tema; sobre fondo ACCENT usar
  `text(weight: "bold")`.
- LaTeX con spconf: en `\name`/`\address`, un `\\` seguido de `[` se interpreta
  como espaciado opcional; proteger con `{[...]}`.

## Pendiente
- Reemplazar `[Integrante 2]` en la presentación Typst y en el LaTeX.
- Descargar CAVE y registrarse en NTIRE 2022 para obtener ARAD-1K.
- Las presentaciones HTML publicadas antes en claude.ai quedaron desactualizadas
  (dicen 13 de noviembre e incluyen PPID en el núcleo); la vigente es la Typst.
