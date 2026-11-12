#import "@preview/touying-simpl-swufe:0.1.0": *

// === Tema de color ===
#let ACCENT = rgb(155, 24, 34)
#let ACCENT-DARK = rgb(95, 12, 20)
#let BLUE = rgb(42, 85, 150) // tareas de la persona A en el cronograma

#show: swufe-theme.with(
  aspect-ratio: "16-9",
  lang: "es",
  config-info(
    title: [Demosaicking multiespectral agnóstico al patrón],
    subtitle: [Plan de proyecto y alcance — 6 semanas (5 oct – 11 nov 2026)],
    short-title: [Demosaicking agnóstico al patrón],
    authors: ([Benjamín Cisternas], [[Integrante 2]]),
    author: [B. Cisternas y [Integrante 2]],
    institution: ([Fotografía Computacional — Universidad Técnica Federico Santa María],),
    date: datetime.today(),
    banner: none,
  ),
  config-colors(
    primary: ACCENT,
    primary-dark: ACCENT-DARK,
  ),
  config-common(new-section-slide-fn: none),
)

// Corrige la alineación vertical del marcador del tema
#set list(marker: move(dy: -0.55em, components.knob-marker(primary: ACCENT)))

#let hl(x) = text(fill: ACCENT, weight: "bold", x)
#let cell-hl = ACCENT.lighten(85%)

// === Patrones MSFA (16 bandas, violeta → rojo) ===
#let BAND = (
  "#5B3FD0", "#4F4FD8", "#3F62DC", "#2A75DB", "#1E88D0", "#1A99B8", "#1BA49B", "#2DAE7A",
  "#5BB55A", "#8FBC40", "#C2C036", "#E3B935", "#E8A030", "#E8822C", "#DD5E31", "#C93A3A",
).map(rgb)

// Dibuja un recorte n×n de un MSFA; f(i, j) entrega la banda (1..16) del píxel.
#let msfa(f, n: 16, cell: 0.17cm) = grid(
  columns: (cell,) * n,
  rows: (cell,) * n,
  ..range(n * n).map(k => {
    let i = calc.quo(k, n)
    let j = calc.rem(k, n)
    rect(width: cell, height: cell, fill: BAND.at(f(i, j) - 1), stroke: none)
  }),
)

#let tiled(kernel) = (i, j) => kernel.at(calc.rem(i, 4)).at(calc.rem(j, 4))
#let imec = tiled(((7, 8, 6, 5), (15, 16, 14, 13), (11, 12, 10, 9), (3, 4, 2, 1)))
#let btes = tiled(((1, 13, 4, 15), (9, 5, 11, 7), (3, 16, 2, 14), (12, 8, 10, 6)))
#let rnd = tiled(((11, 3, 14, 6), (1, 9, 16, 4), (13, 7, 2, 12), (5, 15, 8, 10)))
#let osp = (i, j) => calc.rem(7 * (i + 1) + 4 * (j + 1), 16) + 1

// Caja para diagramas
#let nodo(body, fill: white, stroke: ACCENT) = box(
  inset: 0.55em,
  radius: 5pt,
  fill: fill,
  stroke: 1pt + stroke,
  align(center, body),
)
#let flecha = text(size: 1.6em, fill: ACCENT)[→]

#title-slide()

#speaker-note[Plan de proyecto para la dirección elegida: una red de demosaicking que funcione con cualquier patrón de filtros. Tenemos 6 semanas, del 5 de octubre al miércoles 11 de noviembre, con unas 6 horas por semana cada uno.]

= Problema

== El problema: las redes quedan atadas a su patrón

#grid(columns: (1.3fr, 1fr), column-gutter: 1em, align: horizon)[
  - Una cámara MSFA mide *1 de 16 bandas* por píxel → cerca del 94 % del cubo se estima con _demosaicking_.
  - Las *redes neuronales* superan a los métodos clásicos en fidelidad espacial y espectral [1].
  - Pero aprenden el patrón de forma *implícita*: si cambia el MSFA, hay que #hl[reentrenar] (≈ 1,5 días en 2 GPU por patrón en [1]).
][
  #align(center)[
    #msfa(imec, n: 8, cell: 0.3cm)
    #text(size: 0.6em)[MSFA IMEC 4 × 4 (16 bandas) [10]]
  ]
  #v(0.3em)
  #tblock(title: [Diaz et al., 2023 [1], sec. IV-I])[
    #set text(size: 0.8em)
    _“A significant limitation of the proposed method is the necessity of retraining for specific MSFAs.”_
  ]
]

#speaker-note[~1 min. Falta casi todo el cubo; las redes dan los mejores resultados pero quedan atadas al patrón con que se entrenaron. Los propios autores lo reconocen como su principal limitación.]

== Pregunta de investigación e hipótesis

#[
#align(center, block(
  fill: ACCENT.lighten(90%),
  stroke: (left: 4pt + ACCENT),
  inset: 0.8em,
  width: 92%,
  text(size: 1.1em, weight: "bold", fill: ACCENT-DARK)[
    ¿Puede una sola red reconstruir con cualquier MSFA, incluso uno que nunca vio, sin reentrenar?
  ],
))

#v(0.5em)
#grid(
  columns: (1fr, 1fr, 1fr),
  column-gutter: 0.8em,
  tblock(title: [H1])[
    #set text(size: 0.8em)
    Una red con *patrón fijo* pierde calidad fuertemente en patrones no vistos.
  ],
  tblock(title: [H2])[
    #set text(size: 0.8em)
    La red agnóstica queda a *menos de 1 dB* de cada red fija en el patrón de esa red.
  ],
  tblock(title: [H3])[
    #set text(size: 0.8em)
    En patrones no vistos, la red agnóstica *supera claramente* a las redes fijas.
  ],
)

#v(0.4em)
#text(size: 0.75em)[Cualquier resultado es informativo: si H2 o H3 fallan, medimos y explicamos cuánto depende la red del patrón.]
]

#speaker-note[~1 min. Una pregunta y tres hipótesis que se verifican con una sola tabla. El umbral de 1 dB es una meta, no un resultado.]

= Propuesta

== Método: la máscara del patrón como entrada

#[
#align(center, grid(
  columns: 5,
  column-gutter: 0.7em,
  align: horizon,
  stack(
    spacing: 0.5em,
    nodo[Mosaico $bold(Y)$ \ #text(size: 0.65em)[$H times W$]],
    nodo(stroke: BLUE)[#text(fill: BLUE)[Máscara $bold(M)$] \ #text(size: 0.65em)[one-hot, $H times W times 16$]],
  ),
  flecha,
  nodo(fill: ACCENT-DARK, stroke: ACCENT-DARK)[
    #text(fill: white, weight: "bold")[U-Net pequeña [5]] \
    #text(fill: white, size: 0.65em)[< 2 M parámetros · pérdida $ell_1$]
  ],
  flecha,
  nodo[Cubo $hat(bold(X))$ \ #text(size: 0.65em)[$H times W times 16$]],
))

#v(0.2em)
#align(center, text(size: 0.75em)[
  Captura simulada: $bold(Y) = sum_(k=1)^L bold(X)_k dot.o bold(C)_k + bold(N)$ #h(1.5em) Red: $hat(bold(X)) = f_theta (bold(Y), bold(M))$
])

#v(0.2em)
#set text(size: 0.85em)
- *Entrenamiento*: en cada lote se sortea un patrón aleatorio válido (cada banda aparece igual número de veces).
- *Por qué funciona*: la red no puede memorizar un patrón → tiene que #hl[leer la máscara].
- *Línea base*: la misma U-Net sin máscara, entrenada con un solo patrón (IMEC u OSP).
]

#speaker-note[~1 min. Además del mosaico, la red recibe la máscara que dice qué banda midió cada píxel. Para que la use de verdad, cada lote trae un patrón distinto.]

== Experimento: patrones vistos y no vistos

#grid(columns: (1.5fr, 1fr), column-gutter: 1em, align: horizon)[
  #set text(size: 0.75em)
  #table(
    columns: (auto, 1fr, 1fr, 1fr, 1fr),
    align: (left, center, center, center, center),
    inset: 0.45em,
    stroke: 0.5pt + luma(180),
    fill: (_, y) => if y == 4 { cell-hl },
    table.header[*Método*][*IMEC*][*OSP*][*BTES*][*Aleatorio*],
    [WB [4]], [ref.], [ref.], [ref.], [ref.],
    [U-Net fija-IMEC], hl[visto], [no visto], [no visto], [no visto],
    [U-Net fija-OSP], [no visto], hl[visto], [no visto], [no visto],
    [*U-Net agnóstica*], [*no visto*], [*no visto*], [*no visto*], [*no visto*],
  )
  #v(0.3em)
  Cada celda: PSNR, SSIM y SAM (media ± desv.) en las 32 escenas de CAVE.

  La red agnóstica se entrena *solo con patrones aleatorios*: ninguno de los 4 de prueba.
][
  #set text(size: 0.6em)
  #grid(
    columns: 2,
    column-gutter: 0.6em,
    row-gutter: 0.5em,
    align: center,
    [#msfa(imec) IMEC [10]], [#msfa(osp) OSP [1]],
    [#msfa(btes) BTES [11]], [#msfa(rnd) Aleatorio (ej.)],
  )
]

#speaker-note[~1,5 min. Todo el proyecto apunta a llenar esta tabla. Las redes fijas deberían funcionar solo en su patrón; la agnóstica nunca vio ninguno de los cuatro.]

== Recursos: datos, métricas, cómputo y código

#[
#set text(size: 0.8em)
#grid(
  columns: (1fr, 1fr),
  column-gutter: 0.8em,
  row-gutter: 0.6em,
  tblock(title: [Datos])[
    - *Entrenamiento*: 200–300 escenas de ARAD-1K [7]; 10–15 % para validación.
    - *Prueba*: 32 escenas de CAVE [6]; 16 de 31 bandas.
  ],
  tblock(title: [Métricas])[
    - PSNR y SSIM [8]: fidelidad espacial.
    - SAM [9]: fidelidad espectral.
    - PSNR por banda, firmas espectrales e inspección visual.
  ],
  tblock(title: [Cómputo])[
    - Mac M5 16 GB (PyTorch + MPS): desarrollo.
    - Kaggle (GPU gratis): entrenamientos largos.
  ],
  tblock(title: [Código])[
    - Propio en Python: simulador, métodos, redes, evaluación.
    - Código MATLAB de [2] solo como referencia.
  ],
)
#v(0.3em)
Solo #hl[3 redes] (fija-IMEC, fija-OSP y agnóstica) con parches de 64 × 64: cabe sin GPU propia.
]

#speaker-note[~1 min. Entrenamos y probamos con conjuntos distintos; CAVE como prueba para comparar con el paper.]

= Alcance

== Alcance en tres niveles

#[
#set text(size: 0.75em)
#grid(
  columns: (1fr, 1fr, 1fr),
  column-gutter: 0.7em,
  tblock(title: [A · Mínimo — semana 3])[
    - Simulador con 4 patrones
    - WB en CAVE
    - U-Net fija-IMEC
    - Medición de H1
    *Resultado:* cuánto cae una red fija en patrones no vistos.
  ],
  block(stroke: 2pt + ACCENT, radius: 7pt, tblock(title: [B · Objetivo — semana 5])[
    - Todo lo del nivel A
    - U-Net agnóstica
    - U-Net fija-OSP
    - Tabla completa (H1–H3) y figuras
    *Resultado:* una red para cualquier patrón.
  ]),
  tblock(title: [C · Extra — si sobra tiempo])[
    - PPID [3] como baseline
    - Robustez al ruido
    - Filtros IMEC reales
    - Entradas $[bold(Y), bold(M)]$ vs. $bold(Y) dot.o bold(M)$
    *Resultado:* evaluación más realista.
  ],
)
#v(0.4em)
#align(center)[Cada nivel incluye el anterior. #hl[A está garantizado.] B es el objetivo; C solo si vamos adelantados. \
#text(size: 0.85em)[Dedicación: 6 h/semana por persona ≈ 65 h en total · nivel B estimado en ≈ 45–50 h.]]
]

#speaker-note[~1 min. A se cierra en la semana 3 y ya responde H1. B se cierra en la semana 5, dejando la 6 para escribir.]

== Qué hacemos y qué queda fuera

#[
#set text(size: 0.9em)
#grid(
  columns: (1fr, 1fr),
  column-gutter: 0.8em,
  tblock(title: [Dentro del alcance])[
    - Datos *simulados* a partir de cubos reales (CAVE, ARAD-1K)
    - *16 bandas* y filtros ideales
    - Una *U-Net pequeña* y 3 redes entrenadas
    - *4 patrones* de prueba: IMEC, OSP, BTES y aleatorio
  ],
  tblock(title: [Fuera del alcance])[
    - Captura con *cámara real* o hardware propio
    - Reproducir *TRevSCI* o el paper completo
    - *Diseñar u optimizar* patrones nuevos
    - MSFA de *25 bandas* o más
  ],
)
#v(0.5em)
#align(center, text(size: 0.85em)[Lo que queda fuera puede mencionarse como trabajo futuro en el informe.])
]

#speaker-note[~45 s. Dejamos explícito lo que no haremos, para que el alcance sea realista en 6 semanas.]

= Plan

== Cronograma (nivel B)

#let bar(who, span: 1) = table.cell(
  colspan: span,
  fill: if who == "A" { BLUE } else if who == "B" { ACCENT } else { ACCENT-DARK },
  align(center, text(fill: white, weight: "bold", who)),
)
#let gap(n) = if n > 0 { (table.cell(colspan: n)[],) } else { () }
#let row(task, start, span, who) = (
  (task,) + gap(start - 1) + (bar(who, span: span),) + gap(6 - (start - 1) - span)
)

#[
#set text(size: 0.68em)
#table(
  columns: (2.3fr, 1fr, 1fr, 1fr, 1fr, 1fr, 1fr),
  align: (left + horizon, center, center, center, center, center, center),
  inset: 0.42em,
  stroke: 0.5pt + luma(200),
  table.header(
    [*Tarea*],
    [*S1* \ 5–11 oct], [*S2* \ 12–18 oct], [*S3* \ 19–25 oct],
    [*S4* \ 26 oct–1 nov], [*S5* \ 2–8 nov], [*S6* \ 9–11 nov],
  ),
  ..row([Datos, simulador y métricas], 1, 1, "A"),
  ..row([Entorno PyTorch y U-Net base], 1, 1, "B"),
  ..row([Baseline clásico (WB)], 2, 1, "A"),
  ..row([U-Net fija-IMEC], 2, 1, "B"),
  ..row([Evaluación en 4 patrones (H1)], 3, 1, "A"),
  ..row([U-Net agnóstica], 3, 2, "B"),
  ..row([U-Net fija-OSP], 4, 1, "A"),
  ..row([Tabla completa y artefactos], 4, 2, "A + B"),
  ..row([Informe y presentación], 5, 2, "A + B"),
)
#v(0.2em)
#text(fill: BLUE, weight: "bold")[A] = datos y evaluación · #hl[B] = redes · Interfaz común: `simulate` → `method` → `evaluate`
]

#speaker-note[~1 min. Dos roles en paralelo. Los experimentos terminan en la semana 5; la semana 6 queda para escribir.]

== Hitos y punto de decisión

#let hito(fecha, titulo, cuerpo, destacado: false) = block(
  width: 100%,
  height: 10.5em,
  inset: 0.7em,
  radius: 6pt,
  fill: if destacado { ACCENT } else { ACCENT.lighten(92%) },
  stroke: (top: 4pt + if destacado { ACCENT-DARK } else { ACCENT }),
  {
    set align(top + left)
    set text(fill: if destacado { white } else { black })
    set par(spacing: 0.6em)
    text(size: 0.75em, weight: "bold", fecha)
    parbreak()
    text(size: 0.95em, weight: "bold", titulo)
    parbreak()
    text(size: 0.75em, cuerpo)
  },
)

#[
#grid(
  columns: (1fr, 1fr, 1fr, 1fr),
  column-gutter: 0.7em,
  hito([S2 · vie 16 oct], [Pipeline funcionando], [Patrón, captura, reconstrucción y métricas, con WB y la red fija.]),
  hito(destacado: true, [S3 · vie 23 oct], [Punto de decisión], [¿La red fija-IMEC funciona y H1 está medida? \ #text(weight: "bold")[Sí:] seguimos al nivel B. \ #text(weight: "bold")[No:] cerramos en el nivel A.]),
  hito([S5 · vie 6 nov], [Experimentos cerrados], [Tabla completa. Desde aquí no se agregan experimentos nuevos.]),
  hito([S6 · mié 11 nov], [Entrega final], [Informe (formato ICIP), presentación y repositorio con README.]),
)

#v(0.4em)
#align(center, text(size: 0.75em)[Entrega oficial: miércoles 11 de noviembre · dedicación: 6 h/semana por persona.])
]

#speaker-note[~45 s. El hito clave es el 23 de octubre: si la red fija no funciona, cerramos en el nivel A y usamos el tiempo restante en hacerlo bien.]

== Riesgos y plan B

#[
#set text(size: 0.8em)
#table(
  columns: (1fr, 1.6fr),
  inset: 0.5em,
  stroke: none,
  fill: (_, y) => if calc.odd(y) { ACCENT.lighten(92%) },
  table.header[*Riesgo*][*Plan B*],
  [El entrenamiento es lento en el Mac], [Entrenar en Kaggle; usar menos escenas o parches de 48 × 48.],
  [ARAD-1K no está disponible a tiempo], [Solo CAVE con validación cruzada: 24 escenas para entrenar y 8 para probar.],
  [La red agnóstica no generaliza], [Se reporta y se analiza: cuánto y por qué depende la red del patrón.],
  [Poco tiempo: 6 h/semana por persona], [Los entrenamientos largos corren solos en Kaggle; PPID queda en el nivel C.],
  [Atraso general], [Punto de decisión del 23 de octubre: cerrar en el nivel A.],
)
]

#speaker-note[~45 s. Para cada riesgo hay una alternativa que no cambia la pregunta del proyecto.]

== Referencias

#[
#set text(size: 0.55em)
#set par(spacing: 0.75em)
[1] N. Diaz, A. Alvarado, P. Meza, F. Guzmán y E. Vera, “Multispectral filter array design by optimal sphere packing,” _IEEE Trans. Image Process._, vol. 32, pp. 3634–3649, 2023.

[2] Código de [1]: #link("https://github.com/nelson10/DemosaickingMultispectral3DSpherePacking")[github.com/nelson10/DemosaickingMultispectral3DSpherePacking]

[3] S. Mihoubi, O. Losson, B. Mathon y L. Macaire, “Multispectral demosaicing using pseudo-panchromatic image,” _IEEE Trans. Comput. Imaging_, vol. 3, no. 4, 2017.

[4] J. Brauers y T. Aach, “A color filter array based multispectral camera,” _Workshop Farbbildverarbeitung_, 2006.

[5] O. Ronneberger, P. Fischer y T. Brox, “U-Net: Convolutional networks for biomedical image segmentation,” _MICCAI_, 2015.

[6] F. Yasuma, T. Mitsunaga, D. Iso y S. K. Nayar, “Generalized assorted pixel camera,” _IEEE Trans. Image Process._, vol. 19, no. 9, 2010. (CAVE)

[7] B. Arad et al., “NTIRE 2022 spectral recovery challenge and data set,” _CVPR Workshops_, 2022. (ARAD-1K)

[8] Z. Wang, A. C. Bovik, H. R. Sheikh y E. P. Simoncelli, “Image quality assessment: From error visibility to structural similarity,” _IEEE Trans. Image Process._, vol. 13, no. 4, 2004. (SSIM)

[9] F. A. Kruse et al., “The spectral image processing system (SIPS),” _Remote Sens. Environ._, vol. 44, 1993. (SAM)

[10] B. Geelen, N. Tack y A. Lambrechts, “A compact snapshot multispectral imager with a monolithically integrated per-pixel filter mosaic,” _Proc. SPIE_, vol. 8974, 2014. (IMEC)

[11] L. Miao, H. Qi, R. Ramanath y W. E. Snyder, “Binary tree-based generic demosaicking algorithm for multispectral filter arrays,” _IEEE Trans. Image Process._, vol. 15, no. 11, 2006. (BTES)
]
