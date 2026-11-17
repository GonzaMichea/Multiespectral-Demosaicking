// Póster de avance — Demosaicking multiespectral agnóstico al patrón
// Compilar desde la raíz del proyecto:
//   typst compile --root . Poster/poster.typ
// Los números y figuras se leen de results/, así que basta recompilar
// después de `python -m msdemosaic.evaluate_all` y `python -m msdemosaic.figures`.

#let PAPER = "a1" // confirmar formato con el profesor
#let ACCENT = rgb(155, 24, 34)
#let ACCENT-DARK = rgb(95, 12, 20)
#let BLUE = rgb(42, 85, 150)

#set page(paper: PAPER, margin: (x: 2.2cm, top: 2cm, bottom: 1.8cm))
#set text(font: ("Helvetica Neue", "Helvetica", "Arial"), size: 23pt, lang: "es")
#set par(justify: false, leading: 0.6em)
#set list(marker: text(fill: ACCENT)[■], indent: 0.2em, body-indent: 0.6em)

// === Datos de resultados ===
#let rows = csv("../results/summary.csv", row-type: dictionary)
#let hyp = json("../results/hypotheses.json")
#let has(m) = rows.any(r => r.method == m)
// La red sin máscara es opcional: su fila aparece solo cuando tiene resultados.
#let METHODS = (
  ("wb", [WB (clásico)]),
  ("conventional_imec", [U-Net sin máscara (IMEC)]),
  ("fixed_imec", [U-Net fija-IMEC]),
  ("fixed_osp", [U-Net fija-OSP]),
  ("agnostic", [*U-Net agnóstica*]),
).filter(((m, label)) => m != "conventional_imec" or has(m))
#let PATTERNS = ("imec", "osp", "btes", "random")
#let SEEN = (conventional_imec: "imec", fixed_imec: "imec", fixed_osp: "osp")
#let training = hyp.at("training", default: (:))
#let miles(n) = if n >= 1000 { str(calc.quo(n, 1000)) + " " + ("000" + str(calc.rem(n, 1000))).slice(-3) } else { str(n) }
#let fmt(x, digits: 2) = {
  let s = str(calc.round(float(x), digits: digits))
  if digits > 0 {
    // str() quita los ceros finales (36.4 en vez de 36.40): se reponen.
    let parts = s.split(".")
    let frac = if parts.len() > 1 { parts.at(1) } else { "" }
    s = parts.at(0) + "." + frac + "0" * (digits - frac.len())
  }
  s.replace(".", ",").replace("-", "−")
}
#let psnr(m, p) = float(rows.find(r => r.method == m and r.pattern == p).psnr_mean)
#let cell(m, p, metric: "psnr", digits: 2) = {
  let r = rows.find(r => r.method == m and r.pattern == p)
  if r == none { text(fill: luma(120), style: "italic")[en curso] } else {
    [#fmt(r.at(metric + "_mean"), digits: digits) #text(size: 0.75em, fill: luma(90))[± #fmt(r.at(metric + "_std"), digits: digits)]]
  }
}

// === Patrones MSFA ===
#let BAND = (
  "#5B3FD0", "#4F4FD8", "#3F62DC", "#2A75DB", "#1E88D0", "#1A99B8", "#1BA49B", "#2DAE7A",
  "#5BB55A", "#8FBC40", "#C2C036", "#E3B935", "#E8A030", "#E8822C", "#DD5E31", "#C93A3A",
).map(rgb)
#let msfa(f, n: 16, cell: 0.33cm) = grid(
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

// === Componentes ===
#let seccion(titulo, body) = block(width: 100%, below: 0.9em, {
  block(width: 100%, fill: ACCENT, inset: (x: 0.6em, y: 0.4em), radius: (top: 6pt), below: 0pt,
    text(fill: white, weight: "bold", size: 1.15em, titulo))
  block(width: 100%, stroke: 1.5pt + ACCENT, inset: 0.7em, radius: (bottom: 6pt), body)
})
#let hl(x) = text(fill: ACCENT, weight: "bold", x)
#let nodo(body, fill: white, stroke: ACCENT) = box(inset: 0.5em, radius: 5pt, fill: fill, stroke: 1.5pt + stroke, align(center, body))
#let flecha = text(size: 1.5em, fill: ACCENT)[→]

// === Encabezado ===
#block(width: 100%, fill: ACCENT-DARK, inset: (x: 1em, y: 0.9em), radius: 8pt, below: 0.9em)[
  #set text(fill: white)
  #text(size: 2.1em, weight: "bold")[Demosaicking multiespectral agnóstico al patrón]
  #v(-0.3em)
  #text(size: 1.15em)[¿Puede una sola red reconstruir con cualquier arreglo de filtros (MSFA), incluso uno que nunca vio?]
  #v(0.1em)
  #text(size: 0.9em)[Benjamín Cisternas · Gonzalo Michea #h(1fr) Fotografía Computacional 2026-2 · Universidad Técnica Federico Santa María]
]

#grid(columns: (1fr, 1fr), column-gutter: 1.6cm,
  // ---------------- Columna izquierda ----------------
  [
    #seccion[1 · Problema][
      - Una cámara MSFA mide *1 de 16 bandas* por píxel: cerca del 94 % del cubo espectral se estima con _demosaicking_.
      - Las redes neuronales superan a los métodos clásicos, pero aprenden el patrón de forma implícita: si cambia el MSFA hay que #hl[reentrenar].
      #block(fill: ACCENT.lighten(90%), stroke: (left: 4pt + ACCENT), inset: 0.6em, width: 100%)[
        #set text(size: 0.85em)
        _“A significant limitation of the proposed method is the necessity of retraining for specific MSFAs.”_ — Diaz et al., IEEE TIP 2023 [1], sec. IV-I
      ]
    ]

    #seccion[2 · Pregunta e hipótesis][
      #align(center, text(weight: "bold", fill: ACCENT-DARK)[¿Una sola red para cualquier MSFA, sin reentrenar?])
      #v(0.2em)
      - *H1*: una red de patrón fijo pierde calidad fuertemente en patrones no vistos.
      - *H2*: la red agnóstica queda a menos de 1 dB de cada red fija en el patrón de esa red.
      - *H3*: en patrones no vistos, la red agnóstica supera a las redes fijas.
    ]

    #seccion[3 · Método: la máscara como entrada][
      #align(center, grid(columns: 5, column-gutter: 0.5em, align: horizon,
        stack(spacing: 0.4em,
          nodo[Mosaico $bold(Y)$ \ #text(size: 0.7em)[$H times W$]],
          nodo(stroke: BLUE)[#text(fill: BLUE)[Máscara $bold(M)$] \ #text(size: 0.7em)[one-hot, 16 canales]],
        ),
        flecha,
        nodo(fill: ACCENT-DARK, stroke: ACCENT-DARK)[
          #text(fill: white, weight: "bold")[U-Net pequeña] \
          #text(fill: white, size: 0.7em)[1,07 M parámetros · pérdida $ell_1$]
        ],
        flecha,
        nodo[Cubo $hat(bold(X))$ \ #text(size: 0.7em)[$H times W times 16$]],
      ))
      #v(0.2em)
      #align(center, text(size: 0.9em)[$bold(Y) = sum_(k=1)^(16) bold(X)_k dot.o bold(C)_k$ #h(1.2em) $hat(bold(X)) = f_theta (bold(Y), bold(M))$])
      - *Entrada*: la red recibe las muestras separadas por banda ($bold(Y) dot.o bold(M)$), la máscara $bold(M)$ y una interpolación inicial WB [2], que refina.
      - *Entrenamiento agnóstico*: cada lote usa un patrón aleatorio distinto (cada banda aparece igual número de veces) → la red no puede memorizar un patrón, tiene que #hl[leer la máscara].
      - *Líneas base*: WB [2] y la misma U-Net, con la misma entrada, entrenada con un solo patrón (IMEC u OSP)#if has("conventional_imec") [; además, una U-Net convencional _sin máscara_ entrenada con IMEC].
    ]

    #seccion[4 · Patrones de prueba][
      #align(center, grid(columns: 4, column-gutter: 0.8em, row-gutter: 0.4em, align: center,
        msfa(imec), msfa(osp), msfa(btes), msfa(rnd),
        [IMEC [3]], [OSP [1]], [BTES [4]], [Aleatorio],
      ))
      #v(0.2em)
      #text(size: 0.9em)[Ninguno de los cuatro se usa para entrenar la red agnóstica.]
    ]

    #seccion[5 · Montaje experimental][
      #let n-test = hyp.num_test_scenes
      - *Datos*: #if hyp.data_mode == "cave-split" [CAVE [5], 32 escenas de 512 × 512: 21 para entrenar, 3 para validar y #n-test para probar (partición fija).] else [entrenamiento con 220 escenas de ARAD-1K (30 de validación); prueba con las #n-test escenas de CAVE [5].]
      - 16 de 31 bandas (400–700 nm), filtros ideales, sin ruido.
      - Parches de 64 × 64#if training.len() > 0 [, #miles(calc.max(..training.values().map(t => t.iters))) iteraciones por red], Adam, Mac M5 (MPS).
      - *Métricas*: PSNR, SSIM y SAM (media ± desv. sobre las escenas de prueba).
    ]

    #seccion[Referencias][
      #set text(size: 0.62em)
      #set par(spacing: 0.55em)
      [1] N. Diaz et al., “Multispectral filter array design by optimal sphere packing,” _IEEE TIP_, 2023.

      [2] J. Brauers y T. Aach, “A color filter array based multispectral camera,” _Workshop Farbbildverarbeitung_, 2006.

      [3] B. Geelen et al., “A compact snapshot multispectral imager with a monolithically integrated per-pixel filter mosaic,” _Proc. SPIE_, 2014.

      [4] L. Miao et al., “Binary tree-based generic demosaicking algorithm for multispectral filter arrays,” _IEEE TIP_, 2006.

      [5] F. Yasuma et al., “Generalized assorted pixel camera,” _IEEE TIP_, 2010. (CAVE)
    ]
  ],
  // ---------------- Columna derecha ----------------
  [
    #seccion[6 · Resultados a la fecha][
      #text(weight: "bold")[PSNR (dB) por método y patrón de prueba]
      #v(-0.2em)
      #{
        set text(size: 0.8em)
        table(
          columns: (1.5fr, 1fr, 1fr, 1fr, 1fr),
          align: (left + horizon, center + horizon, center + horizon, center + horizon, center + horizon),
          inset: 0.45em,
          stroke: 0.6pt + luma(170),
          fill: (x, y) => if y == 0 { ACCENT.lighten(85%) },
          table.header[*Método*][*IMEC*][*OSP*][*BTES*][*Aleatorio*],
          ..METHODS.map(((m, label)) => (
            label,
            ..PATTERNS.map(p => table.cell(
              fill: if SEEN.at(m, default: none) == p { BLUE.lighten(82%) },
              cell(m, p),
            )),
          )).flatten(),
        )
      }
      #v(-0.3em)
      #text(size: 0.75em)[Celda azul: patrón con que se entrenó esa red (visto); los demás son patrones no vistos.]

      #v(0.3em)
      #text(weight: "bold")[SAM (grados, menor es mejor)]
      #v(-0.2em)
      #{
        set text(size: 0.8em)
        table(
          columns: (1.5fr, 1fr, 1fr, 1fr, 1fr),
          align: (left + horizon, center + horizon, center + horizon, center + horizon, center + horizon),
          inset: 0.45em,
          stroke: 0.6pt + luma(170),
          fill: (x, y) => if y == 0 { ACCENT.lighten(85%) },
          table.header[*Método*][*IMEC*][*OSP*][*BTES*][*Aleatorio*],
          ..METHODS.map(((m, label)) => (label, ..PATTERNS.map(p => cell(m, p, metric: "sam")))).flatten(),
        )
      }

      #v(0.2em)
      #align(center, image("../results/figs/psnr_bars.png", width: 88%))

      #block(fill: ACCENT.lighten(90%), stroke: (left: 4pt + ACCENT), inset: 0.6em, width: 100%)[
        #set text(size: 0.9em)
        *Estado de las hipótesis*
        - *H1*: #if hyp.h1_drop_db == none [en curso.] else [la red fija-IMEC pierde #hl[#fmt(hyp.h1_drop_db) dB] en promedio al pasar a patrones no vistos#if hyp.h1_drop_db_osp != none [ (fija-OSP: #fmt(hyp.h1_drop_db_osp) dB)]#if hyp.at("h1_drop_db_conventional", default: none) != none [; la red sin máscara pierde #hl[#fmt(hyp.h1_drop_db_conventional) dB]].]
        - *H2*: #if hyp.h2_gap_db == none [en curso.] else [
            // "por poco": la brecha mayor queda a menos de 0,5 dB del umbral.
            diferencia fija − agnóstica: #hyp.h2_gap_db.pairs().map(((k, v)) => [#upper(k) #fmt(v) dB]).join[, ] → #if hyp.h2_holds [#hl[se cumple] (< 1 dB).] else if calc.max(..hyp.h2_gap_db.values()) < 1.5 [#hl[no se cumple por poco] (umbral: 1 dB).] else [#hl[no se cumple] con el umbral de 1 dB.]
            #if has("conventional_imec") and has("agnostic") {
              let d = psnr("conventional_imec", "imec") - psnr("agnostic", "imec")
              if d > 0 [Aun así, en IMEC la agnóstica queda en promedio a solo #fmt(d) dB de la U-Net convencional sin máscara.] else [En IMEC la agnóstica supera en promedio a la U-Net convencional sin máscara.]
            }
          ]
        - *H3*: #if hyp.h3_margin_db == none [en curso.] else {
            let vals = hyp.h3_margin_db.values()
            [ventaja de la agnóstica sobre las fijas en patrones no vistos: entre #fmt(calc.min(..vals)) y #fmt(calc.max(..vals)) dB → #if hyp.h3_holds [#hl[se cumple].] else [#hl[no se cumple] en todos los casos.]]
          }
      ]
    ]

    #seccion[7 · Inspección visual (patrón IMEC)][
      #align(center, image("../results/figs/visual_imec.png", width: 100%))
      #text(size: 0.8em)[Arriba: falso color (660/540/460 nm). Abajo: error absoluto medio por píxel.]
    ]

    #seccion[8 · Próximos pasos (entrega final: 25 nov)][
      - #if not (has("agnostic") and has("fixed_osp")) [Terminar el entrenamiento de las redes pendientes y cerrar la tabla H1–H3] else if hyp.h2_holds == false [Reducir la brecha de H2: probar la red agnóstica con más capacidad y más iteraciones] else [Confirmar H1–H3 con más semillas de entrenamiento].
      - Analizar errores por banda, zipper y falso color.
      - Informe en formato ICIP y repositorio con README.
      - Si hay tiempo: robustez al ruido.
    ]
  ],
)
