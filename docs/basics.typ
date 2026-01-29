#set text(font: "Nunito", weight: 600)
#show figure.caption: set text(.5em)
#show table.cell: set text(size: .5em)

#import "@preview/touying:0.6.1": *
#import "@preview/fletcher:0.5.8" as fletcher: diagram, node, edge
#import "@preview/cetz:0.4.2" as cetz: draw
// #import "@preview/neural-netz:0.3.0"

#let diagram = touying-reducer.with(
  reduce: fletcher.diagram, cover: fletcher.hide
)

#let title = [Pre-requisites for Autonomous Systems]
#let author = [Lokesh Mohanty]
#let institute = [AiREX Lab\ Indian Institute of Science]

#import themes.simple: *
#show: simple-theme.with(aspect-ratio: "16-9", footer:[#title])
#title-slide[
  = #title
  #v(2em)

  #author

  #{[
    #set text(.8em)
    #institute

    #datetime.today().display("[day] [month repr:short], [year]")
  ]}
]

#outline()

// =====================================
// ==== Transformer
// =====================================

== Transformer
- This architecture #footnote[[Vaswani et. al.] Attention is all you need @Vaswani2017Attention] was introduced for Seq-to-Seq tasks
- Excels in learning spatial and temporal relations
- Higly parallelizable leading to fast execution

#align(center)[
  #set text(.7em)
  #diagram(
    node-shape: rect,
    node-fill: orange.lighten(40%),
    node-corner-radius: 5pt,
    node-inset: 10pt,
    node-outset: 5pt,
    edge-stroke: 1.8pt,
    spacing: (20pt, 20pt),

    node((0,0), [Input\ sequence(spatial/temporal) of tokens]),
    edge("-|>"),
    node((0,+1.8), [Attention Module\ enriches information of each token with all pair-wise relations], fill: red.lighten(40%)),
  )
]

// =====================================
// ==== Attention Mechanism
// =====================================

#slide(repeat: 5, self => [
  #let (uncover, only, alternatives) = utils.methods(self)
  #set text(.65em)
  #grid(columns: (45%, 60%), [
    - Three vectors (*query($q_i$)*, *key($k_i$)*, *value($v_i$)*) are generated from each input token($x_i$)
    - Each token *queries* ($q$) how much of *attention* it should give to every other token

    #uncover("3-")[
      - In *Self-Attention*, query and key, value belong to the same sequence
      - In *Cross-Attention*, query belongs to a different sequence
    ]
    #uncover("5")[
      - *Attention* layers are computationally less expensive than 
        recurrent and convolutional layers
    ]
  ], align(center)[
    #show math.equation: it => {
      show ",": it => [, #h(.5em)]
      it
    }
    #uncover("2-")[
      $ 
      X -> "input", n -> "number of tokens", d -> "token dimension" \
      x_i -> i^"th" "token" in RR^d, W^{Q, K, V} in RR^(d times d) \
      q_i -> x_i W^Q, k_i -> x_i W^K, v_i -> x_i W^V  \

      "probs"(p) <- "Softmax"("Attention Scores" <- (q_i K^T) / sqrt(d)) \
      #strong("Attention") (a_i) <- p V \
      $
    ]

    #uncover("4-")[
      #block(
        [$
        X in RR^(n times d), Q, K, V in RR^(n times d) \
        Q <- X W^Q, K <- X W^K, V <- X W^V \
        #strong("Attention") <- "Softmax"((Q K^T) / sqrt(d)) V

        $], stroke: 1pt, inset: 15pt, radius: 5pt)
      ]
    ])
  ])

  // ==============================================
  // ====  Overall Architecture (Encoder-Decoder)
  // ==============================================

  #slide(repeat: 6, self => [
    #let (uncover, only, alternatives) = utils.methods(self)
    #grid(columns: (60%, 40%), column-gutter: 8pt, [
      #set text(.85em)
      #uncover("3-")[
        - In the *attention module* the attention computed is added
          to the input to incorporate contextual information
      ]
      #uncover("4-")[
        - *Normalization* and *Dropout* layers are added for regularization
      ] 
      #uncover("5-")[
        - *Feed Forward Networks* add non-linearity to the model
      ]
      #uncover("6-")[
        - Multiple *Attention* + *FFN* layers in sequence form 
          the backbone of the transformer
      ]
    ], [
      #set text(purple.lighten(40%))
      #diagram(
        node((5em, 5em), image("./fig/base-transformer-arch.png"), height: 12em),
        only("1", node((0em, 14em), [Encoder])),
        only("1", edge("->")),
        only("1", node((.5em, 1em), width: 4.2em, height: 8em, 
          stroke: 2pt + purple.lighten(40%))),
        only("2", node((-1em, 14em), [Decoder])),
        only("2", edge("->")),
        only("2", node((9.5em, 5em), width: 4.2em, height: 12em, 
          stroke: 2pt + purple.lighten(40%))),
      )
    ])
  ])

  // ==============================================
  // ====  Multi-Head, Masking
  // ==============================================

  #slide[
    #grid(columns: (40%, 30%, 30%), [
      #set text(.8em)
      - *Multi-Head* allows each head to attend to different feature
      - *Masking* allows hiding information from the model
      - It allows the model to process variable sized input
      - It also works as a regularizer
    ], [
      #figure(
        image("./fig/base-transformer-heads.png", height: 60%),
        caption: [Multi-Head]
      )
    ], [
      #figure(
        image("./fig/base-transformer-attn.png", height: 60%),
        caption: [Scaled Dot-Product Attention]
      )
    ])
  ]

  // ==============================================
  // ====  Positional Encoding
  // ==============================================

  #slide[
    #set text(.7em)
    #grid(
      columns: (60%, 40%), rows: (auto, auto), 
      column-gutter: 10pt, row-gutter: 25pt, [
        *Absolute Positional Encoding* @Vaswani2017Attention (add to the embedding)
        $
        "PE"_("pos", i) = cases(
          sin("pos" / (1000^(i/d_"model"))) "if" i "is even",
          cos("pos" / (1000^(i/d_"model"))) "if" i "is odd"
        ) \
        "pos" <- [0, "seq_length"), i <- [0, d_"model"/2)
        $
      ], align(horizon)[
        *Rotary Positional Encoding* @su2023RoPE (rotate the embedding)
        $
        theta_i = 10000^(-(2i)/d_"model"), i in [1, d_"model"/2]
        $

      ], grid.cell([
        - Effect of positional embedding decreases from the first dimension to the last dimension
        - Early dimensions preserve the position information and higher dimensions preseve the token information
        - This helps the model to choose which information to attend more
      ], colspan: 2))
    ]

    // ==============================================
    // ====  QFormer
    // ==============================================

    == QFormer

    // ==============================================
    // ====  Diffusion
    // ==============================================

    == Diffusion

    // ==============================================
    // ====  Neural Operator
    // ==============================================

    == Neural Operator

    #bibliography("./works.bib")
