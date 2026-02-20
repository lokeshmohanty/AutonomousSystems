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

#set text(.8em)
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

    // --- Generative Modeling Goal ---
    #slide[
      #set text(.75em)
      === The Generative Modeling Problem

      *Goal:* Given samples ${x_0^((i))}_(i=1)^N$ from an unknown distribution $q(x_0)$, learn a model $p_theta(x)$ such that we can:
      - Evaluate $p_theta(x)$ (density estimation)
      - Draw new samples $x ~ p_theta(x)$ (generation)

      *Approach — Latent Variable Models:*
      Introduce latent variables $x_(1:T)$ and define a joint distribution:
      $
      p_theta(x_0) &= integral p_theta(x_(0:T)) dif x_(1:T)
      $
      This integral is intractable, so we use *variational inference*.

      *Evidence Lower Bound (ELBO):*
      $
      log p_theta(x_0) &>= EE_(q(x_(1:T)|x_0)) [log (p_theta(x_(0:T)))/(q(x_(1:T)|x_0))] =: cal(L)_"ELBO"
      $
      Maximizing $cal(L)_"ELBO"$ w.r.t. $theta$ is equivalent to minimizing $D_"KL"(q || p_theta)$.
    ]

    // --- Diffusion: The Big Picture ---
    #slide[
      #set text(.75em)
      === Diffusion — Defining $q$ and $p_theta$

      #align(center)[
        #diagram(
          node-stroke: .1em,
          spacing: 3em,
          node((0,0), $x_0$, radius: 1.2em, fill: teal.lighten(60%)),
          edge("->", bend: -30deg, label: $q(x_1|x_0)$, label-size: .55em),
          node((1,0), $x_1$, radius: 1.2em, fill: teal.lighten(70%)),
          edge("->", bend: -30deg, label: $q$, label-size: .55em),
          node((2,0), $dots.c$, stroke: none),
          edge("->", bend: -30deg, label: $q$, label-size: .55em),
          node((3,0), $x_T$, radius: 1.2em, fill: gray.lighten(40%)),

          edge((3,0), (2,0), "->", bend: -30deg, label: $p_theta$, label-size: .55em, stroke: teal),
          edge((2,0), (1,0), "->", bend: -30deg, label: $p_theta$, label-size: .55em, stroke: teal),
          edge((1,0), (0,0), "->", bend: -30deg, label: $p_theta(x_0|x_1)$, label-size: .55em, stroke: teal),
        )
        #v(.3em)
        #text(.6em)[Forward $q$: fixed, adds noise | #text(teal)[Reverse $p_theta$: learned, removes noise]]
      ]

      #grid(columns: (50%, 50%), gutter: 1em, [
        *Forward (fixed, no params):*
        $ q(x_(1:T)|x_0) &:= product_(t=1)^T q(x_t|x_(t-1)) $
        $ q(x_t|x_(t-1)) &:= cal(N)(x_t; sqrt(1-beta_t) x_(t-1), beta_t I) $
      ], [
        *Reverse (learned):*
        $ p_theta(x_(0:T)) &:= p(x_T) product_(t=1)^T p_theta(x_(t-1)|x_t) $
        $ p_theta(x_(t-1)|x_t) &:= cal(N)(x_(t-1); mu_theta(x_t, t), Sigma_theta(x_t, t)) $
      ])

      where $beta_t in (0,1)$ is a noise schedule and $p(x_T) = cal(N)(0, I)$.
    ]

    // --- Forward Process: Closed Form ---
    #slide[
      #set text(.75em)
      === Forward Process — Closed-Form Sampling

      Define $alpha_t := 1 - beta_t$ and $overline(alpha)_t := product_(s=1)^t alpha_s$. Derive by recursive substitution:
      $
      x_1 &= sqrt(alpha_1) x_0 + sqrt(1 - alpha_1) epsilon_1 \
      x_2 &= sqrt(alpha_2) x_1 + sqrt(1 - alpha_2) epsilon_2 \
           &= sqrt(alpha_2 alpha_1) x_0 + sqrt(1 - alpha_2 alpha_1) overline(epsilon)_2 quad #text(.7em)[(merging two Gaussians)] \
      &dots.v \
      x_t &= sqrt(overline(alpha)_t) x_0 + sqrt(1 - overline(alpha)_t) epsilon, quad epsilon ~ cal(N)(0, I)
      $
      $ q(x_t | x_0) &= cal(N)(x_t; sqrt(overline(alpha)_t) x_0, (1 - overline(alpha)_t) I) $

      *Key properties:*
      - As $t -> T$, $overline(alpha)_t -> 0$, so $q(x_T|x_0) approx cal(N)(0, I)$ — the data is destroyed
      - We can sample *any* $x_t$ directly from $x_0$ without iterating (crucial for training)

      *Posterior is tractable* when conditioned on $x_0$ (by Bayes' rule):
      $
      q(x_(t-1)|x_t, x_0) &= (q(x_t|x_(t-1), x_0) dot q(x_(t-1)|x_0))/(q(x_t|x_0)) \
      &prop underbrace(cal(N)(x_t; sqrt(alpha_t) x_(t-1), beta_t I), "one-step forward") dot underbrace(cal(N)(x_(t-1); sqrt(overline(alpha)_(t-1)) x_0, (1-overline(alpha)_(t-1)) I), "closed-form jump") \
      &= cal(N)(x_(t-1); tilde(mu)_t(x_t, x_0), tilde(beta)_t I) quad #text(.65em)[(completing the square)]
      $
      $
      tilde(mu)_t &= (sqrt(overline(alpha)_(t-1)) beta_t)/(1 - overline(alpha)_t) x_0 + (sqrt(alpha_t)(1-overline(alpha)_(t-1)))/(1 - overline(alpha)_t) x_t, quad tilde(beta)_t = (1-overline(alpha)_(t-1))/(1-overline(alpha)_t) beta_t
      $
    ]

    // --- ELBO Decomposition ---
    #slide[
      #set text(.75em)
      === Deriving the Training Objective

      Expanding the ELBO for diffusion @ho2020denoising:
      $
      cal(L)_"ELBO" &= underbrace(EE_q [log p_theta(x_0|x_1)], cal(L)_0 "reconstruction") - underbrace(D_"KL"(q(x_T|x_0) || p(x_T)), cal(L)_T approx 0) - sum_(t=2)^T underbrace(EE_q [D_"KL"(q(x_(t-1)|x_t,x_0) || p_theta(x_(t-1)|x_t))], cal(L)_(t-1) "denoising matching")
      $

      *Each $cal(L)_(t-1)$:* Match the learned reverse $p_theta(x_(t-1)|x_t)$ to the tractable posterior $q(x_(t-1)|x_t, x_0)$.

      Since both are Gaussians, the KL reduces to matching *means*:
      $
      cal(L)_(t-1) &= EE_q [ 1/(2sigma_t^2) || tilde(mu)_t(x_t, x_0) - mu_theta(x_t, t) ||^2 ] + C
      $

      *Reparametrize $mu_theta$* to predict $epsilon$ instead of $mu$ directly:
      $
      mu_theta(x_t, t) &= 1/sqrt(alpha_t) (x_t - (1-alpha_t)/sqrt(1-overline(alpha)_t) epsilon_theta(x_t, t))
      $
    ]

    // --- Simplified Objective ---
    #slide[
      #set text(.75em)
      === Simplified Objective & Training

      Substituting the reparametrization into $cal(L)_(t-1)$ and dropping constants:

      #block(stroke: 1.5pt + teal, inset: 14pt, radius: 5pt, fill: teal.lighten(95%))[
        $ cal(L)_"simple" &= EE_(t, x_0, epsilon) [ || epsilon - epsilon_theta(x_t, t) ||^2 ] $
        where $t ~ cal(U)({1,...,T})$, $epsilon ~ cal(N)(0,I)$, $x_t = sqrt(overline(alpha)_t) x_0 + sqrt(1-overline(alpha)_t) epsilon$
      ]

      #grid(columns: (55%, 45%), gutter: 1em, [
        *Training (per iteration):*
        + Sample $x_0 ~ q(x_0)$
        + Sample $t ~ cal(U)({1,...,T})$, $epsilon ~ cal(N)(0,I)$
        + Compute $x_t = sqrt(overline(alpha)_t) x_0 + sqrt(1-overline(alpha)_t) epsilon$
        + Gradient step on $nabla_theta || epsilon - epsilon_theta(x_t, t) ||^2$
      ], align(center)[
        #diagram(
          spacing: 1.5em,
          node((0,0), $x_t$, radius: 1em, fill: gray.lighten(40%)),
          node((0,2), $t$, radius: .5em, stroke: none),
          edge((0,0), (1,1), "->"),
          edge((0,2), (1,1), "->"),
          node((1,1), [$epsilon_theta$], shape: rect, fill: purple.lighten(80%), stroke: purple, inset: 1em),
          edge((1,1), (2,1), "->"),
          node((2,1), $hat(epsilon)$, stroke: none),
        )
        #v(.5em)
        #text(.65em)[Neural network predicts the noise $epsilon$ from $(x_t, t)$]
      ])
    ]

    // --- Sampling ---
    #slide[
      #set text(.75em)
      === DDPM Sampling (Inference)

      Starting from $x_T ~ cal(N)(0,I)$, iteratively apply:

      $
      x_(t-1) &= 1/sqrt(alpha_t) (x_t - (1-alpha_t)/sqrt(1-overline(alpha)_t) epsilon_theta(x_t, t)) + sqrt(beta_t) z, quad z ~ cal(N)(0,I)
      $

      #block(stroke: 1pt + teal, inset: 12pt, radius: 5pt)[
        *Algorithm:*
        + Sample $x_T ~ cal(N)(0,I)$
        + *For* $t = T, T-1, ..., 1$:
          - Sample $z ~ cal(N)(0,I)$ if $t > 1$, else $z = 0$
          - $x_(t-1) = 1/sqrt(alpha_t) (x_t - (1-alpha_t)/sqrt(1-overline(alpha)_t) epsilon_theta(x_t, t)) + sqrt(beta_t) z$
        + *Return* $x_0$
      ]

      *Problem:* Requires $T approx 1000$ sequential neural network evaluations — very slow.
    ]

    // --- DDIM ---
    #slide[
      #set text(.75em)
      === DDIM — Accelerated Sampling @song2021denoising

      *Key observation:* The ELBO only depends on $q(x_t|x_0)$, not the full joint $q(x_(1:T)|x_0)$.

      DDIM defines a *non-Markovian* forward process with the *same marginals*:
      $
      q_sigma(x_(t-1) | x_t, x_0) &= cal(N)(sqrt(overline(alpha)_(t-1)) x_0 + sqrt(1-overline(alpha)_(t-1)-sigma_t^2) dot (x_t - sqrt(overline(alpha)_t) x_0)/sqrt(1 - overline(alpha)_t), sigma_t^2 I)
      $

      #grid(columns: (50%, 50%), gutter: 1em, [
        *Consequence:*
        - Uses the *same* trained $epsilon_theta$ from DDPM
        - Can skip timesteps: use subsequence $tau_1, ..., tau_S$ with $S << T$
        - No retraining needed
      ], [
        *Stochasticity control via $sigma_t$:*
        - $sigma_t = sqrt((1-overline(alpha)_(t-1))/(1-overline(alpha)_t)) sqrt(beta_t)$: recovers DDPM
        - $sigma_t = 0$: fully *deterministic* (ODE)
      ])
    ]

    // --- DDIM Update Rule ---
    #slide[
      #set text(.75em)
      === DDIM — Update Rule

      #grid(columns: (55%, 45%), gutter: 1em, align(horizon)[
        $
        x_(t-1) &= underbrace(sqrt(overline(alpha)_(t-1)) dot (x_t - sqrt(1 - overline(alpha)_t) epsilon_theta(x_t, t))/sqrt(overline(alpha)_t), "predicted " x_0 " scaled to " t-1) \ &+
        underbrace(sqrt(1 - overline(alpha)_(t-1) - sigma_t^2) dot epsilon_theta(x_t, t), "direction pointing to " x_t) + underbrace(sigma_t epsilon_t, "noise")
        $
      ], align(center)[
        #diagram(
          spacing: 3em,
          node((1,0), $x_t$, radius: 1em, fill: gray.lighten(40%)),
          edge((1,0), (0,1), "->", label: [$hat(x)_0 = (x_t - sqrt(1-overline(alpha)_t) epsilon_theta) / sqrt(overline(alpha)_t)$], label-size: .45em, label-pos: 0.3),
          node((0,1), $hat(x)_0$, radius: 1em, stroke: (dash: "dashed"), fill: teal.lighten(80%)),
          edge((0,1), (2,1), "->", label: [reproject], label-size: .6em),
          node((2,1), $x_(t-1)$, radius: 1em, fill: gray.lighten(50%)),
          edge((1,0), (2,1), "-->", label: [+ direction + noise], label-size: .5em, bend: -20deg),
        )
      ])

      When $sigma_t = 0$: the mapping $x_T |-> x_0$ is *deterministic*, enabling latent interpolation.
    ]


  #slide[
    #set text(.8em)
    === DDPM vs DDIM — Summary

    #table(
      columns: (auto, 1fr, 1fr),
      inset: 10pt,
      align: horizon,
      table.header([], [*DDPM*], [*DDIM*]),
      [*Forward process*], [Markovian], [Non-Markovian],
      [*Marginals $q(x_t|x_0)$*], [Same], [Same],
      [*Inference steps*], [~1000], [~50 (20x faster)],
      [*Deterministic?*], [No], [Yes ($sigma_t=0$)],
      [*Latent interpolation*], [Hard], [Easy],
      [*Model reuse*], [—], [Uses DDPM's $epsilon_theta$],
    )
  ]

  // ==============================================
  // ====  Neural Operator
  // ==============================================

  == Neural Operator

  #bibliography("./works.bib")
