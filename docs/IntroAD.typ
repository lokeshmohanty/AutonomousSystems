#set text(font: "Nunito", weight: 600)
#show figure.caption: set text(.5em)
#show table.cell: set text(size: .5em)

#import "@preview/touying:0.6.1": *         // Slides
#import "@preview/tiaoma:0.3.0": qrcode,ean // QRCode
#import "@preview/showybox:2.0.4": showybox // Boxes
#import "@preview/fletcher:0.5.8" as fletcher: diagram, node, edge
#import "@preview/cetz:0.4.2" as cetz: draw
#import "@preview/cetz-plot:0.1.3": smartart
// #import "@preview/neural-netz:0.3.0"

#let title = [Planning in E2E Autonomous Driving Systems]
#let author = [Lokesh Mohanty]
#let institute = [Indian Institute of Science]

#import themes.simple: *
#show: simple-theme.with(aspect-ratio: "16-9", footer:[#title])
#title-slide[
  = #title
  #v(2em)

  #author #footnote[#institute] #h(1em)

  #datetime.today().display("[day] [month repr:short], [year]")
]

// #import "my-theme.typ": *
// #show: my-theme.with(
//   aspect-ratio: "16-9",
//   footer: self => self.info.institution,
//   config-info(
//     title: [Planning in E2E Autonomous Driving],
//     // subtitle: [Subtitle],
//     author: [Lokesh Mohanty],
//     date: datetime.today(),
//     institution: [Indian Institute of Science],
//   ),
// )
// #title-slide()

// #outline()

= Introduction

== Levels of Driving Automation
#figure(
  image("./fig/ad-levels.jpg", width: 75%),
  caption: [Levels of Driving Automation @dongWhyAutonomousVehicles]
)

// Lateral dynamics: Steering, Longitudinal dynamics: Acceleration

== Why is automation in driving required?

- Reducing road accidents
- Decreasing traffic congestion
- Minimizing fuel consumption
- Reducing air pollution

#v(1fr)

#rect([
  #set text(.55em)
  - There is a vast amount data available for the autonomous driving scenario along with highly competitive research.
  - This can lead to high quality models which can be fine-tuned for other applications like *Robotics* and *UAVs*#footnote[Unmanned Aerial Vehicles]
], inset: 10pt, radius: 5pt, fill: orange.lighten(70%))
 

== Current Scenario of Autonomous Driving

- To date very few models of private cars equipped with automation level 3 are legally allowed to circulate with limitations on location and speed @dongWhyAutonomousVehicles
- While no private vehicle is licensed to roam public roads in "full" or "semi" autonomous mode (i.e., automation level 4 and 5) @dongWhyAutonomousVehicles

#figure(
  image("./fig/ad-pipeline.jpg"),
  caption: [Schematic overview of ADS perception pipeline @dongWhyAutonomousVehicles]
)

== Challenges / Problems

#slide[
- *Perception*: 
  - sensor limitations, 
  - depth esitmation, 
  - object detection, 
  - sun glare, 
  - adverse weather, 
  - data size, 
  - real-time localization and mapping
][
- *Path Planning*: 
  - Real-time decision making, 
  - handling uncertainity and failures, 
  - safety vs efficiency, 
  - behaviour prediction, 
  - automated negotiation
]

#slide[
- *Common-sense Reasoning*: 
  - Grounding problem, 
  - Cornercase scenarious, 
  - accuracy and safety requirements, 
  - vulnerable road users, 
  - Narrow vs General AI
][
- *Road Infrastructure*: 
  - Inconsistent road signs, 
  - unpredictable road conditions
- *Ethics*: 
  - Trolley Problem, 
  - Accountability
]
- *End-users*: 
  - Situational awareness, 
  - explainable automation, 
  - data privacy
- *Others*:
  - Policy making
  - ...

== Why E2E (end-to-end) models?

#{[
  #set text(.9em)
- Traditional systems use a sequential pipeline of task specific modules 
  leading to 
  - *information loss* due to data abstraction at each module
  - *cascading errors* as downstream tasks cannot correct upstream mistakes

#align(center)[
  #let steps = ([Perception], [Tracking], [Prediction], [Planning], [Control])
  #let colors = (red, orange, green).map(c => c.lighten(40%))

  #cetz.canvas({
    smartart.process.basic(
      steps, step-style: colors,
      equal-height: true,
      dir: ltr, name: "chart",
    )
  })
]

- E2E models primarily address these fundamental structural limitations of 
  traditional pipelines along with *unified objective* and *reduced engineering complexity*
]}

= End-to-End Models

== Problem Formulation

#{[
#set text(.7em)
#grid(
  columns: (41%, 31%, 30%), gutter: 8pt,  
  rows: (auto, auto), row-gutter: 25pt,
  grid.cell([
    $a_t = H(F(x|theta))$

    #let steps = ([Inputs (x)], [Vision Model (F)], [Action Head (H)], [Output ($a_t$)])
    #let colors = (red, orange, green).map(c => c.lighten(40%))

    #cetz.canvas({
      smartart.process.basic(
        steps, step-style: colors,
        equal-height: true,
        dir: ltr, name: "chart",
      )
    })
  ], colspan: 3, align: center), [ 
    #showybox(title: "Inputs (x)", [
      - Sensor observations
      - Latent scene representations
      - Language instructions
      - Proprioceptive states
    ], frame: (title-color: red.mix(black).lighten(20%)))
  ], [
    #showybox(title: "Backbone (F)", [
      - VLM/VM as backbone
    ], frame: (title-color: yellow.mix(black).lighten(20%)))
    #v(-15pt)
    #showybox(title: "Action Head (H)", [
      - Predicts next action
    ], frame: (title-color: green.mix(black).lighten(20%)))
  ], [
    #showybox(title: [Outputs ($a_t$)], [
      - Trajectory (discrete/continuous)
      - Direct Control
      - Language
    ], frame: (title-color: red.mix(black).lighten(20%)))
  ]
)
]}

== Inputs (x)

- Sensor Inputs
  - Visual Images: $x_"img" in RR^(N_C times H times W times 3)$
  - LiDAR Point Clouds : $x_"lidar" in RR^(N_p times D)$
- Latent Representations
  - Bird's-Eye View (BEV) Features: $x_"bev" in RR^(C times H_"bev" times W_"bev")$
  - Occupancy Grids: $x_"occ" in RR^(C_"occ" times X times Y times Z)$
- Language Inputs: $x_"lang" in RR^(T times D_"emb")$
- Ego-Vehicle State Information: $x_"state" in RR^(D_"state")$
== VLM/VM as Backbone (F)

- VLM/VM for direct action generation (Single System)
- VLM for guidance generation (Dual System) (slow thinking + fast execution)

== Action Heads (H)

It converts the latent representation from VLM to action outputs
- *Language Head*: generates free-form textual commands which are then passed to a parser for final actions
- *Regression Head*: directly predicts continuous numerical values like steering angles, throttle and brake values
- *Trajectory Selection*: Evaluates a set of candidate trajectories and selects the most optimal one
- *Trajectory Generation*: Generates trajectory through probabilistic modelling of actions

== Outputs ($a_t$)

#columns(2)[
#set text(.8em)
- Discrete Trajectory Representations#footnote[$(x_i, y_i) -> "coordinates"$, $phi -> "prediction horizon"$]

  $ a_t = {(x_i, y_i)}^phi_(i=1), " where" (x_i, y_i) in RR^2$

- Continuous Trajectory Representations#footnote[
    $v(t) -> "speed profile"$, $k(t) -> "curvature profile"$, $T -> "time horizon"$
  ]

  $ a_t = (v(t), k(t)), "    for" t in [0, T] $

#colbreak()

- Direct Control Representations#footnote[
    $delta_t -> "steering angle"$, $tau_t -> "throttle input"$, $beta_t -> "brake input"$
  ]
  $ a_t = (delta_t, tau_t, beta_t) $

- Language Representations#footnote[$w_i -> i^"th" "token in the models vocabulary" cal(V)$]

  $ a_t = {w_1, w_2, ..., w_T} $
]

== Vision-Action Models

#grid(columns: (auto, auto), gutter: 10pt, [
  *End-to-End Models*
  #set text(0.7em)

  Directly predicts the control actions or the 
  trajectories from sensor inputs _(action as output)_

  - Action-Only Models
  - Perception-Action Models

], [
  *World Models*
  #set text(0.7em)

  Explicitly models action-conditioned future dynamics 
  to support policy learning and decision making 
  _(action as input)_

  - Image-Based World Models#footnote[outputs: raw pixels]
  - Occupancy-Based World Models#footnote[outputs: free space geometry]
  - Latent-Based World Models

  #{[
    #set text(.8em)
    _(Generaly Diffusion-Based or Autoregressive Models)_
  ]}
])

== Vision-Action Models
#grid(columns: (auto, auto), gutter: 8pt, align(bottom)[
  #figure(
    image("./fig/va-e2e.png", width: 100%),
    caption: [End-to-End Models],
  )
], [
  #figure(
    image("./fig/va-world.png", width: 93%),
    caption: [World Models],
  )
])

== Why aren't Vision-Action Models sufficient?

#grid(
  columns: (45%, 55%), [ 
    *Vision-Action (VA) Models*
    - Limited interpretability
    - Fragile generalization
    - Directly map perception to low-level actions 
  ], [
    *Vision-Language-Action (VLA) Models*
    - VLM backbone with action-prediction head
    - Instruction following
  ]
)

== Vision-Language-Action Models

#grid(columns: (40%, auto), gutter: 10pt, [
  *End-to-End VLA*
  #set text(0.8em)

  Single model that directly maps 
  the inputs to actions

  - Textual Action Generator
    - Meta Actions
    - Trajectory Waypoints
  - Numerical Action Generator
    - Additional Action Head
    - Additional Action Tokens

], [
  *Dual-System VLA*
  #set text(0.8em)

  Dual model, a slow planner with a fast executor

  - Explicit Action Guidance
    - Meta Action Guidance
    - Waypoint Supervision
  - Implicit Representations Transfer
    - Knowledge Distillation
    - Multimodal Feature Fusion
])

== Vision-Language-Action Models
#grid(columns: (auto, auto), gutter: 8pt, align(bottom)[
  #figure(
    image("./fig/vla-e2e.png"),
    caption: [End-to-End VLA],
  )
], [
  #figure(
    image("./fig/vla-dual.png"),
    caption: [Dual-System VLA],
  )
])

== VLA: Challenges

- Real-Time Processing and Latency
- Lack of Domain-Specific Foundational Models
- Generalizing to Rare and Novel Scenarios
- High-Quality Data Cost
- Interpretability and Hallucination
- Long-Horizon Temporal Coherence

== Future Directions

- Unified Vision-Lanugage World Models
- Richer Multimodal Fusion
- Socially Aware, Knowledge-Grounded Driving
- Continual & Onboard Learning
- Standardized Evaluation & Safety Guarantees
- Human-Centric Interaction & Personalization

= Experiments

== Datasets

Most widely used datasets
#block(
  table(
    columns: 2,
    align: horizon,
    stroke: (x: none),
    inset: 15pt,
    [Open-Loop dataset, uses trajectory-based metrics], [nuScenes @caesar2020nuscenes], 
    [Open-Loop dataset, uses 3D Gaussian Splatting for pseudo-simulation of closed-loop dataset], [NAVSIM @dauner2024navsim],
    [Closed-Loop dataset, built on CARLA @Dosovitskiy2017carla (simulator)], [Bench2Drive @jia2024bench2drive],
    [Most recent dataset introducing long-tail, safety-critical 
      scenes with human-preference annotations ], [WOD-E2E @xu2025wod]
  ),
  width: 80%,
)

= Evaluation Metrics

== Action-Planning Open-Loop Evaluation  

#table(
  columns: 3,
  [L2 ↓], [L2 Error], [L2 distance error between the planned trajectory and the human  driving trajectory in 3 seconds],
  [CR ↓], [Collision Rate], [How often the self-driving vehicle would collide with other agents  on the road],
  [ADE ↓], [Average  Displacement Error], [Mean displacement error between predicted trajectories and expert waypoints across the horizon, reflecting overall trajectory accuracy],
  [FDE ↓], [Final Displacement Error], [Displacement error at the final predicted waypoint compared  with expert trajectories, emphasizing long-term accuracy],
  [MR ↓], [Miss Rate], [Fraction of prediction time steps where displacement error exceeds horizon-specific thresholds, reflecting failure in trajectory coverage],
  [AHE ↓], [Average Heading Error], [Mean absolute angular deviation between predicted and expert heading over the trajectory horizon, measuring orientation accuracy],
  [FHE ↓], [Final Heading Error], [Absolute angular deviation of predicted heading from expert at  the final timestep, reflecting terminal orientation accuracy],
  [SLE ↓], [Speed L1 Error], [Mean absolute error of predicted speed control signals],
  [SALE ↓], [Steer Angle L1  Error], [Mean absolute error of predicted steering angle control signals],
)

== Trajectory-Based Closed-Loop Evaluation  

#table(
  columns: 3,
  [RC ↑], [Route Completion], [The percentage of route distance completed],
  [DS ↑], [Driving Score], [RC weighted by a penalty factor that accounts for collisions with  pedestrians, vehicles, etc],
  [NC ↑], [No Collision], [Fraction of scenarios without ego-fault collisions, focusing exclusively on responsibility-aware collision evaluation],
  [DAC ↑], [Driving  Admissibility Check], [Boolean evaluation that checks whether the ego vehicle remains  inside drivable polygons throughout the rollout],
  [TTC ↑], [Time To Collision], [Boolean verification that the time-to-collision value exceeds safety  thresholds, preventing imminent crashes],
  [C ↑], [Driving Comfort], [The comfort of driving],
  [EP ↑], [Ego Progress], [Penalization of excessive jerk, acceleration, or yaw-rate, reflecting  ride quality and passenger comfort],
  [PDMS ↑], [Predictive Driver Model Score], [A flexible weighted evaluation score in autonomous driving that aggregates multiple safety, progress, and comfort subscores into a single metric],
  [SR ↑], [Success Rate], [Percentage of navigation episodes that successfully reach the goal  within a fixed time budget, indicating overall task completion],
  [ID ↑], [Infraction Distance], [Average driving distance between two infractions, with longer  distances reflecting safer and more reliable policy behavior],
)

== Text-Based Action Evaluation

#table(
  columns: 3,
  [CIDEr ↑],  [Consensus-based Image Description Evaluation], [Measures similarity of generated captions to multiple human  references using TF-IDF weighted n-grams],
  [BLEU ↑],  [Bilingual Evaluation Understudy], [Precision-based metric that compares n-grams of the generated  text against reference texts],
  [METEOR ↑],  [Metric for Evaluation of Translation with Explicit Ordering], [Considers unigram precision and recall with stemming, synonym  matching, and fragmentation penalty],
  [Rouge ↑],  [Recall-Oriented Understudy for Gisting Evaluation], [Recall-focused metric using overlapping n-grams, word sequences,  or word pairs between generated and reference texts],
  [Top-1  Acc ↑], [Visual Question Answering Top-1 Accuracy], [Percentage of predictions where the most confident output  matches the ground truth label],
)

// = Handling Causal Confusion / Grounding Problem
//
// == What is causal confusion?
//
// #slide[
//   - It increases with increase in the size of data
// ][
//   #figure(
//     image("./fig/causal_confusion.pdf"),
//     caption: [Causal Confusion @chenEndtoEndAutonomousDriving2024],
//   ) <causal_confusion>
// ]
//
// == Baseline: ORION
// - Title: ORION: A Holistic E2E Autonomous Driving Framework by Vision-Language Instructed Action Generation
// - Authors: Haoyu Fu, Diankun Zhang, Zongchuang Zhao, ...
// - Affiliation: Huazhong University of Science and Technology, Xiaomi EV
//
// VLMs provide common sense which is helpful for causal reasoning
//
// VLMs focus on the semantic reasoning space, but we need planning results in the action space

= References
== Important Papers
+ Why Autonomous Vehiches Are Not Ready Yet @dongWhyAutonomousVehicles
+ Vision-Language-Action Models for Autonomous Driving @huVisionLanguageActionModelsAutonomous2026

== Others

- Awesome VLA Models: #link("https://github.com/worldbench/awesome-vla-for-ad")
- Github Repository: #link("https://github.com/airexlab/AutonomousSystems")

#figure(
  qrcode("https://github.com/airexlab/AutonomousSystems", options: (scale: 3.0)),
  caption: [Our Github Repository: #link("https://github.com/airexlab/AutonomousSystems")]
)


#pagebreak()
#bibliography("works.bib")
