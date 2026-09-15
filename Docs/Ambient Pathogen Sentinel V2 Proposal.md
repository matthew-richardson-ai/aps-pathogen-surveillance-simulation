# Project Overview & Proposal: Ambient Pathogen Sentinel (APS)

---

## 1. Executive Summary (The Plain-English Concept)

### What Are We Building?
Think of this project as **"Smart Smoke Detectors for Pathogens."**

Traditional public health tracking is slow because it is reactive. When a new virus or bacterial strain emerges, officials only find out after people become symptomatic, visit a clinic, take a PCR test, and wait for lab results—incurring a **7 to 14 day delay**. 

Instead of waiting for sick patients to visit clinics, high-traffic communal touchpoints (e.g., automated handwash sinks, greywater traps, towel dispensers) can passively capture biological traces. Our project builds the **AI intelligence layer** that processes these continuous biosensor streams to:
1. Detect whether an observed pathogen carries dangerous new mutations.
2. Backtrack the contagion through the building's physical layout to pinpoint **Ground Zero** (the room or facility where "Patient Zero" introduced it).
3. Automatically generate containment advisories (HVAC airflow changes, localized sanitation directives).

> **Important Note for the Team:** We are **not** working with wet-lab biological samples or physical hardware. The entire project is implemented in **Python** using open-source AI models, network graph libraries, and synthetic outbreak simulations.

---

## 2. Recommended Video Watch List

To build intuition before diving into the code, check out these concepts:

* **Wastewater & Environmental Biosurveillance:**
  * [SciShow: How Sewage Predicts Disease Outbreaks](https://www.youtube.com/watch?v=r32s_2m-Nrg) — Demonstrates how environmental sampling detects viral spikes 1–2 weeks before hospital admissions.
* **Biological Foundation Models & Neural Networks:**
  * [3Blue1Brown: But what is a Neural Network?](https://www.youtube.com/watch?v=aircAruvnKk) — The foundational visual primer for how weights, activations, and neural network layers process feature streams.
* **Network Graphs & Contagion Dynamics:**
  * [3Blue1Brown: Simulating an Epidemic](https://www.youtube.com/watch?v=gxAaO2rsdIs) — Visualizes SIR/SEIR agent transmission, network hubs, and quarantine interventions across dynamic graphs.

---

## 3. System Architecture & The 3 AI Stages

```
   [Touchpoint Sensor Percepts]
   (Continuous FASTA/k-mer Reads)
                 │
                 ▼
 ┌───────────────────────────────┐
 │ AI Tool 1: Genomic LM         │  --> Scores genetic drift, virulence, and
 │ (DNABERT-2 / HyenaDNA)        │      receptor-binding risk from mutations
 └───────────────┬───────────────┘
                 │
                 ▼
 ┌───────────────────────────────┐
 │ AI Tool 2: Spatial Graph      │  --> Traces contagion backward along facility
 │ (PyTorch Geometric / NetworkX)│      connections to locate Patient Zero
 └───────────────┬───────────────┘
                 │
                 ▼
 ┌───────────────────────────────┐
 │ AI Tool 3: Incident Agent     │  --> Generates structured containment plans
 │ (Structured LLM Pipeline)     │      (HVAC adjustment, UV-C sanitation alerts)
 └───────────────────────────────┘
```

1. **AI Tool 1 — Mutation & Virulence Scoring:**  
   Treats nucleotide sequences (A, C, G, T) like tokens in a language model. An open pre-trained genomic model reads sequence snippets to determine whether observed variations represent baseline drift or high-risk functional mutations (such as increased receptor-binding affinity or immune evasion).
2. **AI Tool 2 — Spatial Back-Tracing (Ground Zero Localization):**  
   Models a facility (e.g., campus building, transit terminal) as a directed graph where nodes are rooms/sensors and edges are hallways, doors, and plumbing lines. When positive readings flag across multiple points over time, an inverse-diffusion search algorithm calculates the most probable origin node.
3. **AI Tool 3 — Automated Action Agent:**  
   A constrained language model (using JSON output schemas) parses the graph output and drafts immediate operational directives (e.g., adjust HVAC airflow, deploy targeted UV-C disinfection cycles, issue localized health advisories).

---

## 4. Suggested Division of Labor (4 Team Members)

* **Member 1 (Simulation & Environment Design):**
  * Builds the synthetic facility graph using `NetworkX`.
  * Creates the agent-based outbreak simulation (SEIR model) that generates timestamped sensor events.
* **Member 2 (Genomic AI Pipeline):**
  * Loads and configures an open pre-trained genomic model from Hugging Face (`DNABERT-2` or `HyenaDNA`).
  * Implements the script that takes sequence reads and outputs pathogenicity/mutation risk scores.
* **Member 3 (Source Inversion Algorithm):**
  * Implements the graph-traversal / inverse-diffusion algorithm in `PyTorch Geometric` or `NetworkX`.
  * Back-traces infection paths to rank the top candidate nodes for "Patient Zero."
* **Member 4 (LLM Agent, Evaluation & Benchmarking):**
  * Builds the structured LLM responder (using Pydantic / JSON mode via Ollama or an open API).
  * Measures accuracy metrics (Top-1 and Top-3 localization rates) and formats final report visualizations.

---

## 5. Course Alignment & Evaluation Plan

* **Use of Existing Tools:** Combines open-source genomic transformers, graph frameworks, and LLM APIs into a unified Python notebook workflow.
* **Measurable Quantitative Evaluation:**
  * Runs 50 Monte Carlo outbreak scenarios across the graph.
  * Measures the **Localization Success Rate** (how often the true seed node is ranked in the top 3 candidates).
  * Benchmarks **Lead-Time Advantage** (how many hours/days earlier detection occurs compared to clinical symptomatic reporting).
* **Failed Experiments Documentation:** Compares simple distance heuristics against graph-based inverse diffusion to document why naive approaches fail (directly fulfilling the rubric).
* **Commercial & Ethical Bonus:** Includes a market justification for transit hubs, hospitals, and enterprise facilities, alongside an analysis of genetic privacy preservation (discarding human DNA to focus strictly on microbial/viral reads).