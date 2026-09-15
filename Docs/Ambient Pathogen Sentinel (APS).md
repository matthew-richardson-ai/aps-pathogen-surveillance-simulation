# Project Proposal Option 1: Ambient Pathogen Sentinel (APS)

## 1. Executive Summary
The Ambient Pathogen Sentinel is an environmental early-warning AI system that continuously analyzes genomic signals at high-frequency touchpoints (e.g., smart handwash drains, automated towel dispensers) within high-density facilities. The system detects high-risk viral/microbial mutations and backtracks contagion spread across a spatial facility graph to pinpoint "Patient Zero" or "Ground Zero" days before clinical symptoms emerge.

## 2. Motivation & Problem Statement
Current biosurveillance relies on symptomatic clinical reporting (incurring a 7–14 day lag) or broad municipal wastewater data that lacks localized spatial context. APS provides non-invasive, continuous environmental telemetry. It sequences exclusively viral and microbial genomes while discarding human DNA to ensure complete user privacy.

## 3. AI Methodology & System Architecture
The proposed system will be implemented as an automated Python pipeline:
* **Mutation & Virulence Classifier:** Ingests raw nucleotide reads and utilizes an open pre-trained genomic foundation model (such as DNABERT-2 or HyenaDNA) to predict whether novel point mutations alter receptor-binding affinity or increase immune-evasion risk.
* **Ground Zero Localization Engine:** Represents the physical facility (rooms, transit corridors, plumbing lines) as a directed spatio-temporal graph using `NetworkX` and `PyTorch Geometric`. An inverse-diffusion search algorithm backtracks timestamped sensor spikes to calculate the most probable origin node (the specific room/floor of Patient Zero).
* **Agentic Incident Responder:** A structured LLM pipeline (via LangGraph or DSPy with JSON schema output) that translates graph localization and pathogenicity scores into automated containment actions (e.g., targeted UV-C disinfection cycles, HVAC airflow modulation, and clinical PCR screening directives).

## 4. Evaluation Strategy
* **Localization Accuracy:** Test across 50 simulated outbreak scenarios across a synthetic facility graph, measuring whether the true origin node is correctly placed within the Top-1 and Top-3 candidate ranks.
* **Lead-Time Quantification:** Benchmark detection timestamp against simulated traditional symptomatic presentation thresholds.
* **Ablation Studies:** Document the comparative performance and failure modes of naive geometric distance heuristics versus learned graph diffusion to satisfy course requirements for reporting failed experiments.

## 5. Course Alignment & Commercial Potential
* **Course Fit:** Direct application of rational agent concepts (PEAS framework, reasoning under uncertainty, spatio-temporal environment modeling).
* **Market Viability:** High-value enterprise biosecurity for transit hubs (airports, cruise ships), university campuses, hospitals, and critical infrastructure.