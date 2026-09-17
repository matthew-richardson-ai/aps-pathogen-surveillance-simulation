# Adaptive Pathogen Surveillance (APS)

## AI-Driven Simulation for Environmental Outbreak Detection and Source Localization

### Team Members

Matthew Richardson  
Emmanuel Zuniga  
Justin Hortopanu  
Jimmy Finnegan  
Owen Brown  

### Motivation

Environmental pathogen surveillance can provide early indicators of infectious disease activity, but detection alone does not necessarily determine where an outbreak originated, how genomic changes should be interpreted, or where additional sampling would be most useful. Our project proposes **Adaptive Pathogen Surveillance (APS)**, a simulation-based AI system that combines genomic analysis, spatial reasoning, and adaptive sampling to investigate these problems. APS will be entirely software-based and will not involve physical biosensors, real biological samples, or clinical diagnosis.

### Project Description

APS will simulate a facility containing connected rooms, environmental sensors, and a spreading pathogen. The simulator will preserve a hidden ground-truth outbreak origin while generating imperfect observations including positive detections, false positives, false negatives, and delayed readings.

Existing AI and data-analysis tools will be integrated to analyze the simulated outbreak. For example, genomic sequence models such as DNABERT-2 may be explored for genomic anomaly analysis using complete reference genome data, including the SARS-CoV-2 Wuhan-Hu-1 reference genome (NCBI RefSeq NC_045512.2) in FASTA format. Graph-based methods using tools such as NetworkX and machine-learning techniques will be used to estimate the most probable outbreak source from facility structure, sensor locations, and detection patterns.

The project's primary innovation is an adaptive surveillance strategy. Instead of relying only on fixed observations, APS will evaluate uncertainty and determine where an additional simulated sample could provide the greatest information value. New observations will update the outbreak hypothesis, creating a dynamic feedback loop between detection, inference, and additional sampling. An AI-assisted response component will then convert the results into clear simulated recommendations.

### Plan of Execution and Evaluation

The project will consist of five connected areas: simulation and sensor modeling, genomic analysis, graph-based outbreak localization, response generation, and system integration/evaluation. Existing AI libraries and models will be evaluated and selected based on their suitability for each task.

Performance will be tested across multiple simulated outbreaks with known ground truth. We will compare a static surveillance approach against the proposed adaptive approach using metrics such as source-localization accuracy, graph-distance error, mutation-detection correctness, and the number of observations required to reach a confident prediction.

### Brief Market Analysis

If developed beyond the classroom simulation, APS could be positioned as a facility-level outbreak-intelligence platform for hospitals, universities, transportation hubs, government facilities, and other high-traffic environments. Hospitals provide a particularly relevant initial market; the American Hospital Association reports approximately 6,100 hospitals in the United States. A potential commercial model would use per-facility annual software subscriptions for surveillance analytics, outbreak localization, alerts, and adaptive sampling recommendations, with additional enterprise or government contracts for larger deployments. Existing national wastewater-surveillance programs demonstrate that environmental infectious-disease monitoring already has practical public-health value, while APS explores how AI could make such surveillance more localized, adaptive, and actionable.