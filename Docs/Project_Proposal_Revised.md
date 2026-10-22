# Adaptive Pathogen Surveillance (APS)
## CSC 4240/5240 Project Proposal

Team Members: Matthew Richardson, Emmanuel Zuniga, Justin Hortopanu, Jimmy Finnegan, Owen Brown

### Motivation and Problem

Environmental pathogen surveillance can provide early warning of an outbreak, but sensor readings may be noisy, missing, delayed, or misleading. A high reading also does not necessarily identify where an outbreak began.

Our project asks:

> Can AI use imperfect environmental sensor data to identify the likely source of a simulated outbreak, and can targeted additional sampling improve that prediction?

APS will address this problem entirely through simulation.

### Proposed Solution

APS will simulate a small facility with 6–8 connected zones and multiple environmental sensors. Each experiment will begin with a hidden outbreak source. The simulator will generate sensor readings affected by location, spread, noise, and missing data.

We will compare three approaches:

1. A simple baseline using the strongest or earliest sensor evidence.
2. A machine-learning model, such as Random Forest or Logistic Regression, that predicts the source from sensor readings.
3. An adaptive method that selects one additional sampling location when confidence is low and then updates its prediction.

Python, NetworkX, NumPy, pandas, scikit-learn, and visualization tools will support the system. A Streamlit interface may be used for the final demonstration.

### Evaluation

Because the true outbreak source is known by the simulator, predictions can be directly evaluated against ground truth.

We will measure:

- source-location accuracy;
- distance between predicted and actual source;
- performance under noisy or missing sensor data; and
- improvement from adaptive sampling compared with static sampling.

Multiple simulated outbreaks will be generated so results are based on repeated experiments rather than a single demonstration.

### Market Analysis

A future APS platform could support hospitals, universities, government facilities, military installations, transportation hubs, and other high-density environments.

The United States has approximately 6,100 hospitals. Using an illustrative software subscription of $25,000–$50,000 per facility per year, the theoretical U.S. hospital market would be approximately $152.5–$305 million annually. A 1% share would represent approximately $1.5–$3.0 million in annual recurring revenue.

Potential business models include per-facility subscriptions, enterprise contracts, analytics services, and integration with existing environmental monitoring systems.

### 9-Week Plan

| Week | Activity |
|---|---|
| 1 | Finalize scope, data format, metrics, and team responsibilities |
| 2 | Build facility, outbreak, and sensor simulation |
| 3 | Generate datasets and implement baseline localization |
| 4 | Implement and evaluate machine-learning models |
| 5 | Add adaptive sampling and complete end-to-end APS |
| 6 | Test noise, missing data, and failure conditions |
| 7 | Run final experiments and generate results |
| 8 | Complete demonstration interface, README, and report |
| 9 | Final testing, presentation preparation, and project cleanup |

The complete APS pipeline is scheduled to operate by the end of Week 5, leaving the remaining weeks for evaluation, refinement, documentation, and presentation.

### Scope

The class project is limited to simulation, sensor-data generation, source localization, adaptive sampling, and quantitative evaluation. Physical biosensors, wet-lab testing, clinical diagnosis, and production deployment are outside the project scope.