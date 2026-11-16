import numpy as np

from Jimmy.train_model import train_model
from Simulation.facility import build_facility
from Localization.baselines import strongest_sensor
from Evaluation.metrics import source_correct, source_distance

def evaluate_model(seed=42):
    model, X_test, y_test = train_model(seed)

    graph = build_facility()

    predictions = model.predict(X_test)

    correct = []
    distances = []

    for predicted, actual in zip(predictions, y_test):
        correct.append(source_correct(predicted, actual))
        distances.append(source_distance(graph, predicted, actual))

    accuracy = np.mean(correct)
    mean_distance = np.mean(distances)
    within_one_hop = np.mean(np.array(distances) <= 1)

    print("Random Forest Evaluation")
    print(f"Exact accuracy: {accuracy:.2%}")
    print(f"Mean graph-distance error: {mean_distance:.3f}")
    print(f"Within-one-hop accuracy: {within_one_hop:.2%}")

if __name__ == "__main__":
    evaluate_model()