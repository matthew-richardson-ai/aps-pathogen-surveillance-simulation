from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from Jimmy.generate_training_data import generate_dataset

def train_model(seed=42):
    X, y = generate_dataset(n_samples=10000, seed=seed)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=seed,
        stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=seed,
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    return model, X_test, y_test

if __name__ == "__main__":
    model, X_test, y_test = train_model()

    print("Model trained successfully!")
    print("Number of trees:", model.n_estimators)
    print("Test examples:", len(X_test))
    print("First prediction:", model.predict(X_test[:1])[0])
    print("Actual source:", y_test[0])