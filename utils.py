# src/utils.py
import pickle

def save_model(model, path: str):
    """Save the trained model as a pickle file."""
    with open(path, 'wb') as f:
        pickle.dump(model, f)
    print(f"✅ Model saved to {path}")

def load_model(path: str):
    """Load a model from a pickle file."""
    with open(path, 'rb') as f:
        model = pickle.load(f)
    print(f"📦 Model loaded from {path}")
    return model
