import numpy as np
from sklearn.ensemble import IsolationForest
import joblib

# Generate synthetic training data (temperature, humidity, pressure)
X_train = np.random.normal(loc=[25, 57, 1010], scale=[5, 8, 8], size=(2000, 3))

# Train IsolationForest
model = IsolationForest(contamination=0.1, random_state=42)
model.fit(X_train)

# Save the model
joblib.dump(model, "models/iforest_model.pkl")

print("✅ IsolationForest model trained and saved as 'iforest_model.pkl'")
