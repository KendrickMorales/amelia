from src.data import load_data
from src.model import BruiseDetector


# Load the data
train_data, test_data = load_data()


# Load Model
model = BruiseDetector()
# Train Model
model.fit(train_data)

# Save Model
model.save()

# Evaluate The model
model.evaluate()