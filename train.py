from src.data import load_dataset
from src.model import BruiseDetector


# Load the data
train_data, test_data = load_dataset(batch_size=1)


# Load Model
model = BruiseDetector()
# Train Model
model.fit(train_data, batch_size=1, epochs=1)

# Save Model
# model.save()

# Evaluate The model
# model.evaluate()
