from datasets import Dataset, DatasetManager

# Create a dataset instance
dataset = Dataset('path/to/dataset')

# Load data
data_samples = dataset.load_data()

# Access a specific sample
sample = dataset.get_sample(index=0)

# Manage multiple datasets
manager = DatasetManager()
manager.add_dataset(dataset)