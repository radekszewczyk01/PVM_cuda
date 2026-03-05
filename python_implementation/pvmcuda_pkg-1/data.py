from pvmcuda_pkg.data import DataLoader, DataStorage

# Initialize DataLoader
data_loader = DataLoader()

# Load and preprocess data
data = data_loader.load_data('data/dataset.csv')
processed_data = data_loader.preprocess_data(data)

# Split data
train_data, val_data, test_data = data_loader.split_data(processed_data, 0.8, 0.1)

# Initialize DataStorage
data_storage = DataStorage()

# Save processed data
data_storage.save_data(train_data, 'data/train_data.pkl')