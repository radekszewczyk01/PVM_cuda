from data_carla import load_carla_data, preprocess_data

data = load_carla_data('path/to/carla_data.json')
processed_data = preprocess_data(data)