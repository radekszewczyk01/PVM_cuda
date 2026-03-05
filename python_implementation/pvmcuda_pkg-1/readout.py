from pvmcuda_pkg.readout import read_data, preprocess_data

raw_data = read_data('data/input_file.csv')
processed_data = preprocess_data(raw_data)