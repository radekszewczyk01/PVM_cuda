from disp import display_results, plot_data, save_plot

results = {'accuracy': 0.95, 'loss': 0.05}
display_results(results)

data = [1, 2, 3, 4, 5]
plot_data(data, title='Sample Data Plot')
save_plot('sample_plot.png')