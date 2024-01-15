import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os


# Define the file path and name
file_name = 'test_memory_consumption_model_results_8x8_epochs.xlsx'
file_path = os.path.join(r'C:\ANNs\Results', file_name)

# Read the Excel file
df = pd.read_excel(file_path, decimal=',')


# Convert 'Accuracy' column to numeric (remove commas and convert to float)
#df['Accuracy'] = pd.to_numeric(df['Accuracy'].str.replace(',', '.'))

# Loop through unique combinations of 'Model' and 'Quantized'
for model, quantized in df.groupby(['Model', 'Quantized']):
    model_id = model[0]
    quantized_flag = 'Quantized' if model[1].any() else 'Not Quantized'

    # Create a pivot table
    heatmap_data = quantized.pivot_table(values='Accuracy', index='Epochs', columns='Sparsity')
     # Sort the index in descending order
    heatmap_data = heatmap_data.sort_index(ascending=False)

    # Create a heatmap using Seaborn
    plt.figure(figsize=(10, 8))
    sns.heatmap(heatmap_data, annot=True, cmap='viridis', fmt='.3f', linewidths=.5)
    plt.title(f'Model {model_id} - {quantized_flag} - Accuracy Heatmap')
    plt.show()
