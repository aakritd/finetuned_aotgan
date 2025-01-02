import pandas as pd
import matplotlib.pyplot as plt

# 1. Load both CSV files
file1 = r"E:\AOT-GAN-for-Inpainting-finetune\logs_csv\train_generatorloss(1).csv"  # Replace with the path to the first file
file2 = r"E:\AOT-GAN-for-Inpainting-finetune\logs_csv\train_generatorloss(2).csv"  # Replace with the path to the second file

data1 = pd.read_csv(file1)
data2 = pd.read_csv(file2)

# 2. Merge the two DataFrmes by concatenation
combined_data = pd.concat([data1, data2], ignore_index=True)

# 3. Sort by 'Step' to ensure continuity
combined_data = combined_data.sort_values(by='Step').reset_index(drop=True)

# 4. Extract 'Step' and 'Value' columns
steps = combined_data['Step'] 
values = combined_data['Value']

# 5. Plot the combined data
plt.figure(figsize=(10, 6))
plt.plot(steps, values, linestyle='-', color='b', label='Loss Value')

# 6. Customize the plot
plt.title("Training Loss of Generator ( 0.01 * Ladv + 1 * L1 + 250 * Lsty + 0.1 * Lper)")
plt.xlabel("Step")
plt.ylabel("Loss Value")
plt.legend()
plt.grid(True)

# 7. Save the plot
output_path = r"E:\AOT-GAN-for-Inpainting-finetune\logs_csv\plots\train_generator.png"  # Replace with your desired path
plt.savefig(output_path, format='png', dpi=300, bbox_inches='tight')
print(f"Plot saved successfully to: {output_path}")

# 7. Show the plot
plt.show()

