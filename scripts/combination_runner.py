import os
import subprocess

# Layer names to edit
layer_names = ["k_proj", "q_proj", "v_proj", "out_proj", "fc_in", "fc_up", "fc_out"]

# Layer numbers (0 to 39)
layer_numbers = list(range(40))

# Reduction percentages
reduction_percentages = [10, 25, 40, 50, 60, 75, 90, 92.5, 95, 97.5, 98, 98.5, 99, 99.5, 99.75]

# Convert percentages to rates: rate = 10 - (percentage / 10)
rates = [(p / 10) for p in reduction_percentages]

# Iterate over all combinations of lname, lnum, and rate
for lname in layer_names:
    for lnum in layer_numbers:
        for rate in rates:
            rate_str = f"{rate:.10f}".rstrip('0').rstrip('.')

            # Construct the save folder name
            folder_name = f"{lname}-{lnum}-{rate_str}"
            os.makedirs(folder_name, exist_ok=True)

            # Call the script for the current combination
            cmd = [
                "python3",
                "intervention_granite.py",
                "--lname", lname,
                "--lnum", str(lnum),
                "--rate", rate_str
            ]
            print(f"Running command: {' '.join(cmd)}")
            subprocess.run(cmd)
            print(f"Completed: {lname}-{lnum}-{rate_str}")