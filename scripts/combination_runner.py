import os
import subprocess

# Layer names to edit
layer_names = ["v_proj", "fc_in", "fc_out"]

# Layer numbers (0 to 39)
layer_numbers = list(range(29,40,5))

# Reduction percentages
reduction_percentages = [10, 50, 99]

# Convert percentages to rates: rate = 10 - (percentage / 10)
rates = [(p / 10) for p in reduction_percentages]

# Iterate over all combinations of lname, lnum, and rate
for lname in layer_names:
    for lnum in layer_numbers:
        for rate in rates:
            rate_str = f"{rate:.10f}".rstrip('0').rstrip('.')

            # Call the script for the current combination
            cmd = [
                "python3",
                "src/intervention_granite.py",
                "--lname", lname,
                "--lnum", str(lnum),
                "--rate", rate_str
            ]
            print(f"Running command: {' '.join(cmd)}")
            subprocess.run(cmd)
            print(f"Completed: {lname}-{lnum}-{rate_str}")