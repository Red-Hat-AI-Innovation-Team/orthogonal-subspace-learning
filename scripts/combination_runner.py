import os
import subprocess

# Layer names to edit
layer_names = ["k_proj", "q_proj", "v_proj", "out_proj", "fc_in", "fc_up", "fc_out"]

# Layer numbers (0 to 39)
layer_numbers = list(range(40))

# Reduction percentages
reduction_percentages = [10, 50, 90, 99, 99.75]

# Convert percentages to rates: rate = 10 - (percentage / 10)
rates = [(p / 10) for p in reduction_percentages]

excluded_combinations = {
    "fc_in-29-1.0", "fc_in-34-1.0", "fc_in-39-1.0", "fc_out-29-1.0", "fc_out-34-1.0", "fc_out-39-1.0",
    "fc_in-28-9.9", "v_proj-29-5.0", "v_proj-34-5.0", "v_proj-39-5.0",
    "fc_in-29-5.0", "fc_in-34-5.0", "fc_in-39-5.0", "fc_out-29-5.0", "fc_out-34-5.0", "fc_out-39-5.0",
    "fc_out-28-9.9", "v_proj-29-9.9", "v_proj-34-9.9", "v_proj-39-9.9",
    "fc_in-29-9.9", "fc_in-34-9.9", "fc_in-39-9.9", "fc_out-29-9.9", "fc_out-34-9.9", "fc_out-39-9.9",
    "v_proj-29-1.0", "v_proj-34-1.0", "v_proj-39-1.0", "v_proj-5-1.0", "v_proj-5-5.0"
}

# Iterate over all combinations of lname, lnum, and rate
for lname in layer_names:
    for lnum in layer_numbers:
        for rate in rates:
            if lname in ['k_proj', 'q_proj']:
                continue
            if lname == 'v_proj' and lnum <= 4:
                continue
            rate_str = f"{rate:.10f}".rstrip('0').rstrip('.')

            combination = f"{lname}-{lnum}-{rate}"
            if combination in excluded_combinations:
                print(f"Skipping excluded combination: {combination}")
                continue

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