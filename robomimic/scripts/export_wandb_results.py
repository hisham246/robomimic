import wandb
import pandas as pd

# Initialize the W&B API
api = wandb.Api()

# Retrieve the specific run (/<project>/<run_id>)
run = api.run("/airlab_uw/cami_extension/runs/iohsh34g")

# Scan the run history
history_list = [row for row in run.scan_history()]

# Convert to a Pandas DataFrame
history_df = pd.DataFrame(history_list)

# Save to a CSV file
history_df.to_csv("run_history_cami_cance_act_joint.csv", index=False)
