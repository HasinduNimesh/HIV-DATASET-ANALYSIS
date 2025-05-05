import pandas as pd

# Load metrics
metrics = pd.read_csv("results/model_metrics.csv")

# Save as Markdown (for easy copy-paste into your manuscript)
with open("results/metrics_table.md", "w") as f:
    f.write(metrics.to_markdown(index=False))

imp_df = pd.read_csv("results/model_metrics.csv", index_col=None)  # or reload from your imp_df
# If you saved imp_df earlier as CSV:


with open("results/imp_table.md", "w") as f:
    f.write(imp_df.to_markdown(index=False))
