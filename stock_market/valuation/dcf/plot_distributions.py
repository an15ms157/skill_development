import json
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
import math

# Load the accepted parameters from the JSON file
#file_path = "output/accepted_parameters_AMZN.json"
#file_path = "output/accepted_parameters_NKE.json"
file_path = "output/accepted_parameters.json"
with open(file_path, "r") as f:
    data = json.load(f)

# Apply filter on data for wacc between 0.04 and 0.12
#data = [entry for entry in data if .0 <= entry["high_growth_rate"] <= .6]
#if not data:
#    raise ValueError("No data entries remain after applying the WACC filter. Please check the input data or filter criteria.")

# Load default values from NKE_config.json
default_path="NKE_config.json"
config_path = os.path.join(os.getcwd(), "config_file", default_path)
with open(config_path, "r") as config_file:
    config_data = json.load(config_file)
    
# Define the fields to extract
fields = [
    "wacc",
    "high_growth_rate",
    "terminal_growth_rate",
    "growth_decay",
    "margin_compression",
    "reinvestment_cap",
    "high_growth_years",
    "mc_mean - target_price"
]

# Calculate 'mc_mean - target_price' if not present
if "mc_mean - target_price" not in data[0]:
    for entry in data:
        entry["mc_mean - target_price"] = entry["mc_mean"] - entry["target_price"]

# Convert to DataFrame
df = pd.DataFrame(data)

# Add valuation error column
df["valuation_error"] = df["mc_mean"] - df["target_price"]

# Output directory
output_dir = "output"
os.makedirs(output_dir, exist_ok=True)

# -------------------------------
# 1. Distribution Plots (Histograms)
# -------------------------------
num_fields = len(fields)
num_cols = 2
num_rows = (num_fields + num_cols - 1) // num_cols

fig, axes = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=(12, 12))
axes = axes.flatten()

default_wacc = config_data["default_wacc"]
default_high_growth_rate = config_data["high_growth_rate"]
default_margin_compression = config_data["default_margin_compression"]
default_reinvestment_cap = config_data["reinvestment_cap"]

# Add default terminal growth rate (gdp_growth) to the plot
config_path = os.path.join(os.getcwd(), "config_file", "NKE_config.json")
with open(config_path, "r") as config_file:
    config_data = json.load(config_file)

default_terminal_growth_rate = config_data["gdp_growth"]

for i, field in enumerate(fields):
    sns.histplot(df[field], bins=20, kde=True, ax=axes[i], color="blue")
    axes[i].set_title(f"Distribution of {field}")
    axes[i].set_xlabel(field)
    axes[i].set_ylabel("Frequency")
    axes[i].grid(True)

    # Add default value lines for specific fields
    if field == "wacc":
        axes[i].axvline(default_wacc, color="red", linestyle="--", label="Default WACC")
    elif field == "high_growth_rate":
        axes[i].axvline(default_high_growth_rate, color="red", linestyle="--", label="Default High Growth Rate")
    elif field == "margin_compression":
        axes[i].axvline(default_margin_compression, color="red", linestyle="--", label="Default Margin Compression")
    elif field == "reinvestment_cap":
        axes[i].axvline(default_reinvestment_cap, color="red", linestyle="--", label="Default Reinvestment Cap")
    elif field == "terminal_growth_rate":
        axes[i].axvline(default_terminal_growth_rate, color="red", linestyle="--", label="Assumed GDP Growth")

    # Add legend if a default line was added
    if field in ["wacc", "high_growth_rate", "margin_compression", "reinvestment_cap", "terminal_growth_rate"]:
        axes[i].legend()

# Hide any unused subplots
for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.savefig(os.path.join(output_dir, "all_distributions.png"))
plt.close()

# -------------------------------
# 2. Correlation Heatmap
# -------------------------------
# Update correlation heatmap to include 'high_growth_years'
corr = df[fields].corr()
plt.figure(figsize=(10, 8))
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title("Correlation Matrix of Parameters")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "correlation_matrix.png"))
plt.close()

# -------------------------------
# 3. Joint Distributions (KDE plots)
# -------------------------------
# Define parameter pairs to plot
param_pairs = [
    ("wacc", "high_growth_rate"),
    ("high_growth_rate", "growth_decay"),
    ("terminal_growth_rate", "wacc"),
    ("margin_compression", "reinvestment_cap"),
    ("wacc", "mc_mean - target_price"),
    ("high_growth_years", "mc_mean - target_price"),
    ("high_growth_rate", "mc_mean - target_price"),
    ("high_growth_years", "high_growth_rate"),
]


num_pairs = len(param_pairs)
num_cols = 2
num_rows = (num_pairs + num_cols - 1) // num_cols

fig, axes = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=(12, 12))
axes = axes.flatten()

for i, (x, y) in enumerate(param_pairs):
    sns.kdeplot(data=df, x=x, y=y, fill=True, cmap="Blues", ax=axes[i])
    axes[i].set_title(f"{x} vs {y}")
    axes[i].set_xlabel(x)
    axes[i].set_ylabel(y)

# Hide any unused subplots
for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.savefig(os.path.join(output_dir, "all_joint_distributions.png"))
plt.close()
