import os
import pandas as pd

# Define file paths
downloads_folder = os.path.expanduser("~/Downloads")
input_file = os.path.join(downloads_folder, "rapidminernew.xlsx")
output_csv = os.path.join(downloads_folder, "rapidminernew_cleaned.csv")
output_xlsx = os.path.join(downloads_folder, "rapidminernew_cleaned.xlsx")

# Read the input Excel file
df = pd.read_excel(input_file)

# Drop unnecessary columns
columns_to_keep = ["link-href", "user", "timestamp", "title", "body", "tags"]
df = df[columns_to_keep]

# Rename 'link-href' to 'link'
df.rename(columns={"link-href": "link"}, inplace=True)

# Remove "english" (case-insensitive) from the tags column
df["tags"] = df["tags"].str.replace(r"(?i)\benglish\b", "", regex=True)

# Remove any empty or extra commas after the replacement
df["tags"] = df["tags"].str.replace(r",+", ",", regex=True).str.strip(",").replace("", pd.NA)

# Group by timestamp and aggregate tags into a single comma-separated string
df_grouped = (
    df.groupby(["link", "user", "timestamp", "title", "body"], as_index=False)
    .agg({"tags": lambda x: ",".join(x.dropna().unique())})
)

# Save to CSV
df_grouped.to_csv(output_csv, index=False)
print(f"Cleaned data saved to CSV: {output_csv}")

# Save to Excel
df_grouped.to_excel(output_xlsx, index=False, engine="openpyxl")
print(f"Cleaned data saved to Excel: {output_xlsx}")
