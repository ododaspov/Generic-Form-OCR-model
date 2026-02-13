import pandas as pd

# Input / Output
input_csv = "final_loans.csv"
output_csv = "final_loans_clea.csv"

df = pd.read_csv(input_csv)

# Correct ordered field names based on your JSON
correct_columns = [
    "branch",
    "DOA",
    "acc_no",
    "names",
    "nationality",
    "p_code",
    "mobile_no",
    "email",
    "p_address",
    "net_salary",
    "id_no",
    "NOD"
]

# Remove _source_file if it exists
if "_source_file" in df.columns:
    df = df.drop(columns=["_source_file"])

# Get generic field columns
existing_fields = [col for col in df.columns if "field_" in col]

# Sort numerically (field_0, field_1, field_2...)
existing_fields_sorted = sorted(
    existing_fields,
    key=lambda x: int(x.split("_")[1])
)

# Rename them properly
rename_map = {}
for old, new in zip(existing_fields_sorted, correct_columns):
    rename_map[old] = new

df = df.rename(columns=rename_map)

# Reorder columns strictly
df = df[correct_columns]

df.to_csv(output_csv, index=False)

print(f"Clean CSV saved as {output_csv}")
