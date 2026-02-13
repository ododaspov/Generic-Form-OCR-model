"""
Reads OCR JSON results, sends extracted text to Mistral for normalization,
and writes final structured CSV.
"""

import json
import csv
import argparse

# 🔹 Replace this with your real Mistral call
def call_mistral_llm(field_dict):
    """
    field_dict example:
    {
        "branch": {"ocr_text": "...", "crop_path": "..."},
        "email": {"ocr_text": "...", "crop_path": "..."}
    }

    For now we simulate normalization.
    Replace with your actual Mistral API call.
    """

    normalized = {}

    for field_name, field_data in field_dict.items():
        text = field_data.get("ocr_text", "")

        # Example normalization logic
        normalized[field_name] = text.strip()

    return normalized


def process_ocr_json(input_json, output_csv):

    with open(input_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    all_rows = []

    for item in data:
        source_file = item.get("_source_file")
        fields = item.get("fields", {})

        # Send fields to Mistral
        normalized_fields = call_mistral_llm(fields)

        # Add source filename
        normalized_fields["_source_file"] = source_file

        all_rows.append(normalized_fields)

    # Collect all unique field names for CSV header
    header_fields = set()
    for row in all_rows:
        header_fields.update(row.keys())

    header_fields = sorted(header_fields)

    # Write CSV
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=header_fields)
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"Final CSV saved to {output_csv}")


# ------------------ CLI ------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_json", required=True,
                        help="OCR results JSON file")
    parser.add_argument("--output_csv", required=True,
                        help="Final CSV output file")

    args = parser.parse_args()

    process_ocr_json(args.input_json, args.output_csv)
