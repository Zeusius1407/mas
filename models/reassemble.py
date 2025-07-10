def merge_model_parts(output_file, part_prefix, num_parts):
    with open(output_file, 'wb') as outfile:
        for i in range(97, num_parts + 97):
            i = chr(i)
            part_file = f"{part_prefix}a{i}"  # e.g., "model.bin.001"
            with open(part_file, 'rb') as infile:
                outfile.write(infile.read())
    print(f"Merged {num_parts} parts into {output_file}")

# Usage:
merge_model_parts(
    output_file="mistral-model.bin",
    part_prefix="mistral_file_part_",
    num_parts=5  # Adjust to your actual number of parts
)