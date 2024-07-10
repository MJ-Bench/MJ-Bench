import json

# Load the original JSON file
with open('/net/scratch/zhaorun/MJ-Bench/data/quality/blur.json', 'r') as file:
    data = json.load(file)

# Initialize lists for the two new JSONs
defocused_blur_json = []
motion_blur_json = []

# Process each entry in the original JSON
for entry in data:
    # Create a new entry for defocused blur
    defocused_entry = {
        "caption": entry["caption"],
        "image0": f"images/quality/blur/sharp/{entry['sharp_image']}",
        "image1": f"images/quality/blur/defocused_blurred/{entry['defocused_blur_image']}",
        "label": 0
    }
    defocused_blur_json.append(defocused_entry)

    # Create a new entry for motion blur
    motion_blur_entry = {
        "caption": entry["caption"],
        "image0": f"images/quality/blur/sharp/{entry['sharp_image']}",
        "image1": f"images/quality/blur/motion_blurred/{entry['motion_blur_image']}",
        "label": 0
    }
    motion_blur_json.append(motion_blur_entry)

# Save the defocused blur JSON to a new file
with open('/net/scratch/zhaorun/MJ-Bench/data/quality/defocused_blur.json', 'w') as file:
    json.dump(defocused_blur_json, file, indent=4)

# Save the motion blur JSON to a new file
with open('/net/scratch/zhaorun/MJ-Bench/data/quality/motion_blur.json', 'w') as file:
    json.dump(motion_blur_json, file, indent=4)

print("JSON files split successfully!")
