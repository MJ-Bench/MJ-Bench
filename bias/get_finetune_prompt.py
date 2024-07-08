import json
import os

key_paths = {
    "young": {
        "female": {
            "black": "young_female_black.jpg",
            "white": "young_female_white.jpg",
            "asian": "young_female_asian.jpg",
            "hispanic": "young_female_hispanic.jpg",
            "indian": "young_female_indian.jpg",
            "middle eastern": "young_female_middle eastern.jpg"
        },
        "male": {
            "black": "young_male_black.jpg",
            "white": "young_male_white.jpg",
            "asian": "young_male_asian.jpg",
            "hispanic": "young_male_hispanic.jpg",
            "indian": "young_male_indian.jpg",
            "middle eastern": "young_male_middle eastern.jpg"
        },
        "non-binary": {
            "black": "young_non-binary_black.jpg",
            "white": "young_non-binary_white.jpg",
            "asian": "young_non-binary_asian.jpg",
            "hispanic": "young_non-binary_hispanic.jpg",
            "indian": "young_non-binary_indian.jpg",
            "middle eastern": "young_non-binary_middle eastern.jpg"
        }
    },
    "adult": {
        "female": {
            "black": "adult_female_black.jpg",
            "white": "adult_female_white.jpg",
            "asian": "adult_female_asian.jpg",
            "hispanic": "adult_female_hispanic.jpg",
            "indian": "adult_female_indian.jpg",
            "middle eastern": "adult_female_middle eastern.jpg"
        },
        "male": {
            "black": "adult_male_black.jpg",
            "white": "adult_male_white.jpg",
            "asian": "adult_male_asian.jpg",
            "hispanic": "adult_male_hispanic.jpg",
            "indian": "adult_male_indian.jpg",
            "middle eastern": "adult_male_middle eastern.jpg"
        },
        "non-binary": {
            "black": "adult_non-binary_black.jpg",
            "white": "adult_non-binary_white.jpg",
            "asian": "adult_non-binary_asian.jpg",
            "hispanic": "adult_non-binary_hispanic.jpg",
            "indian": "adult_non-binary_indian.jpg",
            "middle eastern": "adult_non-binary_middle eastern.jpg"
        }
    },
    "old": {
        "female": {
            "black": "old_female_black.jpg",
            "white": "old_female_white.jpg",
            "asian": "old_female_asian.jpg",
            "hispanic": "old_female_hispanic.jpg",
            "indian": "old_female_indian.jpg",
            "middle eastern": "old_female_middle eastern.jpg"
        },
        "male": {
            "black": "old_male_black.jpg",
            "white": "old_male_white.jpg",
            "asian": "old_male_asian.jpg",
            "hispanic": "old_male_hispanic.jpg",
            "indian": "old_male_indian.jpg",
            "middle eastern": "old_male_middle eastern.jpg"
        },
        "non-binary": {
            "black": "old_non-binary_black.jpg",
            "white": "old_non-binary_white.jpg",
            "asian": "old_non-binary_asian.jpg",
            "hispanic": "old_non-binary_hispanic.jpg",
            "indian": "old_non-binary_indian.jpg",
            "middle eastern": "old_non-binary_middle eastern.jpg"
        }
    }
}


def get_dirs(path):
    dirs = []
    for root, directories, files in os.walk(path):
        if not directories:  # Check if there are no more subdirectories
            dirs.append(root)
    return dirs

demographic_dirs = get_dirs("/home/czr/MM-Reward/bias/all_dataset")

path_list = []

prompt_list = []

for demographic_dir in demographic_dirs:
    for root, occupation_dir, files in os.walk(demographic_dir):

        for occupation in files:
            image_dir = os.path.join(root, occupation)
            age = True if "education" not in demographic_dir else False
            gender = True
            race = True if "race" in demographic_dir or "education" in demographic_dir else False
            nationality = True if "nationality" in demographic_dir or "education" in demographic_dir  else False
            religion = True if "religion" in demographic_dir else False


            occupation_name =  demographic_dir.split("/")[-1].split("_")[0]
            demographic_name = image_dir.split("/")[-1].split(".jpg")[0].split("_")
            demographic_name = " ".join(demographic_name)

            occupation_name = occupation_name.lower()

            if occupation_name[0] == "a" or occupation_name[0] == "e" or occupation_name[0] == "i" or occupation_name[0] == "o" or occupation_name[0] == "u":
                prompt = "An " + occupation_name + "."
            else:
                prompt = "A " + occupation_name + "."

            prompt_list.append(prompt)
            break

            # path_list.append({
            #     "demographic": image_dir.split("/")[-1].split(".jpg")[0],
            #     "occupation": demographic_dir.split("/")[-1].split("_")[0],
            #     "images_dir": image_dir.split("/home/czr/MM-Reward/bias/")[-1],
            #     "prompt": prompt,
            #     "age": age,
            #     "gender": gender,
            #     "race": race,
            #     "nationality": nationality,
            #     "religion": religion
            # })

# save txt
with open('bias/bias_finetune.txt', 'w') as f:
    for item in prompt_list:
        f.write("%s\n" % item)
