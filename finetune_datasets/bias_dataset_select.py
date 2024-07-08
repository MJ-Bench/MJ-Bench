import os
import sys
sys.path.append("./")
from datasets import load_dataset
from tqdm import tqdm
import spacy
import json

# Define demographic information
age_keywords = ['young', 'old', 'elderly', 'teen', 'adult', 'child', 'kid', 'baby', 'toddler', 'infant', 'adolescent', 'middle-aged', 'senior', 'mature', 'youth', 'teenager', 'sophisticated', 'grown-up', 'young adult', 'aged']
age_bin_dict = {
    0: ['teen', 'child', 'kid', 'baby', 'toddler', 'infant', 'adolescent', 'youth', 'teenager'],
    1: ['adult', 'young adult', 'young', 'grown-up'],
    2: ['middle-aged', 'sophisticated', 'mature'],
    3: ['elderly', 'old', 'senior', 'aged']
}

gender_keywords = ['male', 'female', 'gay', 'lesbian', 'LGBT', 'non-binary', 'queer', 'man', 'woman', 'masculine', 'feminine']
gender_bin_dict = {
    0: ['male', 'man', 'masculine'],
    1: ['female', 'woman', 'feminine'],
    2: ['gay', 'lesbian', 'LGBT', 'non-binary', 'queer']
}

race_keywords = ['white', 'black', 'asian', 'latino', 'hispanic', 'indigenous', 'native', 'african', 'european', 'caucasian', 'middle eastern', 'south asian', 'east asian', 'southeast asian', 'pacific island']
race_bin_dict = {
    0: ['white', 'european', 'caucasian', 'middle eastern'],
    1: ['black', 'african'],
    2: ['asian', 'chinese', 'japanese', 'korean', 'indian', 'south asian', 'east asian', 'southeast asian'],
    3: ['latino', 'hispanic'],
    4: ['indigenous', 'native', 'pacific island']
}

nationality_keywords = ['american', 'canadian', 'mexican', 'brazilian', 'argentinian', 'german', 'french', 'italian', 'spanish', 'british', 'russian', 'polish', 'chinese', 'japanese', 'korean', 'indian', 'australian', 'new zealander', 'nigerian', 'south african', 'egyptian', 'kenyan']
nationality_bin_dict = {
    0: ['american', 'canadian', 'mexican', 'brazilian', 'argentinian'],
    1: ['german', 'french', 'italian', 'spanish', 'british', 'russian', 'polish'],
    2: ['chinese', 'japanese', 'korean', 'indian', 'australian', 'new zealander'],
    3: ['nigerian', 'south african', 'egyptian', 'kenyan']
}

religious_keywords = ['christian', 'church', 'bible', 'muslim', 'mosque', 'koran', 'jewish', 'synagogue', 'torah', 'hindu', 'buddhist', 'temple', 'religion']
religious_bin_dict = {
    0: ['christian', 'church', 'bible'],
    1: ['muslim', 'mosque', 'koran'],
    2: ['jewish', 'synagogue', 'torah'],
    3: ['hindu', 'buddhist', 'temple', 'religion']
}

# Define occupations and educational backgrounds
jobs = ['doctor', 'nurse', 'teacher', 'engineer', 'artist', 'scientist', 'lawyer', 'chef', 'pilot', 'writer']
education = ['PhD student', 'college student', 'high school student', 'graduate student', 'professor', 'researcher', 'scholar', 'educator', 'student']

# Function to filter captions based on demographic information
def check_demographic_group(caption):
    
    valid_flag = False

    has_occupation_education = False
    has_demographic_type = False
    
    # Check for occupations or education
    for job in jobs:
        if job in caption:
            has_occupation_education = True
            break
    for edu in education:
        if edu in caption:
            has_occupation_education = True
            break
    
    # Check for at least one demographic type (age, gender, or race)
    for age_bin, age_keywords in age_bin_dict.items():
        if any(keyword in caption for keyword in age_keywords):
            has_demographic_type = True
            break
    for gender_bin, gender_keywords in gender_bin_dict.items():
        if any(keyword in caption for keyword in gender_keywords):
            has_demographic_type = True
            break
    for race_bin, race_keywords in race_bin_dict.items():
        if any(keyword in caption for keyword in race_keywords):
            has_demographic_type = True
            break
    for nationality, nationality_keywords in nationality_bin_dict.items():
        if any(keyword in caption for keyword in nationality_keywords):
            has_demographic_type = True
            break
    for religion, religion_keywords in religious_bin_dict.items():
        if any(keyword in caption for keyword in religion_keywords):
            has_demographic_type = True
            break

    # If both occupation/education and demographic type are present, add the caption
    if has_occupation_education and has_demographic_type:
        valid_flag = True

    return valid_flag


def find_and_check_demographics(text):
    # Process the text with spaCy
    doc = nlp(text)
    
    # Initialize results dictionary
    results = {}
    
    # Iterate over tokens in the document
    for token in doc:
        # Check if the token is a noun and is in the jobs or education lists
        if token.pos_ == 'NOUN' and (token.text in jobs or any(token.text in edu for edu in education)):
            target_noun = token.text
            demographic_flags = {'age': False, 'gender': False, 'race': False, 'nationality': False, 'religion': False}
            
            # Check modifiers of the target noun
            for child in token.children:
                if child.dep_ == 'amod':  # checking for adjectival modifier
                    if child.text in age_keywords:
                        demographic_flags['age'] = True
                    if child.text in gender_keywords:
                        demographic_flags['gender'] = True
                    if child.text in race_keywords:
                        demographic_flags['race'] = True
                    if child.text in nationality_keywords:
                        demographic_flags['nationality'] = True
                    if child.text in religious_keywords:
                        demographic_flags['religion'] = True
            
            # Store results
            results[target_noun] = demographic_flags

    valid_flag = False
    for key in results:
        for value in results[key].values():
            if value:
                valid_flag = True
                break

    return valid_flag, results


nlp = spacy.load('en_core_web_sm')


dataset = load_dataset("yuvalkirstain/pickapic_v2_no_images")

filter_datalist = []

for example in tqdm(dataset["train"]):
    new_example = {}
    length = len(example["caption"].split())
    # valid_flag, results = find_and_check_demographics(example["caption"])
    # if valid_flag:
    #     print(example)
    #     print(results)
    #     input()
    if length < 20 and check_demographic_group(example["caption"]) and "cat" not in example["caption"] and "dog" not in example["caption"]:
        new_example["image_0_url"] = example["image_0_url"]
        new_example["image_1_url"] = example["image_1_url"]
        new_example["image_0_uid"] = example["image_0_uid"]
        new_example["image_1_uid"] = example["image_1_uid"]
        new_example["are_different"] = example["are_different"]
        new_example["best_image_uid"] = example["best_image_uid"]
        new_example["ranking_id"] = example["ranking_id"]
        new_example["__index_level_0__"] = example["__index_level_0__"]
        new_example["caption"] = example["caption"]

        filter_datalist.append(new_example)
    
    # if len(filter_datalist) >= 50:
    #     break

print(len(filter_datalist))

with open("finetune_datasets/biased_prompt_pickapic_v2_dataset.json", "w") as f:
    json.dump(filter_datalist, f, indent=4)