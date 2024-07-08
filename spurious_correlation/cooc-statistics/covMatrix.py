from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

import numpy as np

import time
import os
from six.moves import cPickle
import traceback
from collections import defaultdict

import seaborn as sns
import matplotlib.pyplot as plt
import spacy
from tqdm import tqdm
from collections import defaultdict
import pandas as pd
import json, pickle

if False:
    ######### object co-occurrence #########
    opt = opts.parse_opt()
    # print(opt)
    # loader = DataLoader(opt)
    covMatrix = pickle.load(open('data/covMatrix/covMatrixAnns.pkl', 'rb'))
    synonyms_dir = "data/synonyms.txt"
    print(covMatrix)

    with open(synonyms_dir, 'r') as file:
        synonyms = file.read()

    # Splitting the text into lines
    synonyms_lines = synonyms.split('\n')

    # Extracting the first word of each line as the category label
    labels = [line.split(',')[0] for line in synonyms_lines if line]  # ensuring no empty lines are processed

    print("labels", labels)

    co_occurrence_matrix = pd.DataFrame(covMatrix, index=labels, columns=labels)
    with open('data/covMatrix/obj_covMatrix_1w.pkl', 'wb') as f:
        pickle.dump(co_occurrence_matrix, f)

    cbar_kws = {"shrink": 0.75}
    # Plot the covariance matrix
    plt.figure(figsize=(50, 50))  # Adjust the size as necessary
    sns.set(font_scale=3.5)  # Adjust font scale for labels as necessary
    ax = sns.heatmap(covMatrix, annot=False, xticklabels=labels, yticklabels=labels,
                    cmap='Reds', square=True, cbar_kws=cbar_kws)  # viridis is a visually appealing colormap
    # ax.set_title('Covariance Matrix from Annotations', fontsize=30)  # Title with a larger font size
    plt.xticks(rotation=90)  # Rotate x-axis labels for better readability
    plt.yticks(rotation=0)  # Keep y-axis labels horizontal
    plt.tight_layout()  # Ensure the labels and title fit well into the plot
    plt.savefig('covMatrix.png')  # Save the plot as an image


if False:
    ######### attributes object co-occurrence #########

    tagging = spacy.load("en_core_web_sm")
    coco_dataset_dir = "data/dataset_coco.json"

    with open(coco_dataset_dir, 'r') as file:
        coco_dataset = json.load(file)

    images = coco_dataset["images"]

    valid_list = ["NOUN", "PROPN", "ADD"] #, "ADJ"]

    objects_with_attributes = []
    for image in tqdm(images[:10000]):
        sentences = image["sentences"]
        # print("sentences", sentences)
        # input()
        for sentence in sentences:
            raw = sentence["raw"]
            doc = tagging(raw)

            for token in doc:
                if token.pos_ == 'NOUN':  # Check if the token is a noun
                    attributes = []
                    # print("token", token)
                    # input()
                    for child in token.children:
                        # print("child", child)
                        if child.dep_ == 'amod':  # Check if the child is an adjective modifying the noun
                            attributes.append(child.text)
                    if attributes:  # If there are adjectives found, add them along with the noun
                        objects_with_attributes.append((attributes, token.text))


    print("objects_with_attributes", objects_with_attributes)

    # Step 1: Initialize a dictionary to store co-occurrence counts
    co_occurrence_counts = defaultdict(int)

    # Step 2: Populate the dictionary with counts
    for attributes, obj in objects_with_attributes:
        for attribute in attributes:
            co_occurrence_counts[(attribute, obj)] += 1

    # Step 3: Convert the dictionary into a DataFrame for easier analysis and visualization
    # Extract unique attributes and objects for the DataFrame index and columns
    attributes_set = set([attr for attr, _ in co_occurrence_counts.keys()])
    objects_set = set([obj for _, obj in co_occurrence_counts.keys()])

    # Initialize a DataFrame filled with zeros
    co_occurrence_matrix = pd.DataFrame(0, index=sorted(objects_set), columns=sorted(attributes_set))

    # Populate the DataFrame with the counts
    for (attribute, obj), count in co_occurrence_counts.items():
        co_occurrence_matrix.at[obj, attribute] = count

    co_occurrence_matrix = co_occurrence_matrix[co_occurrence_matrix.sum(axis=1) >= 5]
    co_occurrence_matrix = co_occurrence_matrix.loc[:, (co_occurrence_matrix.sum(axis=0) >= 3)]

    # save the co_occurrence_matrix
    with open('data/covMatrix/attributes_covMatrix_1w.pkl', 'wb') as f:
        pickle.dump(co_occurrence_matrix, f)

    print("co_occurrence_matrix", co_occurrence_matrix)

    # co_occurrence_matrix = pd.DataFrame(co_occurrence_matrix, index=sorted(objects_set), columns=sorted(attributes_set))

    # Plotting
    cbar_kws = {"shrink": 0.75}
    plt.figure(figsize=(50, 50))  # Adjust the size as necessary
    sns.set(font_scale=4.5)  # Adjust font scale for labels as necessary
    ax = sns.heatmap(co_occurrence_matrix, annot=False, cmap='Purples', square=True, cbar_kws=cbar_kws)  # viridis is a visually appealing colormap
    # ax.set_title('Co-occurrence Matrix of Attributes and Objects', fontsize=14)  # Title with a larger font size


    plt.xticks(rotation=90)  # Rotate x-axis labels for better readability
    plt.yticks(rotation=0)  # Keep y-axis labels horizontal
    plt.tight_layout()  # Ensure the labels and title fit well into the plot

    plt.savefig('attributes_covMatrix.png')  # Save the plot as an image

if False:
    ######### relationship object co-occurrence #########
    
    tagging = spacy.load("en_core_web_sm")
    coco_dataset_dir = "data/dataset_coco.json"

    with open(coco_dataset_dir, 'r') as file:
        coco_dataset = json.load(file)

    images = coco_dataset["images"]
    object_relationships = []

    for image in tqdm(images[:10000]):  # Assuming `images` is a list of image data with sentences
        sentences = image["sentences"]
        for sentence in sentences:
            raw = sentence["raw"]
            doc = tagging(raw)

            for token in doc:
                # Check if the token is a preposition indicating a relationship
                if token.dep_ == "prep":
                    # The object of the preposition (the noun it is related to)
                    prep_obj = [child for child in token.children if child.dep_ == "pobj"]
                    if prep_obj:
                        prep_obj = prep_obj[0]  # Assuming only one object of preposition for simplicity
                        # The subject (noun) the preposition is related to
                        head_noun = token.head
                        if head_noun.pos_ == "NOUN" and prep_obj.pos_ == "NOUN":
                            object_relationships.append((["object1", "object2"], "relationship"))

                            # Replace placeholders with actual text
                            object_relationships[-1] = ([head_noun.text, prep_obj.text], token.text)

    # Adjusted to handle direct noun-noun compound relationships
    for token in doc:
        if token.dep_ in ["nsubj", "dobj"] and token.head.pos_ == "VERB":
            for child in token.children:
                if child.dep_ == "prep":
                    for grandchild in child.children:
                        if grandchild.dep_ == "pobj":
                            object_relationships.append(([token.text, grandchild.text], child.text))

    print(object_relationships)


    # relationship_co_occurrence_counts = defaultdict(int)

    # for (objects, relationship) in object_relationships:
    #     # Convert the list 'objects' to a tuple, making it hashable and usable as a dictionary key
    #     objects_tuple = tuple(objects)  # Convert list to tuple
    #     relationship_co_occurrence_counts[(objects_tuple, relationship)] += 1

    # # Extract unique objects and relationships for the DataFrame index and columns
    # objects_set = set()
    # relationships_set = set()

    # for (objects, relationship) in object_relationships:
    #     objects_set.update(objects)
    #     relationships_set.add(relationship)

    # # Initialize a DataFrame filled with zeros
    # relationship_co_occurrence_matrix = pd.DataFrame(0, index=sorted(objects_set), columns=sorted(relationships_set))

    # # Populate the DataFrame with the counts
    # for ((object1, object2), relationship), count in relationship_co_occurrence_counts.items():
    #     # Here, we might need to think about how to represent relationships since they involve pairs of objects
    #     # For simplicity, let's just increment counts for both objects under the relationship column
    #     relationship_co_occurrence_matrix.at[object1, relationship] += count
    #     relationship_co_occurrence_matrix.at[object2, relationship] += count

    # # Filter the matrix according to your criteria
    # relationship_co_occurrence_matrix = relationship_co_occurrence_matrix[relationship_co_occurrence_matrix.sum(axis=1) >= 0]
    # relationship_co_occurrence_matrix = relationship_co_occurrence_matrix.loc[:, (relationship_co_occurrence_matrix.sum(axis=0) >= 0)]

    # cbar_kws = {"shrink": 0.75}
    # plt.figure(figsize=(50, 50))  # Adjust the size as necessary
    # sns.set(font_scale=4.5)  # Adjust font scale for labels as necessary
    # ax = sns.heatmap(relationship_co_occurrence_matrix, annot=False, cmap='Greens', square=True, cbar_kws=cbar_kws)
    # plt.xticks(rotation=90)  # Rotate x-axis labels for better readability
    # plt.yticks(rotation=0)  # Keep y-axis labels horizontal
    # plt.tight_layout()  # Ensure the labels and title fit well into the plot
    # plt.savefig('relationship_covMatrix.png')  # Save the plot as an image

    relationship_co_occurrence_counts = defaultdict(int)

    # Count occurrences of each object pair with each relationship
    for (objects, relationship) in object_relationships:
        objects_tuple = tuple(objects)  # Ensure objects are in a tuple
        relationship_co_occurrence_counts[(objects_tuple, relationship)] += 1

    # Creating a list of unique object pairs and relationships for indexing
    object_pairs = set()
    relationships = set()

    for (objects_tuple, relationship) in relationship_co_occurrence_counts.keys():
        object_pairs.add(objects_tuple)
        relationships.add(relationship)

    # Sort for consistent ordering
    object_pairs = sorted(list(object_pairs))
    relationships = sorted(list(relationships))

    # Initialize a DataFrame to hold the counts
    relationship_matrix = pd.DataFrame(0, index=object_pairs, columns=relationships)

    # Populate the DataFrame
    for (objects_tuple, relationship), count in relationship_co_occurrence_counts.items():
        relationship_matrix.at[objects_tuple, relationship] = count

    relationship_matrix = relationship_matrix[relationship_matrix.sum(axis=1) >= 2]
    relationship_matrix = relationship_matrix.loc[:, (relationship_matrix.sum(axis=0) > 0)]

    # save the co_occurrence_matrix
    with open('data/covMatrix/relationship_covMatrix_1w.pkl', 'wb') as f:
        pickle.dump(relationship_matrix, f)
        
    # Adjusting the figure size and font scale for visibility
    plt.figure(figsize=(50, 50))  # Adjust as necessary
    sns.set(font_scale=5)  # Adjust as necessary

    # Create a heatmap
    ax = sns.heatmap(relationship_matrix, annot=False, cmap='Greens', cbar_kws={"shrink": 0.75})

    # Improve readability
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    # plt.title("Relationship Co-Occurrence Matrix")
    plt.tight_layout()

    # Save the figure
    plt.savefig('relationship_covMatrix.png')

if False:
    ######### count object co-occurrence #########

    tagging = spacy.load("en_core_web_sm")
    coco_dataset_dir = "/home/czr/object-bias/data/dataset_coco.json"

    with open(coco_dataset_dir, 'r') as file:
        coco_dataset = json.load(file)

    images = coco_dataset["images"]

    valid_list = ["NOUN", "PROPN", "ADD"] #, "ADJ"]

    objects_with_numeric_attributes = []

    for image in tqdm(coco_dataset["images"][:10000]):
        for sentence in image["sentences"]:
            doc = tagging(sentence["raw"])
            for token in doc:
                if token.pos_ in ['NOUN', 'PROPN']:  # Check if the token is a noun or proper noun
                    numeric_attributes = []
                    for child in token.children:
                        if child.dep_ == 'nummod' and child.head == token:  # Check if the child is a numeric modifier of the noun
                            # numeric_attributes.append(child.text + ' ' + token.text)
                            numeric_attributes.append(child.text)
                            # objects_with_numeric_attributes.append((child.text, token.text))
                    if numeric_attributes:  # If there are numeric attributes found, add them
                        objects_with_numeric_attributes.append((numeric_attributes, token.text))


    numeric_attribute_counts = defaultdict(int)

    # for count, obj in objects_with_numeric_attributes:
    #     numeric_attribute_counts[(count, obj)] += 1

    for attributes, obj in objects_with_numeric_attributes:
        for attribute in attributes:
            numeric_attribute_counts[(attribute, obj)] += 1


    # filtered_attributes = {attr: count for attr, count in numeric_attribute_counts.items() if count >= 3}

    # Step 3: Convert the dictionary into a DataFrame for easier analysis and visualization
    # Extract unique attributes and objects for the DataFrame index and columns
    count_set = set([attr for attr, _ in numeric_attribute_counts.keys()])
    objects_set = set([obj for _, obj in numeric_attribute_counts.keys()])

    # Initialize a DataFrame filled with zeros
    co_occurrence_matrix = pd.DataFrame(0, index=sorted(objects_set), columns=sorted(count_set))

    # Populate the DataFrame with the counts
    for (attribute, obj), count in numeric_attribute_counts.items():
        co_occurrence_matrix.at[obj, attribute] = count



    # save the co_occurrence_matrix
    with open('co-occurrence/dataset/covMatrix/count_covMatrix_1w.pkl', 'wb') as f:
        pickle.dump(co_occurrence_matrix, f)

    print("filtered_attributes", co_occurrence_matrix)



    # co_occurrence_matrix = pd.DataFrame(co_occurrence_matrix, index=sorted(objects_set), columns=sorted(attributes_set))

    # # Plotting
    # cbar_kws = {"shrink": 0.75}
    # plt.figure(figsize=(50, 50))  # Adjust the size as necessary
    # sns.set(font_scale=4.5)  # Adjust font scale for labels as necessary
    # ax = sns.heatmap(co_occurrence_matrix, annot=False, cmap='Purples', square=True, cbar_kws=cbar_kws)  # viridis is a visually appealing colormap
    # # ax.set_title('Co-occurrence Matrix of Attributes and Objects', fontsize=14)  # Title with a larger font size


    # plt.xticks(rotation=90)  # Rotate x-axis labels for better readability
    # plt.yticks(rotation=0)  # Keep y-axis labels horizontal
    # plt.tight_layout()  # Ensure the labels and title fit well into the plot

    # plt.savefig('attributes_covMatrix.png')  # Save the plot as an image

if True:
    ######### action object co-occurrence #########

    tagging = spacy.load("en_core_web_sm")
    coco_dataset_dir = "/home/czr/object-bias/data/dataset_coco.json"

    with open(coco_dataset_dir, 'r') as file:
        coco_dataset = json.load(file)

    images = coco_dataset["images"]



    # Dictionary to count object-action pairs
    object_action_counts = defaultdict(int)

    objects_with_actions = []
    # Process each sentence in each image description
    for image in tqdm(coco_dataset["images"][:20000]):
        for sentence in image["sentences"]:
            doc = tagging(sentence["raw"])
            for token in doc:
                if token.pos_ == 'NOUN':
                    action_list = []
                    for child in token.children:
                        if child.dep_ == 'acl' and child.tag_ == 'VBG':  # Looking specifically for verb gerunds
                            action = child.lemma_  # Get the base form of the verb
                            action_list.append(action)

                    if action_list:  # If there are numeric attributes found, add them
                        objects_with_actions.append((action_list, token.text))


    # # Convert to DataFrame for easy handling and visualization
    # object_action_df = pd.DataFrame(list(object_action_counts.items()), columns=['object', 'action'])


    for actions, obj in objects_with_actions:
        for action in actions:
            object_action_counts[(action, obj)] += 1


    # filtered_attributes = {attr: count for attr, count in numeric_attribute_counts.items() if count >= 3}

    # Step 3: Convert the dictionary into a DataFrame for easier analysis and visualization
    # Extract unique attributes and objects for the DataFrame index and columns
    action_set = set([action for action, _ in object_action_counts.keys()])
    objects_set = set([obj for _, obj in object_action_counts.keys()])

    # Initialize a DataFrame filled with zeros
    co_occurrence_matrix = pd.DataFrame(0, index=sorted(objects_set), columns=sorted(action_set))

    # Populate the DataFrame with the counts
    for (action, obj), count in object_action_counts.items():
        co_occurrence_matrix.at[obj, action] = count

    print("co_occurrence_matrix", co_occurrence_matrix)

    # Save the DataFrame
    with open('co-occurrence/dataset/covMatrix/action_covMatrix_2w.pkl', 'wb') as f:
        pickle.dump(co_occurrence_matrix, f)
