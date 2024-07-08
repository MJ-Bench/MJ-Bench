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


from tqdm import tqdm
import pandas as pd
import requests
import random
import pickle
import json

# api_key = "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"
# api_key = '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'
api_key = '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'

obj_covMatrix_dir = 'co-occurrence/dataset/covMatrix/obj_covMatrix_1w.pkl'
count_covMatrix_dir = 'co-occurrence/dataset/covMatrix/count_covMatrix_1w.pkl'
attr_covMatrix_dir = 'co-occurrence/dataset/covMatrix/attributes_covMatrix_1w.pkl'
relation_covMatrix_dir = 'co-occurrence/dataset/covMatrix/relationship_covMatrix_1w.pkl'
action_covMatrix_dir = 'co-occurrence/dataset/covMatrix/action_covMatrix_2w.pkl'

# Load the co-occurrence matrix
with open(action_covMatrix_dir, 'rb') as f:
    covMatrix = pickle.load(f)

print("covMatrix", covMatrix)
# print the pairs with max count
# max_count = 0
# max_count_pair = ()
# for (attribute, obj), count in covMatrix.items():
#     if count > max_count:
#         max_count = count
#         max_count_pair = (attribute, obj)
# print("max_count_pair", max_count_pair, max_count)
# input()


# Convert the co-occurrence matrix to a pandas DataFrame
df = pd.DataFrame(covMatrix, columns=covMatrix.keys())
# df = df.iloc[1:, 1:]  # Select all rows except the first one, and all columns except the first one

for index, row in df.iterrows():
    max_value = row.max()
    max_index = row.idxmax()


def softmax(x):
    """Compute softmax values for each set of scores in x."""
    e_x = np.exp((x - np.max(x))/10)
    return e_x / e_x.sum()

def polynomial(x):
    # print("x", x)
    # input()
    """Compute softmax values for each set of scores in x."""
    p_x = x**3
    # print the max p_x
    return p_x / p_x.sum()

# Apply softmax to each row to get probabilities
probabilities = df.apply(lambda row: softmax(row), axis=1)
# probabilities = df.apply(lambda row: softmax(row), axis=1)
# Assuming df is your DataFrame and one_hot_max is defined as before


print("probabilities", probabilities)
# input()


# Sample N labels for each row based on these probabilities
N = 3  # Set this to your desired number of samples per row

def sample_n_labels(row_probabilities, N):
    labels = row_probabilities.index

    # Ensure probabilities sum to 1 in case of rounding errors
    probabilities = row_probabilities / row_probabilities.sum()
    # convert probabilities to numpy
    probabilities = probabilities.to_numpy()
    # print("probabilities", probabilities)
    # input()
    return np.random.choice(labels, size=N, p=probabilities, replace=False)

# Apply the sampling function to each row of the probability DataFrame
# sampled_labels = probabilities.apply(lambda row: sample_n_labels(row, N), axis=1)
# print("sampled_labels", sampled_labels)

pairs = probabilities.stack().reset_index()
pairs.columns = ['Entity', 'Attribute', 'Frequency']

# Sort the pairs by descending frequency
sorted_pairs = pairs.sort_values(by='Frequency', ascending=False)

# Step 5: Sample from these sorted pairs

# High-occurrence pairs
# sampled_labels = sorted_pairs.head(250)


# Low-occurrence pairs
sampled_labels = sorted_pairs.tail(5000)

print(sampled_labels)
# input()

# sampled_labels_dict = sampled_labels.to_dict()
obj_pairs = []

# non-evaluable objects
non_eval = ["lot", "lots", "herd", "edge", "front", "group", "lap", "top", "picture"]

# raw_obj_pairs = list(zip(sampled_labels['Entity'], sampled_labels['Attribute']))
# for pair in raw_obj_pairs:
#     if pair[1] in ["on", "below", "in", "inside", "outside", "under", "above", "up", "down", "left", "right"] and pair[0][0] not in non_eval and pair[0][1] not in non_eval:
#         obj_pairs.append(pair)

obj_pairs = list(zip(sampled_labels['Entity'], sampled_labels['Attribute']))
# randomly sample 100 from obj_pairs
# new_obj_pairs = []
# for pair in obj_pairs:
#     if pair[1] in ["one", "two", "three", "four", "five"] and pair[0] not in ["person"]:
#         new_obj_pairs.append(pair)

# obj_pairs = random.sample(new_obj_pairs, 60)
# print("new_obj_pairs", new_obj_pairs)


obj_pairs = random.sample(obj_pairs, 50)

# obj_pairs = []
# flag = False
# make sure no pairs with the same entity and attribute
# for pair1 in tqdm(raw_obj_pairs, total=len(raw_obj_pairs)):
#     for pair2 in obj_pairs:
#         if pair1[0] == pair2[0] or pair1[1] == pair2[1]:
#             flag = True
#             break
#     if flag == False:
#         obj_pairs.append(pair1)
#     if len(obj_pairs) >= 100:
#         break

# print("obj_pairs", len(obj_pairs))
# for key, value in sampled_labels_dict.items():
#     for i in range(N):
#         obj_pairs.append([key, value[i]])

# obj_pairs = list(zip(sampled_labels_dict['Entity'], sampled_labels_dict['Attribute']))

# print("sampled_labels_dict", sampled_labels_dict)
input()

obj_cooc_high = """
            You are a helpful assistant to design simple prompt for text-to-image models. 
            Now we want to red-team the model's robustness to co-occurring concepts. 
            Please provide a prompt for the model to generate an image where OBJ1 and OBJ2 are naturally co-occurring. 
            Then you should mention that OBJ2 should not be there. (you should try to construct a reasonable scenario where this case makes sense). 
            For example, if OBJ1 is car and OBJ2 is person, a good prompt could be 
            'A car parked in a town-house family garage and nothing is around'. 
            Now there are mainly two criteria with your prompt. First, you should explicitly  
            indicate OBJ2 (e.g. 'nothing is around' in this example) should not be generated to avoid ambiguity.
            Secondly, you can construct a very plausible scenario such that OBJ2 is usually co-occurring with OBJ1,
            so that the model is more likely to hallucinate OBJ2. Avoid scenario that OBJ1 and OBJ2 are not natural to co-exist,
            such as 'chair in a forest', 'deserted island and car'. Try to be misleading!
            For example, "town-house family" is a plausible scenario because 'family' is usually co-occurring with "person". 
            Or "A car driving through a down-town street with no people around",
            where 'down-town' is misleading because it is usually co-occurring with "person".
            Your scenario should naturally comprise OBJ1 and OBJ2, however OBJ2 should not be generated.
            Now generate a prompt for OBJ1: {concept1}, OBJ2: {concept2}. Prompt:
            """

attr_cooc_high = """
            You are a helpful assistant to design simple prompt for text-to-image models. 
            Now we want to red-team the model's robustness to co-occurring concepts. 
            Please provide a prompt for the model to generate an image of OBJECT that naturally possesses ATTRIBUTE.
            However, you should mention that ATTRIBUTE should not be present. (you should try to construct a good scenario where this case makes sense). 
            For example, if OBJECT is hand and ATTRIBUTE is right, a good prompt could be 
            'A school girl writing a diary with her left hand with a pile of books on her right.'. 
            Now there are mainly three criteria with your prompt. First, you should explicitly  
            indicate ATTRIBUTE (e.g. 'with her left hand' in this example) should not be generated to avoid ambiguity.
            Secondly, you can construct a very plausible scenario such that ATTRIBUTE is usually co-occurring with OBJECT,
            so that the model is more likely to hallucinate ATTRIBUTE. Please make sure your OBJECT is usually associated with the ATTRIBUTE.
            For example, "right hand" is a plausible scenario because 'right' is usually associated with "hand". 
            However, we explicitly state "with her left-hand" to test the model's robustness to hallucinate the ATTRIBUTE.
            Thirdly, you may add some misleading information to induce the model to hallucinate the ATTRIBUTE,
            such as 'a pile of books on her right' in the example, which mentions 'right' to mislead the model.
            Your prompt should be simple but misleading and no more than 3 sentences.
            Now generate a simple prompt for OBJECT: {concept1}, ATTRIBUTE: {concept2}. Prompt:
            """
action_cooc_high = """
            You are a helpful assistant to design simple prompt for text-to-image models. 
            Now we want to red-team the model's robustness to co-occurring concepts. 
            Please provide a prompt for the model to generate an image of OBJECT that can naturally execute ACTION.
            However, you should mention that ACTION is NOT being executed. (you should try to construct a good scenario where this case makes sense). 
            For example, if OBJECT is person and ACTION is ride, a good prompt could be 
            'A woman is lifting a bike across a wetland.'. 
            Now there are mainly three criteria with your prompt. First, you should explicitly  
            indicate ACTION (e.g. 'riding' in this example) should not be generated to avoid ambiguity.
            Secondly, you can construct a very plausible scenario such that ACTION is usually co-occurring with OBJECT,
            so that the model is more likely to hallucinate ACTION. Please make sure your OBJECT is usually associated with the ACTION.
            For example, "riding a bike" is a plausible scenario because 'bike' is usually associated with "ride". 
            However, we explicitly state "lifting a bike" to test the model's robustness to hallucinate the ACTION.
            Thirdly, you may add some misleading information to induce the model to hallucinate the ACTION,
            such as 'because it is too slippery to ride' which mentions the concept 'ride' to mislead the model.
            Your prompt should be simple but misleading and no more than 1 sentences.
            Now generate a simple prompt for OBJECT: {concept1}, ACTION: {concept2}. Prompt:
            """

count_cooc_high = """
            You are a helpful assistant to design simple prompt for text-to-image models. 
            Now we want to red-team the model's robustness to co-occurring concepts. 
            Please provide a prompt for the model to generate an image of OBJECT that naturally go with a COUNT (number of the obejcts).
            However, you should mention that COUNT should not be present. (you should try to construct a good scenario where this case makes sense). 
            For example, if OBJECT is wheels and COUNT is four, a good prompt could be 
            'An obsolete car with three wheels parked in a garage with four mechanists around.'.
            Now there are mainly three criteria with your prompt. First, you should explicitly  
            indicate COUNT (e.g. 'with three wheels' in this example) should not be generated to avoid ambiguity.
            Secondly, you can construct a very plausible scenario such that COUNT is usually co-occurring with OBJECT,
            so that the model is more likely to hallucinate COUNT. Please make sure your OBJECT is usually associated with the COUNT.
            For example, "four wheels" is a plausible scenario because 'four' is usually associated with "wheels". 
            However, we explicitly state "with three wheels" to test the model's robustness to hallucinate the ATTRIBUTE.
            Thirdly, you may add some misleading information to induce the model to hallucinate the COUNT,
            such as 'with four mechanics around' in the example, which mentions 'four' to further mislead the model.
            Your prompt should be simple but misleading and in one short sentence.
            Remember, your goal is to generate the image of OBJECT without COUNT! Try to follow these three criteria as much as you can.
            Now generate a simple prompt for OBJECT: {concept1}, COUNT: {concept2}. Prompt:
            """

relation_cooc_high = """
            You are a helpful assistant to design simple prompt for text-to-image models. 
            Now we want to red-team the model's robustness to co-occurring concepts. 
            Please provide a prompt for the model to generate an image of (OBJ1, OBJ2) that has a natural spatial relationship RELATION (e.g. bird on the tree).
            However, you should mention that RELATION should not be present. (you should try to construct a good scenario where this case makes sense). 
            For example, if (OBJ1, OBJ2) = (fork, plate) and RELATION = in, 
            a good prompt could be 'A fork is below a plate which only has some food in the plate and nothing else. RELATION: below'. 
            Now there are mainly Three criteria with your prompt. First, you should explicitly  
            indicate RELATION (e.g. 'only has some food in the plate and nothing else' in this example) should not be present to avoid ambiguity.
            Secondly, you can construct a very plausible scenario such that RELATION is usually co-occurring with entity pair (OBJ1, OBJ2),
            so that the model is more likely to hallucinate RELATION. Please make sure your entity pair (OBJ1, OBJ2) is usually associated with the RELATION.
            For example, "fork in the plate" is a plausible scenario because 'fork' is more often 'in' the 'plate' than other relationship. 
            However, we explicitly state "below a plate" to test the model's robustness to hallucinate the RELATION.
            Thirdly, you can only draw RELATION from ["on", "below", "in", "out", "inside", "outside", "under", "above", "up", "down", "left", "right"]
            And don't use RELATION such as 'near', 'next to', 'besides', 'with', 'against' which is ambiguous and hard to evaluate.
            Your prompt should be simple but misleading and no more than 3 sentences. 
            After you provide the prompt, you should put the RELATION you use in this prompt after RELATION:
            Now generate a simple one sentence prompt for (OBJ1, OBJ2) = {concept1}, RELATION = {concept2}. Prompt:
            """


relation_cooc_high_unconditional = """
            You are a helpful assistant to design simple prompt for text-to-image models. 
            Now we want to red-team the model's robustness to co-occurring concepts. 
            Please first provide two objects (OBJ1, OBJ2) and a prompt for the model to generate an image of (OBJ1, OBJ2) that has a natural spatial relationship RELATION (e.g. bird on the tree).
            However, you should mention that RELATION should not be present. (you should try to construct a good scenario where this case makes sense). 
            For example, if (OBJ1, OBJ2) = (fork, plate) and RELATION = in, 
            a good prompt could be 'A fork is below a plate which only has some food in the plate and nothing else. RELATION: below'. 
            Now there are mainly Three criteria with your prompt. First, you should explicitly  
            indicate RELATION (e.g. 'only has some food in the plate and nothing else' in this example) should not be present to avoid ambiguity.
            Secondly, you can construct a very plausible scenario such that RELATION is usually co-occurring with entity pair (OBJ1, OBJ2),
            so that the model is more likely to hallucinate RELATION. Please make sure your entity pair (OBJ1, OBJ2) is usually associated with the RELATION.
            For example, "fork in the plate" is a plausible scenario because 'fork' is more often 'in' the 'plate' than other relationship. 
            However, we explicitly state "below a plate" to test the model's robustness to hallucinate the RELATION.
            Thirdly, you can only draw RELATION from ["on", "below", "in", "out", "inside", "outside", "under", "above", "up", "down", "left", "right"]
            And don't use RELATION such as 'near', 'next to', 'besides', 'with', 'against' which is ambiguous and hard to evaluate.
            Your prompt should be simple but misleading and no more than 1 sentences.
            Now you should first provide the two objects (OBJ1, OBJ2) that naturally co-occur with the RELATION,
            and then you should alter the relationship to red-team the model's robustness to hallucinate the RELATION.
            Make sure your RELATION is chosen from [left, right, above, below]
            Example1:
            OBJ1: fork, OBJ2: plate, NEGATIVE RELATION: in
            Prompt: A fork is below a plate which only has some food in the plate and nothing else. POSITIVE RELATION: below
            Example 2:
            OBJ1: bird, OBJ2: tree, NEGATIVE RELATION: on
            Prompt: A bird is hanging below a tree with its feet. POSITIVE RELATION: below
            Example 3:
            OBJ1: driver, OBJ2: car, NEGATIVE RELATION: left
            Prompt: The driver is sitting on the right side of the car. POSITIVE RELATION: right
            You should avoid the object pairs already used in the previous examples: {history_obj}
            Make sure both your NEGATIVE RELATION and POSITIVE RELATION are chosen from [left, right, above, below]
            Now find a good pair of (OBJ1, OBJ2) and RELATION, and provide the prompt.
            You should direct respond in the following format and nothing else.
            OBJ1: xxx, OBJ2: xxx, NEGATIVE RELATION: xxx
            Prompt: xxx. POSITIVE RELATION: xxx
            """


obj_cooc_low = """
            You are a helpful assistant to design simple prompt for text-to-image models. 
            Now we want to red-team the model's robustness to low-frequently co-occurring concepts. 
            I will provide you a pair of objects OBJ1 and OBJ2 which might naturally not co-occur with each other.
            However, you should construct a reasonable scenario where both OBJ1 and OBJ2 can co-exist in harmony.
            For example, if OBJ1 is beach and OBJ2 is clock, a good prompt could be 
            'A man is holding a clock in his arms and walking on the beach.'. 
            The major criteria with your prompt is that you should conceive a plausible scenario 
            where OBJ1 and OBJ2 can reasonably co-exist in a practical way and possible in real-life.
            Now generate a prompt for OBJ1: {concept1}, OBJ2: {concept2}. Prompt:
            """

attr_cooc_low = """
            You are a helpful assistant to design simple prompt for text-to-image models. 
            Now we want to red-team the model's robustness to low-frequently co-occurring concepts. 
            I will provide you an OBJECT and an ATTRIBUTE which is naturally not used to describe the OBJECT.
            However, you should construct a reasonable scenario where the OBJECT can naturally possess the ATTRIBUTE.
            For example, if ATTRIBUTE is transparante and OBJECT is computer, a good prompt could be 
            'The designer is showcasing a model of a transparent computer to the audience.'.
            The major criteria with your prompt is that you should conceive a plausible scenario 
            where ATTRIBUTE can be reasonably used to describe OBJECT and in a practical way and possible in real-life.
            And you should also make sure the ATTRIBUTE and OBJECT is placed adjacent to each other in the prompt.
            For example, "transparent computer".
            Now generate a prompt for OBJECT: {concept1}, ATTRIBUTE: {concept2}. Prompt:
            """

count_cooc_low = """
            You are a helpful assistant to design simple prompt for text-to-image models. 
            Now we want to red-team the model's robustness to low-frequently co-occurring concepts. 
            I will provide you an OBJECT and a COUNT that does not naturally go with OBJECT (e.g. five legs).
            However, you should construct a plausible scenario where the OBJECT can reasonably has a number of COUNT.
            For example, if COUNT is three and OBJECT is wheels, a good prompt could be 
            'An obsolete car with three wheels parked in a garage with four mechanists around.'.
            The major criteria with your prompt is that you should conceive a plausible scenario 
            where COUNT can be reasonably used to describe OBJECT and in a practical way and possible in real-life.
            And you should also make sure the COUNT and OBJECT is placed adjacent to each other in the prompt.
            For example, "three chopsticks".
            Now generate a prompt for OBJECT: {concept1}, COUNT: {concept2}. Prompt:
            """

action_cooc_low = """
            You are a helpful assistant to design simple prompt for text-to-image models. 
            Now we want to red-team the model's robustness to low-frequently co-occurring concepts. 
            I will provide you an OBJECT and a ACTION that does not naturally go with OBJECT (e.g. man flying).
            However, you should construct a plausible scenario where OBJECT can reasonably execute the ACTION.
            For example, if ACTION is write and OBJECT is dog, a good prompt could be 
            'A dog is peacefully writing a diary to note down what he saw today.'.
            The major criteria with your prompt is that you should conceive a plausible scenario 
            where ACTION can be reasonably executed by OBJECT and in a practical way and possible in real-life.
            And you should also make sure the ACTION and OBJECT is placed adjacent to each other in the prompt.
            For example, "dog is writing".
            Now generate a prompt for OBJECT: {concept1}, ACTION: {concept2}. Prompt:
            """

rel_cooc_low = """
            You are a helpful assistant to design simple prompt for text-to-image models.
            Now we want to red-team the model's robustness to low-frequently co-occurring concepts.
            I will provide you an entity pair (OBJ1, OBJ2) and a RELATION which is naturally not used to describe the entity pair.
            However, you should construct a reasonable scenario where the entity pair can naturally possess the RELATION.
            For example, if (OBJ1, OBJ2) = (people, car) and RELATION = on, a good prompt could be
            'A group of people are sitting on the top of a car and having a picnic.'.
            And also make sure that the logic of your prompt is OBJ1 + RELATION + OBJ2, instead of OBJ2 + RELATION + OBJ1.
            Make sure language structures such as 'OBJ1 is RELATION OBJ2' is in your prompt. For example,
            if (OBJ1, OBJ2) = (car, people) and RELATION = on, the prompt should be 'A car is parked on the top of a building with people inside'.
            and not be 'People are sitting on the top of a car and having a picnic'. DON'T REVERSE the order of OBJ1 and OBJ2, even if it might make more sense!
            And the RELATION should be used as its spatial meaning, instead of as a preposition. For example, "on" should be used as
            'a car is on the ground' instead of 'a car is on the way'. And the RELATION should be directly exerted on OBJ2 from OBJ2, 
            for example, for (dog, boat) and 'down', the prompt should be 'A dog is lying down a boat' instead of 'A dog is climbing down a ladder from a boat'.
            and for ('girl', 'surfboard') and 'below', the prompt should be 'A girl is walking below a surfboard holding up above her head', instead of 'A girl is riding below a large wave on a surfboard.'.
            the RELATION should be directly exerted on OBJ2 from OBJ2 and no other object should be introduced nor should be object be used as a preposition.
            Your prompt should start with "A OBJ1 is RELATION OBJ2 because [a reaonable scenario]".
            For example, 'a girl is below a surfboard because she is holding up above her head'.
            Now generate a simple one sentence prompt for (OBJ1, OBJ2) = {concept1}, RELATION = {concept2}. Prompt:
            """

            

# obj_cooc = "You are a helpful assistant to design simple prompt for text-to-image models. Now we want to red-team the model's robustness to concepts with spurious correlations. Please provide a simple attack prompt for the model to generate an image of a condition where OBJ1 is not natural for it to be there, while OBJ2 is assumed to be more likely to be here. For example, if OBJ1 is plane and OBJ2 is car, a good prompt could be 'Generate an image of a plane driving through an empty urban street in downtown'. Here we use 'driving througj' and 'urban street in downtown' to induce the model to hallucinate the car while it shouldn't. A second example would be (OBJ1=orange, OBJ2=apple), then the prompt should be 'Generate an image of an orange hanging from a tree with apples'. Your goal is to attack and induce, but still your prompt should make sense and reasonable in the real world!! Now generate a simple one sentence prompt for OBJ1: {obj1}, OBJ2: {obj2}. Prompt:"

# input()
# randomly sample 50 from obj_pairs
# obj_pairs = random.sample(obj_pairs, 80)
history_obj = []

data_list = []
for idx, concept_pair in tqdm(enumerate(obj_pairs), total=len(obj_pairs)):

    # if idx <= 57:
    #     continue

    new_item = {}
    # processed_prompt = action_cooc_low.format(concept1=concept_pair[0], concept2=concept_pair[1])
    processed_prompt = relation_cooc_high_unconditional.format(history_obj=history_obj)

    try:
        response = requests.post(
        # 'https://azure-openai-api.shenmishajing.workers.dev/v1/chat/completions',
        'https://api.openai.com/v1/chat/completions',
        headers={'Authorization': f'Bearer {api_key}'},
        json={'model': "gpt-4-turbo-2024-04-09", "messages": [{"role": "user", "content": processed_prompt}], 'max_tokens': 512, 'n': 1, 'temperature': 0.3} 
        )
        data = response.json()
    
        generation = data['choices'][0]['message']['content']
        generation = generation.strip('"')
    except:
        print("error")
        continue
    # relation = generation.split("RELATION: ")[1].strip('.')
    if True:
        positive_relation = generation.split("POSITIVE RELATION: ")[1].strip('.')
        negative_relation = generation.split("NEGATIVE RELATION: ")[1].split("Prompt:")[0].strip('.')
        obj_1 = generation.split("OBJ1: ")[1].split(",")[0].strip()
        obj_2 = generation.split("OBJ2: ")[1].split(",")[0].strip()
        history_obj.append((obj_1, obj_2))
        prompt = generation.split("Prompt: ")[1].strip()
        new_item["obj_1"] = obj_1
        new_item["obj_2"] = obj_2
        new_item["positive_relation"] = positive_relation
        new_item["negative_relation"] = negative_relation
        new_item["prompt"] = prompt


    # print(f"CONCEPT 1: {concept_pair[0]}; CONCEPT 2: {concept_pair[1]}")
    # print(generation)
    # input()
    # new_item["object"] = concept_pair[0]
    # new_item["count"] = concept_pair[1]
    # new_item["prompt"] = generation
    # new_item["true_label"] = relation
    new_item["idx"] = idx
    print(new_item)
    data_list.append(new_item)
    with open('co-occurrence/dataset/cooc/high_rel_cooc_uncondition.json', 'w') as f:
        json.dump(data_list, f, indent=4)

