
import json
import sys
sys.path.append("detr/")
import torch
import os
import numpy as np
import open_clip
from PIL import Image
from torch import optim
from torch import nn
from torch.nn import functional as F
from torchvision import transforms
from torchvision.transforms import Normalize, Compose, RandomResizedCrop, InterpolationMode, ToTensor, Resize, \
    CenterCrop
from tqdm import tqdm
from torchvision import utils
import argparse
# from transformers import CLIPProcessor, CLIPModel
import pdb
from nltk.corpus import wordnet as wn
import random
from embedding_preprocess import get_cooc_words_embedding
import matplotlib.pyplot as plt

# Normalization tools
def inverse_normalize(image, mean, std):
    image = image.clone()
    for i in range(3):
        image[i] = image[i] * std[i] + mean[i]
    return image

def normalize(images, mean, std):
    mean = list(mean)
    std = list(std)
    mean = torch.tensor(mean).cuda()
    std = torch.tensor(std).cuda()
    images = images - mean[None, :, None, None]
    images = images / std[None, :, None, None]
    return images


def denormalize(images, mean, std):
    mean = list(mean)
    std = list(std)
    mean = torch.tensor(mean).cuda()
    std = torch.tensor(std).cuda()
    images = images * std[None, :, None, None]
    images = images + mean[None, :, None, None]
    return images


def inverse_transform(image_tensor, mean, std):
    inverse_transform = transforms.Compose([
        transforms.Lambda(lambda x: inverse_normalize(x, mean, std)),
        transforms.ToPILImage()
    ])
    image = inverse_transform(image_tensor)
    return image


def misleading_attack(victim_word, img):
    torch.cuda.is_available()  # cuda可用

    # 寻找近似词向量
    # import gensim.downloader as api
    # model_similarity = api.load('word2vec-google-news-300')

    # Input_folder
    folder_out = f"co-occurrence/optimization"  # TODO 更换存储路径

    # 补充
    folder_add_path = "co-occurrence/optimization"

    os.makedirs(folder_out, exist_ok=True)

    # CLIP_ViT_L_14
    model, _, preprocess = open_clip.create_model_and_transforms('ViT-L-14', pretrained='openai')

    tokenizer = open_clip.get_tokenizer('ViT-L-14')
    torch.autograd.set_detect_anomaly(True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # image = preprocess(Image.open("real_image.jpg")).unsqueeze(0).to(device)
    image_mean = getattr(model.visual, 'image_mean', None)
    image_std = getattr(model.visual, 'image_std', None)
    model.eval()

    logit_scale = model.logit_scale.exp().to(device)

    exclude = lambda n, p: p.ndim < 2 or "bn" in n or "ln" in n or "bias" in n or 'logit_scale' in n
    include = lambda n, p: not exclude(n, p)
    for param in model.parameters():
        param.requires_grad = False

    # get the embedding of high co-occurrence words
    # with open('./relevant_documents/coco_labels_embedding.json', 'r') as file:
    #     potential_similar_words_embedding = json.load(file)  # 从文件中读取 JSON 数据
    potential_similar_words_embedding = get_cooc_words_embedding(victim_word[0], model, tokenizer, device)


    # potential_similar_words_embedding_keys = [list(d.keys())[0] for d in potential_similar_words_embedding]
    potential_similar_words_embedding_keys = list(potential_similar_words_embedding.keys())
    # values = [list(d.values())[0][0] for d in potential_similar_words_embedding]
    values = [potential_similar_words_embedding[key] for key in potential_similar_words_embedding_keys]
    potential_similar_words_embedding_tensor = np.array(values)
    potential_similar_words_embedding_tensor = torch.tensor(potential_similar_words_embedding_tensor)
    potential_similar_words_embedding_tensor /= potential_similar_words_embedding_tensor.norm(dim=-1, keepdim=True)
    potential_similar_words_embedding_tensor = potential_similar_words_embedding_tensor.to(device).float()

    loss_list = []
    # for file in file_list:
    alphas = [1.0]
    updates = [12, 30, 50, 100, 5] #origin=10
    for idx in range(0, 1):  # TODO 更换数据
        file = img
        idx = img.split(".")[0]
        file_name_list = [item.replace(",", "") for item in victim_word]
        # file_name = file_name_list[0]  # TODO 更换数据

        for alpha in alphas:
            for update_set in updates:
                image_path = os.path.join(folder_add_path, file)

                # 找到与给定词语最相似的词语
                # source_word = file.split(".")[0]
                # similar_words = model_similarity.most_similar(source_word, topn=5)

                image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)

                named_parameters = list(model.named_parameters())
                gain_or_bias_params = [p for n, p in named_parameters if exclude(n, p) and p.requires_grad]
                rest_params = [p for n, p in named_parameters if include(n, p) and p.requires_grad]

                image.requires_grad = False

                noise_param = torch.randn_like(image).to(device)
                image_d = denormalize(image, image_mean, image_std).clone().to(device)
                noise_param.data = (noise_param.data + image_d).clamp(0, 1) - image_d.data
                noise_param.requires_grad_(True)
                noise_param.retain_grad()

                optimizer = optim.AdamW(
                    [
                        {"params": noise_param, "weight_decay": 0.},
                    ],
                    lr=1e-3
                )

                # 相似度计算
                # 图文匹配
                image_nor = normalize(image_d, image_mean, image_std)
                image_nor_embedding = model.encode_image(image_nor)

                print("image_nor_embedding", image_nor_embedding.shape)
                print("potential_similar_words_embedding_tensor", potential_similar_words_embedding_tensor.shape)

                # 文本匹配
                # file_text = tokenizer([file_name]).to(device)
                # file_text_features = model.encode_text(file_text)

                # image_nor_embedding torch.Size([1, 768])
                # potential_similar_words_embedding_tensor torch.Size([10, 768])


                similarity_caculate = image_nor_embedding @ potential_similar_words_embedding_tensor.T
                print("similarity_caculate", similarity_caculate)
                top_k_values, top_k_indices = torch.topk(similarity_caculate, k=70)  # TODO 输出全值
                potential_similar_words_t100 = [potential_similar_words_embedding_keys[idx] for idx in
                                                top_k_indices[0]]  # 相似性实体打印
                similar_word = 'ini'

                # 消除同义词
                synonym_sets_list = []
                for file_word in file_name_list:
                    synonym_set = wn.synsets(file_word)
                    synonym_sets = [itrem.lemma_names() for itrem in synonym_set]
                    for item in synonym_sets:
                        item = [item_word.replace("_", " ") for item_word in item]
                        synonym_sets_list = synonym_sets_list + item

                file_name_list_s = [item + "s" for item in file_name_list]
                file_name_list_es = [item + "es" for item in file_name_list]
                file_name_list_ies = [s[:-1] + "ies" for s in file_name_list]
                file_name_list_ves = [s[:-1] + "ves" for s in file_name_list]

                First_three_approximate_words = []

                for word in potential_similar_words_t100:
                    if word not in file_name_list and word not in file_name_list_s and word not in file_name_list_es and word not in file_name_list_ies and word not in file_name_list_ves and word not in synonym_sets_list:
                        # word = "person"
                        First_three_approximate_words.append(word)
                    if len(First_three_approximate_words) >= 3:
                        break

                file_names = ""
                for fileName in file_name_list:
                    file_names = file_names + "," + fileName
                if len(First_three_approximate_words) != 3:
                    similar_word = 'dog'
                    print(file_names + ":didn t find similar_word")
                else:
                    print(file_names + ":similar_word is", First_three_approximate_words[0])

                # # 随机产生一个
                # randomidx = random.randint(0, len(potential_similar_words_embedding))
                # similar_word = potential_similar_words_embedding_keys[randomidx]

                if len(First_three_approximate_words) == 3:
                    similar_word_text = tokenizer([First_three_approximate_words[0], First_three_approximate_words[0],
                                                   First_three_approximate_words[0]]).to(device)
                else:
                    similar_word_text = tokenizer([similar_word, similar_word, similar_word]).to(device)

                ima_p = os.path.join(folder_out, str(idx) + "_att_alpha_" + str(alpha) + "_update_set_" + str(update_set) + "_to_" + First_three_approximate_words[
                    0] + "_image2text_" + ".jpg")  # TODO 存储名称更换

                # print("folder_out", folder_out)
                # input()

                for epoch in tqdm(range(200)):
                    total_loss = 0
                    optimizer.zero_grad()
                    noisy_image = normalize(image_d + noise_param * alpha, image_mean, image_std)
                    image_features = model.encode_image(noisy_image)
                    # text_features = model.encode_text(text)

                    text_features = model.encode_text(similar_word_text)  # TODO 更换为筛选word

                    image_features = torch.div(image_features.clone(), image_features.norm(dim=-1, keepdim=True))
                    text_features /= text_features.norm(dim=-1, keepdim=True)
                    labels = torch.arange(image_features.shape[0], device=device, dtype=torch.long)
                    logits_per_image = image_features @ text_features.T
                    logits_per_text = text_features @ image_features.T
                    logits_per_image = F.softmax(logits_per_image, dim=-1)
                    logits_per_text = F.softmax(logits_per_text.T, dim=-1)

                    # match_loss = -torch.sum(torch.log2(1 - logits_per_image[:, 2]) + torch.log2(1 -logits_per_text[:,2]))/2
                    match_loss = (
                                         F.cross_entropy(logits_per_image, torch.tensor([1], device=device, dtype=torch.long)) +
                                         F.cross_entropy(logits_per_text, torch.tensor([1], device=device, dtype=torch.long))
                                 ) / 2
                    
                    print("match_loss", match_loss)

                    # loss_list.append(match_loss.to("cpu").detach().numpy())
                    total_loss += match_loss
                    m_loss = nn.MSELoss()(noisy_image, image)

                    # total_loss += m_loss
                    noise_loss = torch.mean(torch.square(noise_param))
                    # total_loss += noise_loss
                    match_loss.backward(retain_graph=True)
                    if epoch % 500 == 0:
                        print(
                            f"""total_loss: {total_loss}; match_loss: {match_loss}; noise_loss: {noise_loss}; m_loss: {m_loss}""")
                    noise_param.data = (noise_param.data - update_set / 255 * noise_param.grad.detach().sign()).clamp(-64 / 255,
                                                                                                              64 / 255)
                    noise_param.data = (noise_param.data + image_d.data).clamp(0, 1) - image_d.data
                    noise_param.grad.zero_()
                    model.zero_grad()

                # # plot loss curve
                # plt.plot(loss_list)
                # plt.title('optimizing noise for co-occurrence')
                # plt.xlabel('Epoch')
                # plt.ylabel('Loss')
                # plt.savefig(os.path.join(folder_out, str(idx) + "_att_alpha_" + str(alpha) + "_update_set_" + str(update_set) + "_to_" + First_three_approximate_words[0] + "_image2text_" + "_loss_curve.jpg"))
                utils.save_image((image_d + noise_param * alpha).squeeze(0).detach().cpu(), ima_p)


# img = "/home/czr/MMHallucination/co-occurrence/examples/image-to-text/high_occurrence/object existence/COCO_val2014_000000215867.jpg"
# img = "/home/czr/MMHallucination/co-occurrence/dataset/image_to_text/high_cooc/object_existence/normal/bench_person/COCO_val2014_000000077296.jpg_inpainted.jpg"
img = "/home/czr/MMHallucination/detr/test/demo.jpg"
victim_word = ["bench"]
misleading_attack(victim_word, img)