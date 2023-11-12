import os
import math
import numpy as np
import torch
import clip
from tqdm import tqdm
from torch.utils.data import DataLoader
import data_utils
from PIL import Image, ImageOps
from torchvision import transforms
from matplotlib import pyplot as plt
import cv2
import functools 

PM_SUFFIX = {"max":"_max", "avg":""}

def get_activation(outputs, mode):
    '''
    mode: how to pool activations: one of avg, max
    for fc or ViT neurons does no pooling
    '''
    if mode=='avg':
        def hook(model, input, output):
            if len(output.shape)==4: #CNN layers
                outputs.append(output.mean(dim=[2,3]).detach())
            elif len(output.shape)==3: #ViT
                outputs.append(output[:, 0].clone())
            elif len(output.shape)==2: #FC layers
                outputs.append(output.detach())
    elif mode=='max':
        def hook(model, input, output):
            if len(output.shape)==4: #CNN layers
                outputs.append(output.amax(dim=[2,3]).detach())
            elif len(output.shape)==3: #ViT
                outputs.append(output[:, 0].clone())
            elif len(output.shape)==2: #FC layers
                outputs.append(output.detach())
    return hook

def get_mean_activation(outputs):
    def hook(model, input, output):
        if len(output.shape)==4: #CNN layers
            outputs.append(output.detach())
        elif len(output.shape)==3: #ViT
            outputs.append(output[:, 0].clone())
        elif len(output.shape)==2: #FC layers
            outputs.append(output.detach())
    return hook
    
def get_save_names(clip_name, target_name, target_layer, d_probe, concept_set, pool_mode, save_dir, newSet = False):

    if newSet == False:
        target_save_name = "{}/{}_{}_{}{}.pt".format(save_dir, d_probe, target_name, target_layer,
                                                 PM_SUFFIX[pool_mode])
        clip_save_name = "{}/{}_{}.pt".format(save_dir, d_probe, clip_name.replace('/', ''))
        concept_set_name = (concept_set.split("/")[-1]).split(".")[0]
        text_save_name = "{}/{}_{}.pt".format(save_dir, concept_set_name, clip_name.replace('/', ''))
    else:
        target_save_name = "{}/{}_{}_{}{}_new.pt".format(save_dir, d_probe, target_name, target_layer,
                                                 PM_SUFFIX[pool_mode])
        clip_save_name = "{}/{}_{}_new.pt".format(save_dir, d_probe, clip_name.replace('/', ''))
        concept_set_name = (concept_set.split("/")[-1]).split(".")[0]
        text_save_name = "{}/{}_{}_new.pt".format(save_dir, concept_set_name, clip_name.replace('/', ''))
    
    return target_save_name, clip_save_name, text_save_name

def save_target_activations(target_model, dataset, save_name, target_layers = ["layer4"], batch_size = 1000,
                            device = "cuda", pool_mode='avg'):
    """
    save_name: save_file path, should include {} which will be formatted by layer names
    """
    _make_save_dir(save_name)
    save_names = {}    
    for target_layer in target_layers:
        save_names[target_layer] = save_name.format(target_layer)

    if _all_saved(save_names):
        return
    
    all_features = {target_layer:[] for target_layer in target_layers}
    
    hooks = {}
        
    for target_layer in target_layers:
        command = "target_model.{}.register_forward_hook(get_activation(all_features[target_layer], pool_mode))".format(target_layer)
        hooks[target_layer] = eval(command)
    
    with torch.no_grad():
        for images, labels in tqdm(DataLoader(dataset, batch_size, num_workers=8, pin_memory=True)):
            features = target_model(images.to(device))
    
    for target_layer in target_layers:
        torch.save(torch.cat(all_features[target_layer]), save_names[target_layer])
        hooks[target_layer].remove()
    #free memory
    del all_features
    torch.cuda.empty_cache()
    return


def save_clip_image_features(model, dataset, save_name, batch_size=1000 , device = "cuda"):
    _make_save_dir(save_name)
    all_features = []
    
    if os.path.exists(save_name):
        return
    
    save_dir = save_name[:save_name.rfind("/")]
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    with torch.no_grad():
        for images, labels in tqdm(DataLoader(dataset, batch_size, num_workers=8, pin_memory=True)):
            features = model.encode_image(images.to(device))
            all_features.append(features)
    torch.save(torch.cat(all_features), save_name)
    #free memory
    del all_features
    torch.cuda.empty_cache()
    return

def save_clip_text_features(model, text, save_name, batch_size=1000):
    if os.path.exists(save_name):
        return
    _make_save_dir(save_name)
    text_features = []
    with torch.no_grad():
        for i in tqdm(range(math.ceil(len(text)/batch_size))):
            text_features.append(model.encode_text(text[batch_size*i:batch_size*(i+1)]))
    text_features = torch.cat(text_features, dim=0)
    torch.save(text_features, save_name)
    del text_features
    torch.cuda.empty_cache()
    return

def IoU(box, check_box):
    x1,y1,x2,y2 = box;
    
    box_area = abs(x2-x1) * abs(y2-y1)
    
    cx1, cy1, cx2, cy2 = check_box
    check_area = abs(cx2-cx1) * abs(cy2-cy1)

    overlap_x = max(0,min(cx2, x2) - max(cx1, x1))
    overlap_y = max(0,min(cy2, y2) - max(cy1, y1))

    overlap = overlap_x * overlap_y
    union = box_area + check_area - overlap
    
    return overlap / union
        
def compare(box1, box2):
    x1,y1,x2,y2 = box1;
    box1_area = abs(x2-x1) * abs(y2-y1)
    cx1, cy1, cx2, cy2 = box2
    box2_area = abs(cx2-cx1) * abs(cy2-cy1)
    
    if box1_area < box2_area:
        return 1
    elif box1_area > box2_area:
        return -1
    else:
        return 0

def get_attention_crops(target_name, images, neuron_id, num_crops_per_image = 4, target_layers = ["layer4"], batch_size = 1000,
                            device = "cuda", pool_mode='avg', return_bounding_box = False):
    
    target_model, preprocess = data_utils.get_target_model(target_name, device)
    all_features = {target_layer:[] for target_layer in target_layers}
    hooks = {}
    
    transform = transforms.ToPILImage()
    
    for target_layer in target_layers:
        command = "target_model.{}.register_forward_hook(get_mean_activation(all_features[target_layer]))".format(target_layer)
        hooks[target_layer] = eval(command)
        
    with torch.no_grad():
        for image in images:
            features = target_model(preprocess(image).unsqueeze(0).to(device))
    
    all_heatmaps = {target_layer:[] for target_layer in target_layers}
    for target_layer in target_layers:
        all_features[target_layer] = torch.cat(all_features[target_layer])
        hooks[target_layer].remove()
        
        for i in range(len(all_features[target_layer])):
            if target_layer != 'fc' and target_layer != 'encoder':
                print(all_features[target_layer].shape)
                heatmap = transform(all_features[target_layer][i][neuron_id])
            elif target_layer == 'encoder':
                unflattend_img = torch.unflatten(all_features[target_layer][i],0,(16,16,3))
                unflattend_img = torch.permute(unflattend_img, (2,0,1))
                heatmap = ImageOps.grayscale(transform(unflattend_img))
            else:
                heatmap = transform(all_features[target_layer][i])
            heatmap = heatmap.resize([375,375])
            heatmap = np.array(heatmap)
            # print("Heat map shape: {}".format(heatmap.shape))
            all_heatmaps[target_layer].append(heatmap)
    
    all_image_crops = [];
    all_bb_box = {layer : {i:[] for i in range(len(all_heatmaps[target_layer]))} for layer in target_layers}
    for target_layer in target_layers:
        for i, heatmap in enumerate(all_heatmaps[target_layer]): 
            thresh = cv2.threshold(heatmap, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
            
            bb_cor = []
            # Find contours
            cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cnts = cnts[0] if len(cnts) == 2 else cnts[1]
            for c in cnts:
                x,y,w,h = cv2.boundingRect(c)
                box = (x, y, x + w, y + h)
                bb_cor.append(box)
                
            bb_cor = sorted(bb_cor, key=functools.cmp_to_key(compare))
            
            cropped_bb = []
            for box in bb_cor:
                if len(cropped_bb) == num_crops_per_image:
                    break
                p = 0
                good_to_add = True
                while p < len(cropped_bb):
                    if IoU(box, cropped_bb[p]) <= 0.5: 
                        p += 1
                    else:
                        good_to_add = False
                        break
                if good_to_add and IoU(box,(0,0,375,375)) < 0.8:
                    cropped_img = images[i].crop(box)
                    cropped_img = cropped_img.resize([375,375])
                    all_image_crops.append(cropped_img)
                    cropped_bb.append(box)
            all_bb_box[target_layer][i] = cropped_bb
            
    if return_bounding_box == True:
        return all_bb_box[target_layers[0]], all_image_crops
    else:
        del all_bb_box
        return all_image_crops
       

def get_target_activations(target_name, images, target_layers = ["layer4"], batch_size = 1000,
                            device = "cuda", pool_mode='avg'):
    
    target_model, preprocess = data_utils.get_target_model(target_name, device)
    
    all_features = {target_layer:[] for target_layer in target_layers}
    
    hooks = {}
    for target_layer in target_layers:
        command = "target_model.{}.register_forward_hook(get_activation(all_features[target_layer], pool_mode))".format(target_layer)
        hooks[target_layer] = eval(command)
    
    with torch.no_grad():
        for image in images:
            features = target_model(preprocess(image).unsqueeze(0).to(device))
    
    for target_layer in target_layers:
        all_features[target_layer] = torch.cat(all_features[target_layer])
        hooks[target_layer].remove()
    return all_features[target_layers[0]]

def get_clip_image_features(model, preprocess, images, batch_size=1000, device = "cuda"):
    
    all_features = []
    with torch.no_grad():
        for image in images:
            features = model.encode_image(preprocess(image).unsqueeze(0).to(device))
            all_features.append(features)
    all_features = torch.cat(all_features)
    return all_features
    

def get_clip_text_features(model, text, batch_size=1000):
    """
    gets text features without saving, useful with dynamic concept sets
    """
    text_features = []
    with torch.no_grad():
        for i in range(math.ceil(len(text)/batch_size)):
            text_features.append(model.encode_text(text[batch_size*i:batch_size*(i+1)]))
    text_features = torch.cat(text_features, dim=0)
    return text_features

def save_activations(clip_name, target_name, target_layers, d_probe, 
                     concept_set, batch_size, device, pool_mode, save_dir):
    
    clip_model, clip_preprocess = clip.load(clip_name, device=device)
    target_model, target_preprocess = data_utils.get_target_model(target_name, device)
    #setup data
    data_c = data_utils.get_data(d_probe, clip_preprocess)
    data_t = data_utils.get_data(d_probe, target_preprocess)

    with open(concept_set, 'r') as f: 
        words = (f.read()).split('\n')
    #ignore empty lines
    words = [i for i in words if i!=""]
    
    text = clip.tokenize(["{}".format(word) for word in words]).to(device)
    
    save_names = get_save_names(clip_name = clip_name, target_name = target_name,
                                target_layer = '{}', d_probe = d_probe, concept_set = concept_set,
                                pool_mode=pool_mode, save_dir = save_dir)
    target_save_name, clip_save_name, text_save_name = save_names
    
    save_clip_text_features(clip_model, text, text_save_name, batch_size)
    save_clip_image_features(clip_model, data_c, clip_save_name, batch_size, device)
    save_target_activations(target_model, data_t, target_save_name, target_layers,
                            batch_size, device, pool_mode)
    return
    
def get_similarity_from_activations(target_save_name, clip_save_name, text_save_name, similarity_fn, 
                                   new_target_save_name = None, new_clip_save_name = None, return_target_feats=True, k = 100, device="cuda"):
    
    image_features = torch.load(clip_save_name, map_location='cpu').float()
    text_features = torch.load(text_save_name, map_location='cpu').float()

    with torch.no_grad():
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        clip_feats = (image_features @ text_features.T)
    del image_features, text_features
    torch.cuda.empty_cache()
    
    target_feats = torch.load(target_save_name, map_location='cpu')

    similarity = similarity_fn(clip_feats, target_feats, top_k=k, device=device)
    
    del clip_feats
    torch.cuda.empty_cache()
    
    if return_target_feats:
        return similarity, target_feats
    else:
        del target_feats
        torch.cuda.empty_cache()
        return similarity

def get_cos_similarity(preds, gt, clip_model, mpnet_model, device="cuda", batch_size=200):
    """
    preds: predicted concepts, list of strings
    gt: correct concepts, list of strings
    """
    pred_tokens = clip.tokenize(preds).to(device)
    gt_tokens = clip.tokenize(gt).to(device)
    pred_embeds = []
    gt_embeds = []

    #print(preds)
    with torch.no_grad():
        for i in range(math.ceil(len(pred_tokens)/batch_size)):
            pred_embeds.append(clip_model.encode_text(pred_tokens[batch_size*i:batch_size*(i+1)]))
            gt_embeds.append(clip_model.encode_text(gt_tokens[batch_size*i:batch_size*(i+1)]))

        pred_embeds = torch.cat(pred_embeds, dim=0)
        pred_embeds /= pred_embeds.norm(dim=-1, keepdim=True)
        gt_embeds = torch.cat(gt_embeds, dim=0)
        gt_embeds /= gt_embeds.norm(dim=-1, keepdim=True)

    #l2_norm_pred = torch.norm(pred_embeds-gt_embeds, dim=1)
    cos_sim_clip = torch.sum(pred_embeds*gt_embeds, dim=1)

    gt_embeds = mpnet_model.encode([gt_x for gt_x in gt])
    pred_embeds = mpnet_model.encode(preds)
    cos_sim_mpnet = np.sum(pred_embeds*gt_embeds, axis=1)

    return float(torch.mean(cos_sim_clip)), float(np.mean(cos_sim_mpnet))

def _all_saved(save_names):
    """
    save_names: {layer_name:save_path} dict
    Returns True if there is a file corresponding to each one of the values in save_names,
    else Returns False
    """
    for save_name in save_names.values():
        if not os.path.exists(save_name):
            return False
    return True

def _make_save_dir(save_name):
    """
    creates save directory if one does not exist
    save_name: full save path
    """
    save_dir = save_name[:save_name.rfind("/")]
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    return
