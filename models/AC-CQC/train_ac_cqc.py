"""
Training script for AC-CQC with text-guided attention.

Usage examples:
    # Fitzpatrick17k
    Run this code as: 

        python train_ac_cqc.py \
        --dataset_name fitzpatrick \
        --model_name ac_cqc \
        --metadata_csv /path/to/metadata.csv \
        --emb_orig /path/to/embeddings/original_text_emb.npy \
        --emb_cf /path/to/embeddings/counterfactual_text_emb.npy \
        --emb_names /path/to/embeddings/emb_names.npy \
        --holdout random_holdout \
        --n_epochs 20 \
        --seed 64

    # DDI
        python train_ac_cqc.py \
        --dataset_name ddi \
        --model_name ac_cqc \
        --metadata_csv /path/to/metadata.csv \
        --emb_orig /path/to/embeddings/original_text_emb.npy \
        --emb_cf /path/to/embeddings/counterfactual_text_emb.npy \
        --emb_names /path/to/embeddings/emb_names.npy \
        --holdout random_holdout \
        --n_epochs 15 \
        --seed 64
"""

from __future__ import print_function, division
from sklearn.decomposition import TruncatedSVD
import torch
from torchvision import transforms, models
import pandas as pd
import numpy as np
import os
import skimage
from skimage import io
import warnings
import torch.nn as nn
import torch.optim as optim
from torch.utils.data.sampler import WeightedRandomSampler
from torch.optim import lr_scheduler
import time
import copy
import random
import sys
import cv2
from tqdm import tqdm
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import balanced_accuracy_score
from torch.utils.tensorboard import SummaryWriter


from new_network_ac_cqc_dd import *     #separate for ac-cqc and ac-cqc-dd

warnings.filterwarnings("ignore")

import argparse
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader


import os


def get_output_path(seed: int, holdout_set_name: str, base_folder: str = "Results_ac_cqc_acc") -> str:
    # Construct subfolder name like S36
    seed_folder = f"S{seed}"
    
    # Full path: Results_ac_cqc/holdout_set_name/S36
    output_path = os.path.join(base_folder, holdout_set_name, seed_folder)

    # Create the directory if it doesn't exist
    os.makedirs(output_path, exist_ok=True)

    return output_path


SEED_MAP = {
    62: 4040,
    64: 4242,
    66: 4444,
    68: 4646,
    70: 4848
}

class CounterfactualEmbeddingBank:
    def __init__(self, orig_path, cf_path, names_path, device):
        """
        Loads original and counterfactual embeddings.
        """
        print(f"Loading Original Embeddings from {orig_path}...")
        print(f"Loading Counterfactual Embeddings from {cf_path}...")
        print(f"Loading Names/Hashers from {names_path}...")

        # Load data
        self.orig_data = np.load(orig_path)
        self.cf_data = np.load(cf_path)
        self.names = np.load(names_path)

        # Validation
        if not (len(self.orig_data) == len(self.cf_data) == len(self.names)):
            raise ValueError(
                f"Shape Mismatch! "
                f"Original: {self.orig_data.shape}, "
                f"Counterfactual: {self.cf_data.shape}, "
                f"Names: {self.names.shape}"
            )

        # Convert to tensors
        tensor_orig = torch.from_numpy(self.orig_data).float()
        tensor_cf = torch.from_numpy(self.cf_data).float()

        # Create lookup dictionary
        self.embedding_map = {}
        for i, name in enumerate(self.names):
            clean_name = str(name).strip()
            self.embedding_map[clean_name] = {
                'original': tensor_orig[i],
                'counterfactual': tensor_cf[i]
            }

        self.device = device
        self.embedding_dim = self.orig_data.shape[1]

        print(f"Counterfactual Bank Initialized. Mapped {len(self.embedding_map)} items.")

    def get_batch(self, hashers):
        """
        Retrieves original and counterfactual embeddings for a batch.
        """
        batch_orig = []
        batch_cf = []

        for h in hashers:
            h_str = str(h).strip()

            if h_str in self.embedding_map:
                batch_orig.append(self.embedding_map[h_str]['original'])
                batch_cf.append(self.embedding_map[h_str]['counterfactual'])
            else:
                # Fallback for missing embeddings
                # print(f"Warning: Missing embedding for {h_str}")

                raise KeyError(f"Missing embedding for {h_str}. Check your preprocessing.")
                

        # Stack and move to device
        text_orig = torch.stack(batch_orig).to(self.device)
        text_cf = torch.stack(batch_cf).to(self.device)

        return text_orig, text_cf


    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Load image
        from PIL import Image
        image = Image.open(self.image_paths[idx]).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return {
            'image': image,
            'label': torch.tensor(self.labels[idx], dtype=torch.long),
            'skin_tone': torch.tensor(self.skin_tones[idx], dtype=torch.long),
            'text_emb_true': torch.tensor(self.embeddings_true[idx], dtype=torch.float32),
            'text_emb_cf': torch.tensor(self.embeddings_cf[idx], dtype=torch.float32)
        }


def flatten(list_of_lists):
    if len(list_of_lists) == 0:
        return list_of_lists
    if isinstance(list_of_lists[0], list):
        return flatten(list_of_lists[0]) + flatten(list_of_lists[1:])
    return list_of_lists[:1] + flatten(list_of_lists[1:])



def current_lambda(epoch, max_lambda=0.2):
    if epoch < 2:
        # Short warmup
        return 0.0
    elif epoch < 8:
        # Smooth ramp from 0 → max_lambda
        return max_lambda * (epoch - 2) / 6.0
    else:
        # Full fairness
        return max_lambda


def train_model(label, dataloaders, device, dataset_sizes, model,
                criterion, optimizer, scheduler, num_epochs, bank, alpha=1.0):
    
    print('hyper-parameters alpha: {} '.format(alpha))
    since = time.time()
    training_results = []
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    best_loss = 0.0
    train_step = 0 # for tensorboard
    leading_epoch = 0  # record best model epoch
    lambda_inf = 0.3


    
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        lambda_loss= current_lambda(epoch)



        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
                # scheduler.step()
                tr_phase = True
            else:
                model.eval()   # Set model to evaluate mode
                tr_phase = False

            running_loss = 0.0
            running_corrects = 0.0
            running_balanced_acc_sum = 0.0
            # running_total = 0
            print(phase)
            # Iterate over data.
            loop = tqdm(dataloaders[phase], leave=True, desc=f" {phase}-ing Epoch {epoch + 1}/{num_epochs}")
            for n_iter, batch in enumerate(loop):

                inputs = batch["image"].to(device).float()
                label_c = torch.from_numpy(np.asarray(batch[label])).to(device)                 # label_condition
                label_t = torch.from_numpy(np.asarray(batch['fitzpatrick']-1)).to(device)       # label_type

                # Fetch pair of Embeddings ---
                batch_hashers = batch["hasher"] 
                text_true, text_cf = bank.get_batch(batch_hashers)

                optimizer.zero_grad()

                # ==========================================
                # TRAINING PHASE (Use Text + Fairness)
                # ==========================================
                if phase == 'train':

                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == 'train'):
                        inputs = inputs.float()  # ADDED AS A FIX

                        # Expects Network.forward(x, text_true, text_cf)
                        output = model(inputs, text_true, text_cf)
                        

                        # --- AC-CQC Loss ---
                        l_gqvk, loss_dict = ac_cqc_loss(
                            model_outputs=output,
                            lambda_fair=lambda_loss,
                            metric='l1',
                            device=device
                        )
                        
                        if torch.isnan(l_gqvk):
                            print(f"WARNING: NaN in GQVK loss at iter {n_iter}. Skipping batch.")
                            continue

                        # Standard Classification Losses
                        # output[1] is skin tone logits
                        _, preds = torch.max(output[0], 1)  # branch 1 get condition prediction

                        # output[0] = Text-Guided Diagnosis
                        # output[13] = Inference-Guided Diagnosis (The Global Query)
                        loss_cls_text = criterion[0](output[0], label_c) 
                        loss_cls_inf  = criterion[0](output[13], label_c)

                        loss_confusion = criterion[1](output[1], label_t)
                        loss_skin_detach = criterion[0](output[2], label_t) # Detached branch passed through branch 2
                        
                        # Total Loss 
                        # Weights: 1.0 Class + alpha Skin + 1.0 GQVK(Fairness+NCE)

                        loss = loss_cls_text + lambda_inf*loss_cls_inf + alpha*loss_confusion + loss_skin_detach + l_gqvk
                        
                        '''
                        # Total loss:
                        # 1) Text-guided classification (semantic grounding)
                        # 2) Inference-query classification (deployment alignment)
                        # 3) Skin-tone confusion (attribute suppression)
                        # 4) Detached skin supervision (stability)
                        # 5) AC-CQC fairness + contrastive regularization
                        '''
                        
                        if phase == 'train':
                            loss.backward()        
                            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Set max_norm as desired
                            #print('output grads',output[-1].grad)
                            optimizer.step()

                        attn_text = output[3]
                        attn_cf=output[8]


                        # statistics
                    # tensorboard
                    if phase == 'train':
                        writer.add_scalar('Loss/'+phase, loss.item(), train_step)
                        writer.add_scalar('Loss/'+phase+'loss_cls_text', loss_cls_text.item(), train_step)
                        writer.add_scalar('Loss/'+phase+'loss_cls_inf', loss_cls_inf.item(), train_step)
                        writer.add_scalar('Loss/'+phase+'loss_confusion', loss_confusion.item(), train_step)
                        writer.add_scalar('Loss/'+phase+'loss_skin_detach', loss_skin_detach.item(), train_step)
                        writer.add_scalar('Loss/'+phase+'total_gqvk_loss', l_gqvk.item(), train_step)
                        writer.add_scalar('Loss/' + phase + '/contrastive_loss',loss_dict['contrastive'], train_step)
                        writer.add_scalar('Loss/' + phase + '/fairness_loss',loss_dict['fairness'], train_step)

                        writer.add_scalar('Accuracy/'+phase, (torch.sum(preds == label_c.data)).item()/inputs.size(0), train_step)
                        writer.add_scalar('Balanced-Accuracy/'+phase, balanced_accuracy_score(label_c.data.cpu(), preds.cpu()), train_step)
                        train_step += 1

                    if phase == 'train' and n_iter % 500 == 0 and attn_cf is not None:
                        grid = int(attn_text.shape[-1]**0.5)
                        if grid * grid == attn_text.shape[-1]:
                            map_t = attn_text[0].reshape(grid, grid).detach().cpu().numpy()
                            map_c = attn_cf[0].reshape(grid, grid).detach().cpu().numpy()
                            diff = np.abs(map_t - map_c)
                            writer.add_image('Attn/Text', map_t, train_step, dataformats='HW')
                            writer.add_image('Attn/Counterfactual', map_c, train_step, dataformats='HW')
                            writer.add_image('Attn/Diff', diff, train_step, dataformats='HW')

                else: 
                    with torch.no_grad():
                        # 1. Forward WITHOUT Text (Simulates Deployment)
                        # The modified Network.forward returns [logits] at index 0 in this mode
                        output = model(inputs) 
                        
                        # 2. Calculate Loss (Only Task Loss)
                        # output[0] is now the Inference Query Logits
                        loss = criterion[0](output[0], label_c)
                        
                        _, preds = torch.max(output[0], 1)

                # -------------------------
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == label_c.data)
                running_balanced_acc_sum += balanced_accuracy_score(label_c.data.cpu(), preds.cpu())*inputs.size(0)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects / dataset_sizes[phase]
            epoch_balanced_acc = running_balanced_acc_sum / dataset_sizes[phase]

            print("Accuracy: {}/{}".format(running_corrects,
                                           dataset_sizes[phase]))
            print('{} Loss: {:.4f} Acc: {:.4f} Balanced-Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc, epoch_balanced_acc))
            # tensorboard 
            writer.add_scalar('lr/'+phase, scheduler.get_last_lr()[0], epoch)

            if phase == 'train':
                scheduler.step()
                
            if phase == 'val':
                writer.add_scalar('Loss/'+phase, epoch_loss, epoch)
                writer.add_scalar('Accuracy/'+phase, epoch_acc, epoch)
                writer.add_scalar('Balanced-Accuracy/'+phase, epoch_balanced_acc, epoch)
            # ---------------------   
            # Saving based on accuracy 
            # # ---------------------
            training_results.append([phase, epoch, epoch_loss, epoch_acc.item(), epoch_balanced_acc])
            if epoch > 0:
                 if phase == 'val' and epoch_acc > best_acc:
                     print("New leading accuracy: {}".format(epoch_acc))
                     best_acc = epoch_acc
                     leading_epoch = epoch
                     best_model_wts = copy.deepcopy(model.state_dict())

            elif phase == 'val':
                 best_acc = epoch_acc

            # ---------------------   
            # Saving based on balanced accuracy 
            # ---------------------
            
            # training_results.append([phase, epoch, epoch_loss, epoch_acc.item(), epoch_balanced_acc])
            # if epoch > 0:
         
            #    if phase == 'val' and epoch_balanced_acc > best_acc:
            #        print("New leading accuracy: {}".format(epoch_balanced_acc))
            #        best_acc = epoch_balanced_acc
            #        leading_epoch = epoch
            #        best_model_wts = copy.deepcopy(model.state_dict())

            # elif phase == 'val':
            #    best_acc = epoch_balanced_acc

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))
    print('Best model epoch:', leading_epoch)
    model.load_state_dict(best_model_wts)
    training_results = pd.DataFrame(training_results)
    training_results.columns = ["phase", "epoch", "loss", "accuracy", "balanced-accuracy"]

    return model, training_results


class SkinDataset():
    def __init__(self, dataset_name, csv_file, root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.df = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        self.dataset_name = dataset_name

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        if self.dataset_name == 'ddi':
            img_name = os.path.join(self.root_dir,
                                str(self.df.loc[self.df.index[idx], 'hasher']))
            image = cv2.imread(img_name)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            img_name = os.path.join(self.root_dir,
                                str(self.df.loc[self.df.index[idx], 'hasher']))+'.jpg'
            image = io.imread(img_name)

        if(len(image.shape) < 3):
            image = skimage.color.gray2rgb(image)

        hasher = self.df.loc[self.df.index[idx], 'hasher']
        high = self.df.loc[self.df.index[idx], 'high']
        # mid = self.df.loc[self.df.index[idx], 'mid']
        low = self.df.loc[self.df.index[idx], 'low']
        fitzpatrick = self.df.loc[self.df.index[idx], 'fitzpatrick_scale']

        if self.dataset_name == 'fitzpatrick':
            mid = self.df.loc[self.df.index[idx], 'mid'] 
        else:
            mid = 0
        if self.dataset_name == 'fitzpatrick':
            mid = self.df.loc[self.df.index[idx], 'mid'] 
            partition = self.df.loc[self.df.index[idx], 'label'] 
        else:
            mid = 0
            partition = self.df.loc[self.df.index[idx], 'disease'] 

        if self.transform:
            image = self.transform(image)
        sample = {
                    'image': image,
                    'high': high,
                    'mid': mid,
                    'low': low,
                    'hasher': hasher,
                    'fitzpatrick': fitzpatrick,
                    "partition": partition,
                    "idx": idx
                }
        return sample



DATASET_IMAGE_ROOTS = {
    'fitzpatrick': '/data/fitzpatrick17k/images',
    'ddi': '/data/ddi/images'
}



def custom_load(
        batch_size=32,
        num_workers=10,
        train_dir='',
        val_dir='',
        label = 'low',
        dataset_name = 'fitzpatrick',
        ):
    
    if dataset_name not in DATASET_IMAGE_ROOTS:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    image_dir = DATASET_IMAGE_ROOTS[dataset_name]

    val = pd.read_csv(val_dir)
    train = pd.read_csv(train_dir)
    class_sample_count = np.array(train[label].value_counts().sort_index())
    weight = 1. / class_sample_count
    samples_weight = np.array([weight[t] for t in train[label]])

    samples_weight = torch.from_numpy(samples_weight)
    sampler = WeightedRandomSampler(
        samples_weight.type('torch.DoubleTensor'),
        len(samples_weight),
        replacement=True)
    dataset_sizes = {"train": train.shape[0], "val": val.shape[0]}
    transformed_train = SkinDataset(
        dataset_name = dataset_name,
        csv_file=train_dir,
        root_dir=image_dir,
        transform=transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomResizedCrop(size=256, scale=(0.8, 1.0)),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(),
            transforms.RandomHorizontalFlip(),
            transforms.CenterCrop(size=224),  # Image net standards
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                [0.229, 0.224, 0.225])
            ])
        )
    transformed_test = SkinDataset(
        dataset_name = dataset_name,
        csv_file=val_dir,
        root_dir=image_dir,
        transform=transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(size=256),
            transforms.CenterCrop(size=224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        )
    dataloaders = {
        "train": torch.utils.data.DataLoader(
            transformed_train,
            batch_size=batch_size,
            sampler=sampler,
            # drop_last = True,
            #shuffle=True,
            num_workers=num_workers),
        "val": torch.utils.data.DataLoader(
            transformed_test,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers)
        }
    return dataloaders, dataset_sizes




if __name__ == '__main__':
    parser = argparse.ArgumentParser("AC-CQC Training")

    # Core experiment
    parser.add_argument('--dataset_name', type=str, choices=['fitzpatrick', 'ddi'], required=True)
    parser.add_argument('--holdout', type=str, default='br',
                        choices=['random_holdout', 'a12', 'a34', 'a56', 'dermaamin', 'br'])
    parser.add_argument('--n_epochs', type=int, default=20)
    parser.add_argument('--model_name', type=str, required=True, help='Name of the model')
    parser.add_argument('--seed', type=int, default=64)

    # Paths
    parser.add_argument('--metadata_csv', type=str, required=True,
                        help='Path to dataset metadata CSV')
    parser.add_argument('--emb_orig', type=str, required=True,
                        help='Path to original text embeddings (.npy)')
    parser.add_argument('--emb_cf', type=str, required=True,
                        help='Path to counterfactual text embeddings (.npy)')
    parser.add_argument('--emb_names', type=str, required=True,
                        help='Path to embedding hasher names (.npy)')

    # Debug
    parser.add_argument('--dev_mode', action='store_true',
                        help='Run on a small subset for debugging')

		
    args = parser.parse_args()

    
    seed = args.seed
    dev_mode = args.dev_mode
    n_epochs = args.n_epochs
    dataset_name = args.dataset_name
    model_name = args.model_name
    holdout_set = args.holdout
    seed1 = SEED_MAP.get(seed, seed)

    

    print("CUDA is available: {} \n".format(torch.cuda.is_available()))
    print("Starting... \n")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if dev_mode:
        if dataset_name == 'ddi':
            df = pd.read_csv(args.metadata_csv).sample(300, random_state=seed)
        else:
            df = pd.read_csv(args.metadata_csv).sample(2000, random_state=seed)
    else:
        df = pd.read_csv(args.metadata_csv)

            
    path_op = get_output_path(seed, holdout_set)

    temp_df = df.copy()

    if holdout_set == "expert_select":
        df2 = df
        train = df2[df2.qc.isnull()]
        test = df2[df2.qc=="1 Diagnostic"]  
    elif holdout_set == "random_holdout":
        if dataset_name == 'ddi':
            train, test, y_train, y_test = train_test_split(
                                                    df,
                                                    df['high'],
                                                    test_size=0.2,
                                                    random_state=seed,
                )
        else:
            train, test, y_train, y_test = train_test_split(
                                                    df,
                                                    df['low'],
                                                    test_size=0.2,
                                                    random_state=seed,
                                                    stratify=df['low']) # 
            
    elif holdout_set == "a12":
        train = temp_df[(temp_df.fitzpatrick_scale==1)|(temp_df.fitzpatrick_scale==2)]
        test = temp_df[(temp_df.fitzpatrick_scale!=1)&(temp_df.fitzpatrick_scale!=2)]
        combo = set(train.label.unique()) & set(test.label.unique())
        print(combo)
        train = train[train.label.isin(combo)].reset_index()
        test = test[test.label.isin(combo)].reset_index()
        train["low"] = train['label'].astype('category').cat.codes
        test["low"] = test['label'].astype('category').cat.codes
    elif holdout_set == "a34":
        train = temp_df[(temp_df.fitzpatrick_scale==3)|(temp_df.fitzpatrick_scale==4)]
        test = temp_df[(temp_df.fitzpatrick_scale!=3)&(temp_df.fitzpatrick_scale!=4)]
        combo = set(train.label.unique()) & set(test.label.unique())
        train = train[train.label.isin(combo)].reset_index()
        test = test[test.label.isin(combo)].reset_index()
        train["low"] = train['label'].astype('category').cat.codes
        test["low"] = test['label'].astype('category').cat.codes
    elif holdout_set == "a56":
        train = temp_df[(temp_df.fitzpatrick_scale==5)|(temp_df.fitzpatrick_scale==6)]
        test = temp_df[(temp_df.fitzpatrick_scale!=5)&(temp_df.fitzpatrick_scale!=6)]
        combo = set(train.label.unique()) & set(test.label.unique())
        train = train[train.label.isin(combo)].reset_index()
        test = test[test.label.isin(combo)].reset_index()
        train["low"] = train['label'].astype('category').cat.codes
        test["low"] = test['label'].astype('category').cat.codes


    train_path = path_op +'/'+"temp_train_{}_{}_{}_{}.csv".format(model_name, n_epochs,holdout_set, seed)
    test_path = path_op +'/'+"temp_test_{}_{}_{}_{}.csv".format(model_name, n_epochs, holdout_set, seed)
    train.to_csv(train_path, index=False)
    test.to_csv(test_path, index=False)
        
    level = "high" #9label


    emb_orig = args.emb_orig
    emb_cf   = args.emb_cf
    names_file = args.emb_names


    bank = CounterfactualEmbeddingBank(emb_orig, emb_cf, names_file, device)

    for indexer, label in enumerate([level]):
        # tensorboard
        writer = SummaryWriter(comment="logs_{}_{}_{}_{}_{}.pth".format(model_name, n_epochs, label, holdout_set, seed))
        print(label)
        weights = np.array(max(train[label].value_counts())/train[label].value_counts().sort_index())
        label_codes = sorted(list(train[label].unique()))
        dataloaders, dataset_sizes = custom_load(
                32,
                10,
                "{}".format(train_path),
                "{}".format(test_path),
                label = label,
                dataset_name = dataset_name)
        print(dataset_sizes)

        label_codes = sorted(list(train[level].unique()))
            

        model_ft = GQVKNetwork( [len(label_codes), 6])
        model_ft = model_ft.to(device)
        model_ft = nn.DataParallel(model_ft)
                                                                                                                                                                                 
         
        total_params = sum(p.numel() for p in model_ft.module.image_encoder.parameters())
        print('{} total parameters'.format(total_params))
        i = 0
        for p in model_ft.module.image_encoder.parameters():
            if(i>=50):
                p.requires_grad=True
            else:   p.requires_grad=False
            i+=1
        print('i',i)
        total_trainable_params = sum(
            p.numel() for p in model_ft.module.image_encoder.parameters() if p.requires_grad)
        print('{} total trainable parameters'.format(total_trainable_params))
        k = 0


        class_weights = torch.FloatTensor(weights).cuda()

        criterion = [nn.CrossEntropyLoss(), Confusion_Loss()]
        optimizer_ft = optim.Adam(list(model_ft.parameters()) , 0.0001) #+list(bert_model.parameters())
        exp_lr_scheduler = lr_scheduler.StepLR(
        optimizer_ft,
        step_size=2,
        gamma=0.9)

        print("\nTraining classifier for {}........ \n".format(label))
        print("....... processing ........ \n")
        model_ft, training_results = train_model(
            label,
            dataloaders, device,
            dataset_sizes, model_ft,
            criterion, optimizer_ft,
            exp_lr_scheduler, 
            n_epochs, bank, 
            )
        print("Training Complete")
            
        torch.save(model_ft.state_dict(), path_op +'/'+"model_path_{}_{}_{}_{}_{}_{}.pth".format(model_name, n_epochs, label, holdout_set, dataset_name,seed))
        torch.save(model_ft, path_op +'/'+"model_path_{}_{}_{}_{}_{}_{}.pt".format(model_name, n_epochs, label, holdout_set, dataset_name, seed))
        print("gold")
        training_results.to_csv(path_op +'/'+"training_{}_{}_{}_{}_{}_{}.csv".format(model_name, n_epochs, label, holdout_set, dataset_name, seed))

        model = model_ft.eval()
        loader = dataloaders["val"]
        prediction_list = []
        fitzpatrick_list = []
        hasher_list = []
        labels_list = []
        p_list = []
        topk_p = []
        topk_n = []
        d1 = []
        d2 = []
        d3 = []
        p1 = []
        p2 = []
        p3 = []
        with torch.no_grad():
            running_corrects = 0
            running_balanced_acc_sum  = 0
            total = 0
            for i, batch in enumerate(dataloaders['val']):
                inputs = batch["image"].to(device)
                classes = batch[label].to(device)
                fitzpatrick = batch["fitzpatrick"]  # skin type
                hasher = batch["hasher"]
                    
                output = model(inputs)

                # outputs = model(inputs.float())  # (batchsize, classes num)
                probability = torch.nn.functional.softmax(output[0], dim=1)

                ppp, preds = torch.topk(probability, 1) #topk values, topk indices
                if label == "low":
                    _, preds5 = torch.topk(probability, 3)  # topk values, topk indices
                    # topk_p.append(np.exp(_.cpu()).tolist())
                    topk_p.append((_.cpu()).tolist())
                    topk_n.append(preds5.cpu().tolist())
                running_balanced_acc_sum += balanced_accuracy_score(classes.data.cpu(), preds.reshape(-1).cpu()) * inputs.shape[0]
                running_corrects += torch.sum(preds.reshape(-1) == classes.data)
                p_list.append(ppp.cpu().tolist())
                prediction_list.append(preds.cpu().tolist())
                labels_list.append(classes.tolist())
                fitzpatrick_list.append(fitzpatrick.tolist())
                hasher_list.append(hasher)
                total += inputs.shape[0]
            acc = float(running_corrects)/float(dataset_sizes['val'])
            balanced_acc = float(running_balanced_acc_sum)/float(dataset_sizes['val'])
        
        
        
        if label == "low":
            for j in topk_n: # each sample
                for i in j:  # in k
                    d1.append(i[0])
                    d2.append(i[1])
                    d3.append(i[2])
            for j in topk_p:
                for i in j:
                    # print(i)
                    p1.append(i[0])
                    p2.append(i[1])
                    p3.append(i[2])
            df_x=pd.DataFrame({
                                "hasher": flatten(hasher_list),
                                "label": flatten(labels_list),
                                "fitzpatrick": flatten(fitzpatrick_list),
                                "prediction_probability": flatten(p_list),
                                "prediction": flatten(prediction_list),
                                "d1": d1,
                                "d2": d2,
                                "d3": d3,
                                "p1": p1,
                                "p2": p2,
                                "p3": p3})
        else:
            df_x=pd.DataFrame({
                                    "hasher": flatten(hasher_list),
                                    "label": flatten(labels_list),
                                    "fitzpatrick": flatten(fitzpatrick_list),
                                    "prediction_probability": flatten(p_list),
                                    "prediction": flatten(prediction_list)})
        df_x.to_csv(path_op +'/'+"results_{}_{}_{}_{}_{}_{}.csv".format(model_name, n_epochs, label, holdout_set, dataset_name, seed),
                            index=False)
        print("\n Accuracy: {}  Balanced Accuracy: {} \n".format(acc, balanced_acc))
    print("done")
    # writer.close()
