import os
import glob
import argparse
import numpy as np
from tqdm import tqdm

from PIL import Image

import torch
from torch.nn import functional as F
from torchvision import transforms as T
from torchvision.models import ViT_H_14_Weights, vit_h_14
from torchvision.models import inception_v3

from torchmetrics.functional import accuracy
# from torchmetrics.image.fid import FrechetInceptionDistance

from torcheval.metrics import FrechetInceptionDistance


class OUTPUT_Dataset:
    
    def __init__(self, path, name, n_way_processor, num_samples=5):
        # Set variable
        self.name        = name
        self.num_samples = num_samples+1 # because with ground truth

        # Set Paths
        self.path      = os.path.join(path, name)
        self.all_paths = glob.glob(os.path.join(self.path, '*'))

        #Set processor
        self.n_way_processor = n_way_processor
        self.fid_processor   = T.Compose([
            T.Resize((299, 299)),
            T.ToTensor(),
        ])

    def __len__(self):
        return int(len(self.all_paths)//self.num_samples)

    def __getitem__(self, idx):
        # If you want to apply this codes, you must be change below the naming rules
        names = [os.path.join(self.path, f"{self.name}{idx}-0-{i}.png") for i in range(self.num_samples)]
        
        # Pre-processing for Ground Truth Images
        gt_img     = Image.open(names[0]).convert('RGB')
        ga_gt      = self.n_way_processor(gt_img).unsqueeze(0)
        fid_gt     = self.fid_processor(gt_img).unsqueeze(0)
        
        # Pre=processing for Generated Sample Images
        ga_gen   = []
        fid_gen  = []
        for name in names[1:]:
            img = Image.open(name).convert('RGB')
            ga_gen.append(self.n_way_processor(img))
            fid_gen.append(self.fid_processor(img))
        ga_gen  = torch.stack(ga_gen)
        fid_gen = torch.stack(fid_gen) 

        return ga_gt, ga_gen, fid_gt, fid_gen


class Generated_Image_Tester():
    def __init__(self, path, name, device=None):
        #Set Device
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device

        # for GA Score
        weights = ViT_H_14_Weights.DEFAULT
        self.n_way_model = vit_h_14(weights=weights).to(self.device).eval()
        self.n_way_preprocess = weights.transforms()

        # for FID Score        
        self.fid = FrechetInceptionDistance(feature_dim=2048).to(self.device)
        self.fid.reset()

        # for IS Score
        self.inception_model = inception_v3(pretrained=True, transform_input=False).eval().to(self.device)
        self.inception_model.fc = torch.nn.Identity() 
        self.generated_preds = []


        # Make Dataset
        self.datasets = OUTPUT_Dataset(path, name, self.n_way_preprocess)

        # For Log
        self.metrics = {"GA":[], "FID":[], "IS":[]}
        
    @torch.no_grad()
    def n_way_top_k_acc(self, pred, class_id, n_way, num_trials=40, top_k=1):
        r"""
        Pred : [N, 1000]
        class_id : [1]
        """
        b, len_class = pred.shape
        pick_range =[i for i in np.arange(len_class) if i != class_id]
        acc_list = []
        for t in range(num_trials):
            idxs_picked = np.random.choice(pick_range, n_way-1, replace=False)
            pred_picked = torch.cat([pred[:, [class_id]], pred[:, idxs_picked]], dim=1)
            pred_infered = pred_picked.argmax(1)
            acc = accuracy(pred_infered, torch.zeros([b]).to(self.device, dtype=torch.long), 'multiclass', num_classes=n_way, 
                        top_k=top_k)
            acc_list.append(acc.mean().item())
        return np.mean(acc_list), np.std(acc_list)
        
    @torch.no_grad()
    def get_n_way_top_k_acc(self, pred_imgs, gt_imgs, n_way, num_trials, top_k):
        gt_class_id = self.n_way_model(gt_imgs).softmax(1).argmax(1).item()        
        pred_out    = self.n_way_model(pred_imgs).softmax(1).detach()
        return self.n_way_top_k_acc(pred_out, gt_class_id, n_way, num_trials, top_k)

    @torch.no_grad()
    def update_inception_score(self, pred_imgs):  # pred_imgs: [B, 3, 299, 299]
        if pred_imgs.shape[-1] != 299:
            pred_imgs = torch.nn.functional.interpolate(pred_imgs, size=(299, 299), mode='bilinear')
        pred_imgs = pred_imgs.to(self.device)
        logits = self.inception_model(pred_imgs)
        softmax = F.softmax(logits, dim=1)  # [B, 1000]
        self.generated_preds.append(softmax.cpu())
        
    def compute_inception_score(self, splits=10):
        preds = torch.cat(self.generated_preds, dim=0).numpy()  # [N, 1000]
        N = preds.shape[0]
        split_size = N // splits
        scores = []

        for k in range(splits):
            part = preds[k * split_size: (k + 1) * split_size]
            py = np.mean(part, axis=0)
            kl = part * (np.log(part + 1e-10) - np.log(py + 1e-10))
            scores.append(np.exp(np.mean(np.sum(kl, axis=1))))

        return np.mean(scores)
        
    def set_fid_score(self, pred_imgs, gt_imgs):
        self.fid.update(pred_imgs, is_real=False)
        self.fid.update(gt_imgs, is_real=True)
        
    @torch.no_grad()
    def __call__(self, n_way = 50, n_trial=40, top_k=1):
        progress_bar = tqdm(
                range(0, len(self.datasets)),
                initial=0,
                desc="Testing Generated Image",
            )

        for idx, (ga_gt, ga_sample, fid_gt, fid_sample) in enumerate(self.datasets):
            if idx == (len(self.datasets)-1):
                break
            ga_gt      = ga_gt.to(self.device)
            ga_sample  = ga_sample.to(self.device)
            fid_gt     = fid_gt.to(self.device)
            fid_sample = fid_sample.to(self.device)

            # n_way
            ga_score, ga_std = self.get_n_way_top_k_acc(ga_sample.cuda(),ga_gt.cuda(), n_way, n_trial, top_k)
            self.metrics['GA'].append(ga_score)

            # IS
            self.update_inception_score(fid_sample)
            
            # FID
            self.set_fid_score(fid_gt, fid_sample)

            # Logging
            progress_bar.update(1)
            logs = {"GA Score": ga_score, "index":idx}
            progress_bar.set_postfix(**logs)

        # Calculating FID
        self.metrics['FID'].append(self.fid.compute().item())

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluation of Generated Image",
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-p', '--path', default= "./output", help="Output Image's Path", type=str, dest="path")
    parser.add_argument('-n', '--name', default="val", help = "name of sub folder", type=str, dest="name")
    parser.add_argument('-nw', '--n_way', default=50, help = "GA Score's N", type=int, dest="n_way")
    parser.add_argument('-t', '--trial', default=40, help = "GA Score's Trial", type=int, dest="trial")
    parser.add_argument('-k', '--top_k', default=1, help = "GA Score's Top K", type=int, dest="top_k")
    parser.add_argument('-w', "--write", help='write log', action='store_false', dest="write")
    
    args = parser.parse_args()

    Image_Tester = Generated_Image_Tester(args.path, args.name)
    Image_Tester(args.n_way, args.trial, args.top_k)

    if args.write:
        with open(f"./results_{args.path.split("/")[-1]}.txt", 'w') as f:
            f.write(f"{"="* 30}\n")
            f.write(f"Result of Generated Images\n")
            f.write(f"{"="* 30}\n")
            for key, value in Image_Tester.metrics.items():
                f.write(f"{key} Score : {np.mean(value):.4f}\n")

        
