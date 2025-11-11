import os
import glob
import argparse
from torch_fidelity import calculate_metrics

def compute_inception_score(image_folder):
    metrics = calculate_metrics(input1=image_folder, isc=True, cuda=True)
    return metrics

# Example usage

parser = argparse.ArgumentParser(description="Evaluation of Generated Image",
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-p', '--path', default= "./output", help="Output Image's Path", type=str, dest="path")
parser.add_argument('-n', '--name', default="test_pred", help = "name of sub folder", type=str, dest="name")
args = parser.parse_args()


image_folder = os.path.join(args.path, args.name)
m = compute_inception_score(image_folder)
print(m)

with open(f"./results_{args.path.split("/")[-1]}.txt", 'a') as f:
    f.write(f"IS Score : {m['inception_score_mean']:.4f}")