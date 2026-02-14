import argparse
import torch
from nanochat.common import compute_init, autodetect_device_type
from nanochat.checkpoint_manager import load_model
import numpy as np
import matplotlib.pyplot as plt


parser = argparse.ArgumentParser(description='Chat with the model')
parser.add_argument('-i', '--source', type=str, default="sft", help="Source of the model: sft|rl")
parser.add_argument('-g', '--model-tag', type=str, default=None, help='Model tag to load')
parser.add_argument('-s', '--step', type=int, default=None, help='Step to load')
parser.add_argument('-p', '--prompt', type=str, default='', help='Prompt the model, get a single response back')
parser.add_argument('-t', '--temperature', type=float, default=0.6, help='Temperature for generation')
parser.add_argument('-k', '--top-k', type=int, default=50, help='Top-k sampling parameter')
#NEW
parser.add_argument('-topp', '--top-p', type=int, default=1.0, help='top-p sampling parameter')

parser.add_argument('--device-type', type=str, default='', choices=['cuda', 'cpu', 'mps'], help='Device type for evaluation: cuda|cpu|mps. empty => autodetect')
parser.add_argument('-d', '--dtype', type=str, default='bfloat16', choices=['float32', 'bfloat16'])
args = parser.parse_args()

# Init the model and tokenizer

device_type = autodetect_device_type() if args.device_type == "" else args.device_type
ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)
model, tokenizer, meta = load_model(args.source, device, phase="eval", model_tag=args.model_tag, step=args.step)

gatescore_list = np.array([])


for i, block in enumerate(model.transformer.h):
        #check output-gating-score, which should be sparse
        print(i,"layer gating score:")
        print(block.attn.output_gate.weight.data)
        weight_vals = block.attn.output_gate.weight.detach().cpu().flatten().numpy()
        gatescore_list = np.concatenate([gatescore_list, weight_vals])

plt.figure(figsize=(10, 6))

plt.hist(gatescore_list, bins=50, edgecolor='black', alpha=0.7, color='#1f77b4')
plt.title("Gate score distribution", fontsize=12)
plt.xlabel("Gate score", fontsize=10)
plt.ylabel("Frequency", fontsize=10)
plt.grid(axis='y', alpha=0.3)

plt.axvline(gatescore_list.mean(), color='red', linestyle='--', label=f'Mean: {gatescore_list.mean():.4f}')
plt.axvline(gatescore_list.var(), color='green', linestyle='--', label=f'Variance: {gatescore_list.var():.4f}' )
plt.legend()

plt.show()
