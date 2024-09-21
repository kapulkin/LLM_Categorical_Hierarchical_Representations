#%%

import torch
from torch import nn
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM

import dotenv

config = dotenv.dotenv_values(".env")
model_name = config["MODEL_NAME"]
g_file_path = config["G_FILE_PATH"]
space_char = config.get("SPACE_CHAR", "")

#%%
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name,
                                             device_map="auto")

g = torch.load(g_file_path).to(device) # g_file_path in store_matrices.py

#%%
prompt = "I want to prepare a detailed description of a business. Select some industry and list, please, aspects of the business in that industry I should include into the description."

# prompt = "Hi"

#%%
tokens = tokenizer(prompt, return_tensors="pt").to(model.device)
max_length = len(tokens.input_ids[0]) + 250
output = model.generate(**tokens, max_length=max_length, output_hidden_states=True, return_dict_in_generate=True, output_logits=True)

# %%
output_tokens = output.sequences[0].to('cpu').detach().numpy()
text = tokenizer.decode(output_tokens)
print(text)

# map output_text <--> tokens
token_to_text_position = []
text_position_to_token = []
position = 0
for index, token  in enumerate(output_tokens):
    token_text = tokenizer.decode([token])
    token_to_text_position.append(position)
    for i in range(len(token_text)):
        text_position_to_token.append(index)
    position += len(token_text)
    if position > len(text):
        break

# %%
# find text position of industry name and each apect

import re

def find_gemma_2_positions(markdown):
    # Find the position of the industry name
    industry_match = re.search(r'\*\*Industry:\*\*  \*\*(.*?)\*\*', markdown)
    industry_position = industry_match.start(1) if industry_match else -1
    industry_name = industry_match.group(1) if industry_match else None

    # Find the positions of each aspect name
    aspect_positions = []
    for match in re.finditer(r'\* \*\*(.*?):\*\*', markdown):
        aspect_name = match.group(1)
        aspect_positions.append((aspect_name, match.start(1)))

    return (industry_name, industry_position), aspect_positions
    

def find_llama_3_1_positions(markdown):
    # Find the position of the industry name
    industry_match = re.search(r'"(.*?)"', markdown)
    industry_position = industry_match.start(1) if industry_match else -1
    industry_name = industry_match.group(1) if industry_match else None

    # Find the positions of each aspect name
    aspect_positions = []
    for match in re.finditer(r'\*\*(.*?)\*\*', markdown):
        aspect_name = match.group(1)
        aspect_positions.append((aspect_name, match.start(1)))

    return (industry_name, industry_position), aspect_positions

industry, aspect_positions = find_gemma_2_positions(text)

industry_name, industry_position = industry

print(f"industry: {industry_name}, {industry_position}")
for aspect_name, aspect_position in aspect_positions:
    print(f"aspect: {aspect_name}, {aspect_position}")

#%%
# map text position to token position
# map token position to embedding

# compute from embedding logits and then from logits embedding in g-space
def make_embedding(model, hidden_state, g):
    lm_head = model.get_output_embeddings()
    logits = lm_head(hidden_state)
    logits = logits / model.config.final_logit_softcapping
    logits = torch.tanh(logits)
    logits = logits * model.config.final_logit_softcapping

    logits = logits[0][0].detach()

    return logits @ g

industry_token_position = text_position_to_token[industry_position]
industry_hidden_state = output.hidden_states[industry_token_position][-1]
industry_embedding = make_embedding(model, industry_hidden_state, g)

aspect_token_positions = [text_position_to_token[position] for _, position in aspect_positions ]
aspect_hidden_states = [output.hidden_states[position][-1] for position in aspect_token_positions]
aspect_embeddings = [make_embedding(model, hidden_state, g) for hidden_state in aspect_hidden_states]

#%%
# make dirs from emebddings for industry and aspects
import hierarchical as hrc

def estimate_cat_dir(category_embeddings): 
    lda_dir, category_mean = hrc.estimate_single_dir_from_embeddings(category_embeddings)
    return {'lda': lda_dir, 'mean': category_mean}

industry_dir = estimate_cat_dir(torch.stack(aspect_embeddings))
aspect_dirs = [estimate_cat_dir(torch.unsqueeze(embedding, 0)) for embedding in aspect_embeddings]

# build diagram

#%% Build 2d plot

# does not work
def build_2d_plot(dirs, names):
    fig, axs = plt.subplots(1, 1, figsize=(25,7))

    inds1 = {names[0]: [0],
        names[1]: [1],
        names[2]: [2]
    }

    higher = dirs[0]["lda"]
    subcat1 = dirs[0]["lda"]
    subcat2 = dirs[0]["lda"]

    vocab_list = {}
    added_inds = inds1

    hrc.proj_2d_single_diff(higher, subcat1, subcat2,
                            g, vocab_list, axs[1],
                            normalize = True,
                            orthogonal = True,
                            added_inds=added_inds, k = 50, fontsize= 15,
                            draw_arrows= True,
                            arrow1_name=rf'$\bar{{\ell}}_{{{names[0]}}}$',
                            arrow2_name=rf'$\bar{{\ell}}_{{{names[2]}}} - \bar{{\ell}}_{{{names[1]}}}$',
                            alpha = 0.03,  s = 0.05,
                            target_alpha=0.6, target_s=4,
                            xlim = (-10,10), ylim = (-10,10),
                            right_topk = False,
                            left_topk = False,
                            top_topk = False,
                            bottom_topk = False,
                            xlabel = "", ylabel="",
                            title = rf'${names[0]}$ vs ${names[1]} \Rightarrow {names[2]}$')
    
    fig.tight_layout()
    fig.savefig(f"figures/three_2d_plots.png", dpi=300, bbox_inches='tight')
    fig.show()


#%% Build 3d plot
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

def build_3d_plot(dirs, embeddings):
    fig = plt.figure(figsize=(20, 8))
    ax = fig.add_subplot(111, projection='3d')

    categories = list(dirs.keys())
    cat1 = categories[0]
    cat2 = categories[1]
    cat3 = categories[2]
    cat4 = categories[3]

    dir1 = dirs[cat1]["lda"]
    dir2 = dirs[cat2]["lda"]
    dir3 = dirs[cat3]["lda"]
    dir4 = dirs[cat4]["lda"]

    xaxis = dir1 / dir1.norm()
    yaxis = dir2 - (dir2 @ xaxis) * xaxis
    yaxis = yaxis / yaxis.norm()
    zaxis = dir3 - (dir3 @ xaxis) * xaxis - (dir3 @ yaxis) * yaxis
    zaxis = zaxis / zaxis.norm()

    axes = torch.stack([xaxis, yaxis, zaxis], dim=1)

    g1 = embeddings[0]
    g2 = embeddings[1]
    g3 = embeddings[2]
    
    proj1 = (g1 @ axes).cpu().numpy()
    proj2 = (g2 @ axes).cpu().numpy()
    proj3 = (g3 @ axes).cpu().numpy()
    proj = (g @ axes).cpu().numpy()

    P1 = (dir1 @ axes).cpu().numpy()
    P2 = (dir2 @ axes).cpu().numpy()
    P3 = (dir3 @ axes).cpu().numpy()
    P4 = (dir4 @ axes).cpu().numpy()

    ax.scatter(P1[0], P1[1], P1[2], color='r', s=100)
    ax.scatter(P2[0], P2[1], P2[2], color='g', s=100)
    ax.scatter(P3[0], P3[1], P3[2], color='b', s=100)
    ax.scatter(P4[0], P4[1], P4[2], color='m', s=100)

    verts1 = [list(zip([P1[0], P2[0], P3[0]], [P1[1], P2[1], P3[1]], [P1[2], P2[2], P3[2]]))]
    triangle1 = Poly3DCollection(verts1, alpha=.1, linewidths=1, linestyle =  "--", edgecolors='k')
    triangle1.set_facecolor('yellow')
    ax.add_collection3d(triangle1)

    ax.quiver(0, 0, 0, P1[0], P1[1], P1[2], color='r', arrow_length_ratio=0.01)
    ax.quiver(0, 0, 0, P2[0], P2[1], P2[2], color='g', arrow_length_ratio=0.01)
    ax.quiver(0, 0, 0, P3[0], P3[1], P3[2], color='b', arrow_length_ratio=0.01)


    scatter1 = ax.scatter(proj1[:,0], proj1[:,1], proj1[:,2], c='r', label=cat1)
    scatter2 = ax.scatter(proj2[:,0], proj2[:,1], proj2[:,2], c='g', label=cat2)
    scatter3 = ax.scatter(proj3[:,0], proj3[:,1], proj3[:,2], c='b', label=cat3)
    scatter = ax.scatter(proj[:,0], proj[:,1], proj[:,2], c='gray', s= 0.05, alpha = 0.01)


    scale = 1.2
    ax.text(P1[0]*scale + 2, P1[1]* scale, P1[2]*scale, cat1, bbox=dict(facecolor='r', alpha=0.2))
    ax.text(P2[0]*scale+0.5, P2[1]* scale+0.5, P2[2]*scale, cat2, bbox=dict(facecolor='g', alpha=0.2))
    ax.text(P3[0]*scale, P3[1]* scale, P3[2]*scale, cat3, bbox=dict(facecolor='b', alpha=0.2))
    ax.text(P4[0]-0.6, P4[1]-0.6, P4[2], rf'$\bar{{\ell}}_{{{cat4}}}$', bbox=dict(facecolor='k', alpha=0.2))

    ax.set_xlim(-8,10)
    ax.set_ylim(-8,10)
    ax.set_zlim(-2.5, 15.5)

    ax.view_init(elev=20, azim=70)

    plt.tight_layout()
    fig.savefig(f"figures/two_3D_plots.png", dpi=300, bbox_inches='tight')
    plt.show()

dirs_dict = { aspect_name: aspect_dir for (aspect_name, _), aspect_dir, _ in zip(aspect_positions, aspect_dirs, range(3)) }
dirs_dict[industry_name] = industry_dir

build_3d_plot(
    dirs_dict,
    aspect_embeddings[:3] + [industry_embedding]
)

#%% Check logits calculation

lm_head = model.get_output_embeddings()

logits = [lm_head(h[-1]) for h in output.hidden_states]

tahn_logits = []
for l in logits:
    l = l / model.config.final_logit_softcapping
    l = torch.tanh(l)
    l = l * model.config.final_logit_softcapping
    tahn_logits.append(l)

next_token_logits = [l[:, -1, :].clone() for l in logits]
next_token_scores = [nn.functional.log_softmax(
    l, dim=-1
)  # (batch_size * num_beams, vocab_size)
for l in next_token_logits]

print([l.shape for l in logits])
# %%
[torch.max(torch.abs(l1 - l2)) for l1, l2 in zip(tahn_logits, output.logits)]
# %%
