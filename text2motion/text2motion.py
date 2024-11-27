import os
import sys

root_dir = './' #if __name__ == "__main__" else os.path.join(os.getcwd(), "text2mesh")

import clip
import torch
import numpy as np
import models.vqvae as vqvae
import models.t2m_trans as trans
from utils.motion_process import recover_from_ric
from models.rotation2xyz import Rotation2xyz
import numpy as np
import gc
from visualize.simplify_loc2rot import joints2smpl
from tqdm import tqdm
import options.option_transformer as option_trans
import argparse
import json



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, help="path to input file. Must be json")
    parser.add_argument("--output", type=str, help="output root path")

    args = parser.parse_args()

    input_json = args.input
    output_root = args.output
    parser = None
    gc.collect()

os.environ["PYOPENGL_PLATFORM"] = "egl"
os.environ["MESA_GL_VERSION_OVERRIDE"] = "4.1"
#sys.argv = ["./VQ-Trans/GPT_eval_multi.py"]
#sys.path.append("./VQ-Trans")
#sys.path.append("./pyrender")

args = option_trans.get_args_parser()

args.dataname = "t2m"
args.resume_pth = "/home/under2/anaconda3/envs/video-editing/HMTV" if __name__ == '__main' else os.path.join(root_dir, "pretrained/VQVAE/net_last.pth")
args.resume_trans = "./pretrained/VQTransformer_corruption05/net_best_fid.pth" if __name__ == '__main__' else os.path.join(root_dir, "pretrained/VQTransformer_corruption05/net_best_fid.pth")
args.down_t = 2
args.depth = 3
args.block_size = 51

## load clip model and datasets
is_cuda = torch.cuda.is_available()
device = torch.device("cuda" if is_cuda else "cpu")
print(device)
clip_model, clip_preprocess = clip.load(
    "ViT-B/32", device=device, jit=False, download_root=root_dir
)  # Must set jit=False for training

if is_cuda:
    clip.model.convert_weights(clip_model)

clip_model.eval()
for p in clip_model.parameters():
    p.requires_grad = False

net = vqvae.HumanVQVAE(
    args,  ## use args to define different parameters in different quantizers
    args.nb_code,
    args.code_dim,
    args.output_emb_width,
    args.down_t,
    args.stride_t,
    args.width,
    args.depth,
    args.dilation_growth_rate,
)

trans_encoder = trans.Text2Motion_Transformer(
    num_vq=args.nb_code,
    embed_dim=1024,
    clip_dim=args.clip_dim,
    block_size=args.block_size,
    num_layers=9,
    n_head=16,
    drop_out_rate=args.drop_out_rate,
    fc_rate=args.ff_rate,
)


print("loading checkpoint from {}".format(args.resume_pth))
ckpt = torch.load(args.resume_pth, map_location="cpu")
net.load_state_dict(ckpt["net"], strict=True)
net.eval()

print("loading transformer checkpoint from {}".format(args.resume_trans))
ckpt = torch.load(args.resume_trans, map_location="cpu")
trans_encoder.load_state_dict(ckpt["trans"], strict=True)
trans_encoder.eval()

mean = torch.from_numpy(np.load(os.path.join(root_dir, "/home/under2/anaconda3/envs/video-editing/HMTV/mean.npy")))
std = torch.from_numpy(np.load(os.path.join(root_dir, "/home/under2/anaconda3/envs/video-editing/HMTV/std.npy")))


if is_cuda:
    net.cuda()
    trans_encoder.cuda()
    mean = mean.cuda()
    std = std.cuda()

import cv2
import mediapipe as mp

def extract_keypoints(image_path):
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(static_image_mode=True)
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # 추출
    results = pose.process(image_rgb)
    keypoints = []
    if results.pose_landmarks:
        for landmark in results.pose_landmarks.landmark:
            keypoints.append([landmark.x, landmark.y, landmark.z, landmark.visibility])
    return keypoints

# from some_3d_pose_library import lift_2d_to_3d

# def keypoints_to_3d(keypoints_2d):
#     keypoints_3d = lift_2d_to_3d(keypoints_2d)
#     return keypoints_3d  # shape: (frames, njoints, 3)

def render(motions)->None:
    # motion.shape (frame, 22, 3)
    frames, njoints, nfeats = motions.shape
    MINS = motions.min(axis=0).min(axis=0)

    height_offset = MINS[1]
    motions[:, :, 1] -= height_offset
    is_cuda = torch.cuda.is_available()
    j2s = joints2smpl(num_frames=frames, device_id=0, cuda=is_cuda)
    rot2xyz = Rotation2xyz(device=device)
    
    print(f"Running SMPLify, it may take a few minutes.")
    motion_tensor, opt_dict = j2s.joint2smpl(motions)  # [nframes, njoints, 3]
    vertices = rot2xyz(
        torch.tensor(motion_tensor).clone().detach(),
        mask=None,
        pose_rep="rot6d",
        translation=True,
        glob=True,
        jointstype="vertices",
        vertstrans=True,
    )
    vertices = vertices.detach().cpu()

    return vertices


def predict(clip_text):    
    gc.collect()
    print("prompt text instruction: {}".format(clip_text))
    if torch.cuda.is_available():
        text = clip.tokenize([clip_text], truncate=True).cuda()
    else:
        text = clip.tokenize([clip_text], truncate=True)
    feat_clip_text = clip_model.encode_text(text).float()
    index_motion = trans_encoder.sample(feat_clip_text[0:1], False)
    print('motion', index_motion, index_motion.shape)
    pred_pose = net.forward_decoder(index_motion)
    pred_xyz = recover_from_ric((pred_pose * std + mean).float(), 22)
    xyz = pred_xyz.reshape(1, -1, 22, 3)
    np.save('motion.npy', xyz.detach().cpu().numpy())
    import visualization.plot_3d_global as plot_3d
    pose_vis = plot_3d.draw_to_batch(xyz.detach().cpu().numpy(),clip_text, ['example.gif'])
    
    return render(
        pred_xyz.detach().cpu().numpy().squeeze(axis=0)
    )


if __name__ == "__main__":
    with open(input_json, 'r') as jf:
        prompts = json.load(jf)
        assert isinstance(prompts, list), "prompts must be list type"
    
    save_dict = dict()  # key: prompt, value: path to save

    os.makedirs(output_root, exist_ok=True)

    for i, prompt in enumerate(tqdm(prompts)):
        vertices_3d = predict(clip_text=prompt)
        vts_path = os.path.join(output_root, f'{i+1}.vts')
        torch.save(vertices_3d, vts_path)

        save_dict[prompt] = vts_path
    
    print("Saving prompt-path pair.")
    with open(os.path.join(output_root, "prompt_path.json"), 'w') as jf:
        json.dump(save_dict, jf)

    print('DONE')
