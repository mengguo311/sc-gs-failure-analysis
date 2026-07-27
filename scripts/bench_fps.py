"""Measure render FPS of a trained SC-GS model over the test cameras.

Usage:
  python scripts/bench_fps.py --model_path outputs/jumpingjacks_n512_node \
      --source_path data/jumpingjacks [--rounds 3] [--out results/fps.csv]

Reports mean/median per-frame time (deform.step + render, GPU-synced) and FPS.
"""
import os
import sys
import time
import argparse

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJ, 'SC-GS'))

import numpy as np
import torch
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, OptimizationParams, get_combined_args
from scene import Scene, DeformModel
from gaussian_renderer import render, GaussianModel
from utils.general_utils import safe_state


def main():
    parser = ArgumentParser()
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    op = OptimizationParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument('--gui', action='store_true')
    parser.add_argument('--W', type=int, default=800)
    parser.add_argument('--H', type=int, default=800)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--deform-type", type=str, default='mlp')
    parser.add_argument('--rounds', type=int, default=3)
    parser.add_argument('--out', type=str, default=None)
    parser.add_argument('--label', type=str, default='')
    args = get_combined_args(parser)
    if not args.model_path.endswith(args.deform_type):
        args.model_path = os.path.join(os.path.dirname(os.path.normpath(args.model_path)),
                                       os.path.basename(os.path.normpath(args.model_path)) + f'_{args.deform_type}')
    safe_state(False)
    dataset = model.extract(args)
    pipe = pipeline.extract(args)

    with torch.no_grad():
        deform = DeformModel(K=dataset.K, deform_type=dataset.deform_type, is_blender=dataset.is_blender,
                             skinning=dataset.skinning, hyper_dim=dataset.hyper_dim, node_num=dataset.node_num,
                             pred_opacity=dataset.pred_opacity, pred_color=dataset.pred_color,
                             use_hash=dataset.use_hash, hash_time=dataset.hash_time,
                             d_rot_as_res=dataset.d_rot_as_res, local_frame=dataset.local_frame,
                             progressive_brand_time=dataset.progressive_brand_time, max_d_scale=dataset.max_d_scale)
        deform.load_weights(dataset.model_path, iteration=args.iteration)
        gs_fea_dim = deform.deform.node_num if dataset.skinning and deform.name == 'node' else dataset.hyper_dim
        gaussians = GaussianModel(dataset.sh_degree, fea_dim=gs_fea_dim, with_motion_mask=dataset.gs_with_motion_mask)
        scene = Scene(dataset, gaussians, load_iteration=args.iteration, shuffle=False)
        bg = torch.tensor([1, 1, 1] if dataset.white_background else [0, 0, 0],
                          dtype=torch.float32, device="cuda")
        views = scene.getTestCameras()
        xyz = gaussians.get_xyz

        times = []
        for rnd in range(args.rounds + 1):  # round 0 = warmup, not timed
            for view in views:
                torch.cuda.synchronize()
                t0 = time.time()
                time_input = deform.deform.expand_time(view.fid)
                d_values = deform.step(xyz.detach(), time_input, feature=gaussians.feature,
                                       is_training=False, motion_mask=gaussians.motion_mask)
                render(view, gaussians, pipe, bg, d_values['d_xyz'], d_values['d_rotation'],
                       d_values['d_scaling'], d_opacity=d_values['d_opacity'],
                       d_color=d_values['d_color'], d_rot_as_res=deform.d_rot_as_res)
                torch.cuda.synchronize()
                if rnd > 0:
                    times.append(time.time() - t0)

        times = np.array(times)
        n_gs = xyz.shape[0]
        line = (f"{args.label or dataset.model_path},{n_gs},{deform.deform.node_num},"
                f"{times.mean()*1000:.2f},{np.median(times)*1000:.2f},{1.0/times.mean():.2f}")
        print("label,n_gaussians,node_num,mean_ms,median_ms,fps")
        print(line)
        if args.out:
            new = not os.path.exists(args.out)
            with open(args.out, 'a') as f:
                if new:
                    f.write("label,n_gaussians,node_num,mean_ms,median_ms,fps\n")
                f.write(line + "\n")


if __name__ == '__main__':
    main()
