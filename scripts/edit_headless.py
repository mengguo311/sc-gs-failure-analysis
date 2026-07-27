"""Headless SC-GS editing: scripted rotational drags with different ARAP modes.

Replicates the train_gui.py animation/editing pipeline without a GUI:
  1. Load a trained SC-GS model (gaussians + node deform network).
  2. Build the LapDeform/ARAPDeformer animation tool at a chosen time (animation_initialize).
  3. Select a drag handle group (nearest node to --drag_point, expanded n-ring like the GUI's
     'A' key) and optional static anchor groups (--anchor_point, delta = 0).
  4. Rotate the drag group around its centroid by each angle in --angles (GUI's
     set_rotation_delta semantics) and solve ARAP in one of three modes:
       from_init   : one-shot solve, Laplacian-LSQ init (GUI mode 'arap_from_init')
       iterative   : fine sub-steps of --fine_step deg, warm-started (GUI mode 'arap_iterative')
       progressive : N = --steps coarse sub-steps, warm-started (our Phase-3 improvement)
  5. Propagate to Gaussians exactly as train_gui.test_step does (node_trans_bias -> p2dR ->
     LBS), render fixed test cameras, and record node/Gaussian rigidity metrics to CSV.

Run from anywhere; SC-GS repo path is resolved relative to this file.
"""
import os
import sys
import time
import json

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCGS = os.path.join(PROJ, 'SC-GS')
sys.path.insert(0, SCGS)

import numpy as np
import torch
from scipy.spatial.transform import Rotation as ScipyR

from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, OptimizationParams, get_combined_args
from scene import Scene, DeformModel
from gaussian_renderer import render, GaussianModel
from utils.general_utils import safe_state
from utils.deform_utils import cal_arap_error
from lap_deform import LapDeform
import torchvision


def build_animate_tool(deform, edit_time, t_samp_num=16):
    """Mirror of GUI.animation_initialize (train_gui.py:237)."""
    node_gs = deform.deform.as_gaussians
    fid = torch.tensor(edit_time).cuda().float()
    time_input = fid.unsqueeze(0).expand(node_gs.get_xyz.shape[0], -1)
    values = deform.deform.node_deform(t=time_input)
    pcl = node_gs.get_xyz + values['d_xyz']
    t_samp = torch.linspace(0, 1, t_samp_num).cuda().float()
    time_input_traj = t_samp[None, :, None].expand(node_gs.get_xyz.shape[0], -1, 1)
    trajectory = deform.deform.node_deform(t=time_input_traj)['d_xyz'] + node_gs.get_xyz[:, None]
    node_radius = deform.deform.node_radius.detach()
    tool = LapDeform(init_pcl=pcl, K=4, trajectory=trajectory, node_radius=node_radius)
    return tool, pcl.detach()


def select_group(tool, pcl, seed_point, n_rings):
    """Nearest node to seed_point, expanded with n-ring graph neighbors (GUI 'A' key)."""
    seed = torch.tensor(seed_point).float().cuda()
    idx = (pcl - seed).norm(dim=-1).argmin()[None]
    if n_rings > 0:
        idx = tool.add_n_ring_nbs(idx, n=n_rings)
    return idx


def rotated_targets(pcl, drag_idx, angle_deg, axis, center=None):
    """GUI set_rotation_delta semantics: rotate drag group to absolute targets.
    center=None rotates around the group centroid (GUI default); an explicit center
    (e.g. the shoulder joint) emulates dragging a limb tip along the arc of a joint
    rotation, leaving intermediate nodes free."""
    pts = pcl[drag_idx].cpu().numpy()
    c = pts.mean(axis=0) if center is None else np.asarray(center, dtype=np.float64)
    rot = ScipyR.from_rotvec(np.radians(angle_deg) * np.asarray(axis) / np.linalg.norm(axis)).as_matrix()
    return (pts - c) @ rot.T + c


def edge_metrics(init_pcl, deformed, deformer, region_nodes):
    """Rigidity metrics over the ARAP graph edges (ii, jj)."""
    ii, jj = deformer.ii, deformer.jj
    e0 = (init_pcl[ii] - init_pcl[jj]).norm(dim=-1)
    e1 = (deformed[ii] - deformed[jj]).norm(dim=-1)
    rel = (e1 - e0).abs() / (e0 + 1e-8)
    region_mask = torch.zeros(init_pcl.shape[0], dtype=torch.bool, device=init_pcl.device)
    region_mask[region_nodes] = True
    edge_in_region = region_mask[ii] | region_mask[jj]
    out = {
        'edge_stretch_mean': rel.mean().item(),
        'edge_stretch_p95': rel.quantile(0.95).item(),
        'edge_stretch_max': rel.max().item(),
        'edge_stretch_region_mean': rel[edge_in_region].mean().item(),
        'edge_stretch_region_p95': rel[edge_in_region].quantile(0.95).item(),
        'edge_stretch_region_max': rel[edge_in_region].max().item(),
    }
    node_seq = torch.stack([init_pcl, deformed], dim=0)
    out['arap_error'] = cal_arap_error(node_seq, deformer.ii, deformer.jj, deformer.nn,
                                       K=deformer.K, weight=deformer.normalized_weight).item()
    return out


def gaussian_stretch_metrics(x0, x1, region_gs_mask, K=8):
    """Neighbor-distance stretch of Gaussian centers (canonical KNN edges)."""
    import pytorch3d.ops
    sub0 = x0[region_gs_mask]
    sub1 = x1[region_gs_mask]
    if sub0.shape[0] < K + 1:
        return {'gs_stretch_mean': float('nan'), 'gs_stretch_p95': float('nan'), 'gs_stretch_max': float('nan')}
    _, nn_idx, _ = pytorch3d.ops.knn_points(sub0[None], sub0[None], K=K + 1)
    nn_idx = nn_idx[0, :, 1:]
    d0 = (sub0[:, None] - sub0[nn_idx]).norm(dim=-1)
    d1 = (sub1[:, None] - sub1[nn_idx]).norm(dim=-1)
    rel = (d1 - d0).abs() / (d0 + 1e-8)
    per_gs = rel.max(dim=1).values
    return {
        'gs_stretch_mean': rel.mean().item(),
        'gs_stretch_p95': per_gs.quantile(0.95).item(),
        'gs_stretch_max': per_gs.max().item(),
    }


def solve_schedule(tool, handle_idx, base_pos_np, drag_local_mask, drag_targets_per_step):
    """Run a sequence of ARAP solves. First solve starts from Laplacian init (init_verts=None),
    later solves warm-start from the previous solution — exactly train_gui.py:768-777."""
    trans_bias = None
    t_start = time.time()
    for tgt in drag_targets_per_step:
        handle_pos = base_pos_np.copy()
        handle_pos[drag_local_mask] = tgt
        init_verts = None if trans_bias is None else tool.init_pcl + trans_bias
        with torch.no_grad():
            deformed, quat, _ = tool.deform_arap(handle_idx=handle_idx, handle_pos=handle_pos,
                                                 init_verts=init_verts, return_R=True)
        trans_bias = deformed - tool.init_pcl
    solve_time = time.time() - t_start
    return deformed, trans_bias, solve_time


def main():
    parser = ArgumentParser(description="Headless SC-GS ARAP editing")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    op = OptimizationParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument('--gui', action='store_true')
    parser.add_argument('--W', type=int, default=800)
    parser.add_argument('--H', type=int, default=800)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--deform-type", type=str, default='mlp')
    # Edit protocol
    parser.add_argument('--edit_time', type=float, default=0.0, help='fid in [0,1] to edit at')
    parser.add_argument('--dump_nodes', type=str, default=None,
                        help='dump node positions at edit_time to this .npz and exit')
    parser.add_argument('--drag_point', type=str, default=None, help='"x,y,z" seed of drag group')
    parser.add_argument('--anchor_point', type=str, action='append', default=[],
                        help='"x,y,z" seed of a static anchor group (repeatable)')
    parser.add_argument('--n_rings', type=int, default=2, help='n-ring expansion (GUI default 2)')
    parser.add_argument('--rot_axis', type=str, default='0,0,1')
    parser.add_argument('--rot_center', type=str, default='',
                        help='"x,y,z" rotation center (default: drag-group centroid)')
    parser.add_argument('--angles', type=str, default='15,45,90,135')
    parser.add_argument('--edit_mode', type=str, default='from_init',
                        choices=['from_init', 'iterative', 'progressive'])
    parser.add_argument('--steps', type=int, default=4, help='progressive sub-step count N')
    parser.add_argument('--fine_step', type=float, default=1.0, help='iterative step size (deg)')
    parser.add_argument('--cam_ids', type=str, default='0', help='test camera indices to render')
    parser.add_argument('--out_dir', type=str, required=True)
    parser.add_argument('--tag', type=str, default='')

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

        tool, init_pcl = build_animate_tool(deform, args.edit_time)

        if getattr(args, 'dump_nodes', None):
            np.savez(args.dump_nodes, nodes=init_pcl.cpu().numpy())
            print(f'[dump] wrote {init_pcl.shape[0]} nodes at t={args.edit_time} to {args.dump_nodes}')
            return

        assert args.drag_point, '--drag_point required unless --dump_nodes'
        os.makedirs(args.out_dir, exist_ok=True)
        os.makedirs(os.path.join(args.out_dir, 'renders'), exist_ok=True)
        os.makedirs(os.path.join(args.out_dir, 'nodes'), exist_ok=True)

        drag_idx = select_group(tool, init_pcl, [float(v) for v in args.drag_point.split(',')], args.n_rings)
        anchor_idx_list = [select_group(tool, init_pcl, [float(v) for v in ap.split(',')], args.n_rings)
                           for ap in args.anchor_point]
        anchor_idx = (torch.unique(torch.cat(anchor_idx_list)) if anchor_idx_list
                      else torch.tensor([], dtype=torch.long).cuda())
        # drag nodes take precedence if overlapping
        anchor_idx = anchor_idx[~torch.isin(anchor_idx, drag_idx)]
        handle_idx_t = torch.cat([drag_idx, anchor_idx])
        handle_idx = handle_idx_t.tolist()
        base_pos_np = init_pcl[handle_idx_t].cpu().numpy()
        drag_local_mask = np.zeros(len(handle_idx), dtype=bool)
        drag_local_mask[:len(drag_idx)] = True

        region_nodes = tool.add_n_ring_nbs(drag_idx, n=4)
        axis = [float(v) for v in args.rot_axis.split(',')]
        rot_center = ([float(v) for v in args.rot_center.split(',')]
                      if getattr(args, 'rot_center', '') else None)
        angles = [float(a) for a in args.angles.split(',')]
        cam_ids = [int(c) for c in args.cam_ids.split(',')]
        test_cams = scene.getTestCameras()

        # Region mask on Gaussians: nearest node (at edit time, undeformed) in region
        fid = torch.tensor(args.edit_time).cuda().float()
        time_input = deform.deform.expand_time(fid)
        d_values0 = deform.step(gaussians.get_xyz.detach(), time_input, feature=gaussians.feature,
                                is_training=False, motion_mask=gaussians.motion_mask)
        x_t0 = gaussians.get_xyz + d_values0['d_xyz']
        import pytorch3d.ops
        _, nearest_node, _ = pytorch3d.ops.knn_points(x_t0[None], init_pcl[None], K=1)
        region_mask_nodes = torch.zeros(init_pcl.shape[0], dtype=torch.bool, device='cuda')
        region_mask_nodes[region_nodes] = True
        region_gs_mask = region_mask_nodes[nearest_node[0, :, 0]]

        meta = {
            'model_path': dataset.model_path, 'edit_time': args.edit_time,
            'drag_point': args.drag_point, 'anchor_points': args.anchor_point,
            'n_rings': args.n_rings, 'rot_axis': args.rot_axis,
            'rot_center': args.rot_center or 'group_centroid',
            'drag_idx': drag_idx.tolist(), 'anchor_idx': anchor_idx.tolist(),
            'region_nodes': region_nodes.tolist(),
            'edit_mode': args.edit_mode, 'steps': args.steps, 'fine_step': args.fine_step,
            'node_num': int(init_pcl.shape[0]), 'region_gs_count': int(region_gs_mask.sum().item()),
        }
        with open(os.path.join(args.out_dir, f'meta_{args.edit_mode}{args.tag}.json'), 'w') as f:
            json.dump(meta, f, indent=2)

        csv_path = os.path.join(args.out_dir, 'metrics.csv')
        csv_header = ('mode,tag,angle,steps,solve_time_s,arap_error,'
                      'edge_stretch_mean,edge_stretch_p95,edge_stretch_max,'
                      'edge_stretch_region_mean,edge_stretch_region_p95,edge_stretch_region_max,'
                      'gs_stretch_mean,gs_stretch_p95,gs_stretch_max,handle_residual\n')
        if not os.path.exists(csv_path):
            with open(csv_path, 'w') as f:
                f.write(csv_header)

        for angle in angles:
            if args.edit_mode == 'from_init':
                schedule = [rotated_targets(init_pcl, drag_idx, angle, axis, rot_center)]
                nsteps = 1
            elif args.edit_mode == 'iterative':
                nsteps = max(1, int(round(angle / args.fine_step)))
                schedule = [rotated_targets(init_pcl, drag_idx, angle * (k + 1) / nsteps, axis, rot_center)
                            for k in range(nsteps)]
            else:  # progressive
                nsteps = args.steps
                schedule = [rotated_targets(init_pcl, drag_idx, angle * (k + 1) / nsteps, axis, rot_center)
                            for k in range(nsteps)]

            deformed, trans_bias, solve_time = solve_schedule(
                tool, handle_idx, base_pos_np, drag_local_mask, schedule)

            m = edge_metrics(init_pcl, deformed, tool.arap_deformer, region_nodes)
            m['handle_residual'] = float(np.linalg.norm(
                deformed[handle_idx_t].cpu().numpy() -
                np.concatenate([schedule[-1], base_pos_np[~drag_local_mask]], axis=0), axis=-1).mean())

            # Propagate to Gaussians (train_gui.test_step, node branch)
            d_values = deform.step(gaussians.get_xyz.detach(), time_input, feature=gaussians.feature,
                                   is_training=False, node_trans_bias=trans_bias,
                                   motion_mask=gaussians.motion_mask)
            x_edit = gaussians.get_xyz + d_values['d_xyz']
            m.update(gaussian_stretch_metrics(x_t0.detach(), x_edit.detach(), region_gs_mask))

            d_rotation_bias = d_values.get('d_rotation_bias', None)
            for c in cam_ids:
                cam = test_cams[c]
                out = render(viewpoint_camera=cam, pc=gaussians, pipe=pipe, bg_color=bg,
                             d_xyz=d_values['d_xyz'], d_rotation=d_values['d_rotation'],
                             d_scaling=d_values['d_scaling'], d_opacity=d_values['d_opacity'],
                             d_color=d_values['d_color'], d_rot_as_res=deform.d_rot_as_res,
                             d_rotation_bias=d_rotation_bias)
                img = torch.clamp(out['render'], 0, 1)
                name = f"{args.edit_mode}{args.tag}_a{int(round(angle)):03d}_cam{c}.png"
                torchvision.utils.save_image(img, os.path.join(args.out_dir, 'renders', name))

            np.savez(os.path.join(args.out_dir, 'nodes', f'{args.edit_mode}{args.tag}_a{int(round(angle)):03d}.npz'),
                     init_pcl=init_pcl.cpu().numpy(), deformed=deformed.cpu().numpy(),
                     trans_bias=trans_bias.cpu().numpy(),
                     drag_idx=drag_idx.cpu().numpy(), anchor_idx=anchor_idx.cpu().numpy(),
                     region_nodes=region_nodes.cpu().numpy())

            with open(csv_path, 'a') as f:
                f.write(f"{args.edit_mode},{args.tag},{angle},{nsteps},{solve_time:.4f},{m['arap_error']:.6f},"
                        f"{m['edge_stretch_mean']:.6f},{m['edge_stretch_p95']:.6f},{m['edge_stretch_max']:.6f},"
                        f"{m['edge_stretch_region_mean']:.6f},{m['edge_stretch_region_p95']:.6f},{m['edge_stretch_region_max']:.6f},"
                        f"{m['gs_stretch_mean']:.6f},{m['gs_stretch_p95']:.6f},{m['gs_stretch_max']:.6f},"
                        f"{m['handle_residual']:.6f}\n")
            print(f"[{args.edit_mode}{args.tag}] angle={angle:6.1f} steps={nsteps:4d} "
                  f"arap={m['arap_error']:.4f} edge_region_p95={m['edge_stretch_region_p95']:.4f} "
                  f"gs_p95={m['gs_stretch_p95']:.4f} t={solve_time:.2f}s")


if __name__ == '__main__':
    main()
