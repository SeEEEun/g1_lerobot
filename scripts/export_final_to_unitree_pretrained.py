#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export custom ACT final.pt to a Unitree/LeRobot pretrained directory")
    parser.add_argument('--checkpoint', default='/home/jairlab/g1_lerobot/checkpoints/act_g1/final.pt')
    parser.add_argument('--dataset-root', default='/home/jairlab/g1_lerobot/data/G1_Dex3_GraspSquare')
    parser.add_argument('--dataset-repo-id', default='unitreerobotics/G1_Dex3_GraspSquare_Dataset')
    parser.add_argument('--output-dir', default='/home/jairlab/g1_lerobot/checkpoints/act_g1_unitree_pretrained')
    parser.add_argument('--device', default='cuda')
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    repo_root = Path('/home/jairlab/g1_lerobot/unitree_IL_lerobot/unitree_lerobot/lerobot/src')
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    import torch
    from lerobot.configs.policies import FeatureType, PolicyFeature
    from lerobot.policies.act.configuration_act import ACTConfig
    from lerobot.policies.act.modeling_act import ACTPolicy

    checkpoint_path = Path(args.checkpoint)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cfg = ACTConfig(
        input_features={
            'observation.images.cam_left_high': PolicyFeature(type=FeatureType.VISUAL, shape=(3, 480, 640)),
            'observation.images.cam_right_high': PolicyFeature(type=FeatureType.VISUAL, shape=(3, 480, 640)),
            'observation.state': PolicyFeature(type=FeatureType.STATE, shape=(28,)),
        },
        output_features={
            'action': PolicyFeature(type=FeatureType.ACTION, shape=(28,)),
        },
        chunk_size=100,
        n_action_steps=100,
        vision_backbone='resnet18',
        dim_model=512,
        use_vae=True,
        push_to_hub=False,
        device=args.device,
    )

    print(f'Loading checkpoint from {checkpoint_path}')
    raw = torch.load(checkpoint_path, map_location='cpu')
    state_dict = raw.get('model_state_dict', raw)

    policy = ACTPolicy(cfg)
    missing, unexpected = policy.load_state_dict(state_dict, strict=False)
    print(f'Loaded checkpoint with {len(missing)} missing keys and {len(unexpected)} unexpected keys')
    if missing:
        print('Missing keys:')
        for key in missing:
            print(f'  - {key}')

    policy.save_pretrained(output_dir)
    print(f'Saved pretrained policy to {output_dir}')
    print('Contents:')
    for path in sorted(output_dir.iterdir()):
        print(f'  - {path.name}')


if __name__ == '__main__':
    main()
