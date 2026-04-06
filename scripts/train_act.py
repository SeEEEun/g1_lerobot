#!/usr/bin/env python3
import torch, random, time
from torch.utils.data import DataLoader
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.configs.policies import PolicyFeature, FeatureType
from pathlib import Path

DEVICE = "cuda"
BATCH_SIZE = 4
LR = 1e-4
STEPS = 100000
SAVE_FREQ = 10000
LOG_FREQ = 200
CKPT_DIR = Path("./checkpoints/act_g1")
CKPT_DIR.mkdir(parents=True, exist_ok=True)

print("데이터셋 로드 중...")
ds = LeRobotDataset(
    repo_id="unitreerobotics/G1_Dex3_GraspSquare_Dataset",
    root="./data/G1_Dex3_GraspSquare",
    video_backend="pyav",
    tolerance_s=0.04,
    delta_timestamps={
        "observation.state": [0],
        "observation.images.cam_left_high": [0],
        "observation.images.cam_right_high": [0],
        "action": [i / 30.0 for i in range(100)],
    },
)

# 손상 프레임 스킵 패치
original_getitem = ds.__class__.__getitem__
def safe_getitem(self, idx):
    for _ in range(50):
        try:
            return original_getitem(self, idx)
        except Exception:
            idx = random.randint(0, len(self) - 1)
    raise RuntimeError("50번 시도 후 실패")
ds.__class__.__getitem__ = safe_getitem

loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
print(f"로드 완료: {ds.num_episodes} episodes, {len(ds)} frames")

cfg = ACTConfig(
    input_features={
        "observation.images.cam_left_high": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 480, 640)),
        "observation.images.cam_right_high": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 480, 640)),
        "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(28,)),
    },
    output_features={
        "action": PolicyFeature(type=FeatureType.ACTION, shape=(28,)),
    },
    chunk_size=100, n_action_steps=100,
    vision_backbone="resnet18", dim_model=512, use_vae=True, device=DEVICE,
)

policy = ACTPolicy(cfg, dataset_stats=ds.meta.stats)
policy.to(DEVICE)
policy.train()
optimizer = torch.optim.AdamW(policy.parameters(), lr=LR, weight_decay=1e-4)

print(f"\n훈련 시작 (steps={STEPS}, batch={BATCH_SIZE})\n{'='*50}")
step, running_loss, t0 = 0, 0.0, time.time()

while step < STEPS:
    for batch in loader:
        if step >= STEPS:
            break
        batch = {k: v.to(DEVICE) if hasattr(v, 'to') else v for k, v in batch.items()}
        if batch["observation.state"].dim() == 3:
            batch["observation.state"] = batch["observation.state"].squeeze(1)
        loss, info = policy.forward(batch)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 10.0)
        optimizer.step()
        running_loss += loss.item()
        step += 1
        if step % LOG_FREQ == 0:
            print(f"step {step:6d} | loss {running_loss/LOG_FREQ:.4f} | {time.time()-t0:.0f}s")
            running_loss = 0.0
        if step % SAVE_FREQ == 0:
            ckpt_path = CKPT_DIR / f"step_{step:06d}.pt"
            torch.save({"step": step, "model_state_dict": policy.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict()}, ckpt_path)
            print(f"  체크포인트 저장: {ckpt_path}")

torch.save({"step": step, "model_state_dict": policy.state_dict(),
            "optimizer_state_dict": optimizer.state_dict()}, CKPT_DIR / "final.pt")
print(f"\n훈련 완료 → {CKPT_DIR}/final.pt")
