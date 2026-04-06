#!/usr/bin/env python3
"""
ACT policy → MuJoCo G1+Dex3 sim rollout
체크포인트 로드 후 시뮬레이션에서 grasp 테스트

수정 이력:
  - 체크포인트 경로 수정 (act_g1 → act_g1_single_cam)
  - 단일 카메라 (cam_left_high만) — final.pt는 single cam 모델
  - Renderer 루프 밖에서 한 번만 생성
  - state 읽기: joint name 리스트로 qpos 직접 접근
"""
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np
import torch

from lerobot.configs.policies import FeatureType, PolicyFeature
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.act.modeling_act import ACTPolicy

# ── 설정 ──────────────────────────────────────────────────────────────────────
MJCF_PATH    = str(Path(__file__).parent.parent / "envs/g1_grasp_scene.xml")
CKPT_PATH    = str(Path(__file__).parent.parent / "checkpoints/act_g1_single_cam/final.pt")
DATASET_ROOT = str(Path(__file__).parent.parent / "data/G1_Dex3_GraspSquare_v3")
REPO_ID      = "unitreerobotics/G1_Dex3_GraspSquare_Dataset"
CAMERA_KEY   = "observation.images.cam_left_high"   # single cam — dual 아님
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"
FREQ         = 30    # Hz
N_EPISODES   = 5
STEPS_PER_EP = 300   # 10초 @ 30Hz

# ── policy output 28D → MuJoCo actuator index 매핑 ────────────────────────────
# actuator 순서 (g1_grasp_scene.xml <actuator> 블록 0-based):
#   idx 0-11  : 다리 12개 (policy 미사용)
#   idx 12-14 : 허리 3개  (policy 미사용)
#   idx 15-21 : 왼팔 7개
#   idx 22-28 : 왼손 7개
#   idx 29-35 : 오른팔 7개
#   idx 36-42 : 오른손 7개
POLICY_TO_ACTUATOR_IDX = [
    15, 16, 17, 18, 19, 20, 21,   # action[ 0: 7] → 왼팔
    29, 30, 31, 32, 33, 34, 35,   # action[ 7:14] → 오른팔
    22, 23, 24, 25, 26, 27, 28,   # action[14:21] → 왼손
    36, 37, 38, 39, 40, 41, 42,   # action[21:28] → 오른손
]

# ── state 28D joint 이름 (실로봇 state_28 순서와 1:1 대응) ────────────────────
# run_inference_state_only.py의 STATE_NAMES_28 순서와 동일
JOINT_NAMES_28 = [
    # 왼팔 7
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
    "left_elbow_joint", "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
    # 오른팔 7
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
    "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
    # 왼손 7
    "left_hand_thumb_0_joint", "left_hand_thumb_1_joint", "left_hand_thumb_2_joint",
    "left_hand_middle_0_joint", "left_hand_middle_1_joint",
    "left_hand_index_0_joint", "left_hand_index_1_joint",
    # 오른손 7
    "right_hand_thumb_0_joint", "right_hand_thumb_1_joint", "right_hand_thumb_2_joint",
    "right_hand_index_0_joint", "right_hand_index_1_joint",
    "right_hand_middle_0_joint", "right_hand_middle_1_joint",
]
assert len(JOINT_NAMES_28) == 28


def load_policy() -> ACTPolicy:
    print(f"데이터셋 로드 (stats 추출): {DATASET_ROOT}")
    ds = LeRobotDataset(
        repo_id=REPO_ID,
        root=DATASET_ROOT,
        video_backend="pyav",
        tolerance_s=0.04,
        delta_timestamps={
            "observation.state": [0],
            CAMERA_KEY: [0],
            "action": [i / 30.0 for i in range(100)],
        },
    )

    cfg = ACTConfig(
        input_features={
            CAMERA_KEY: PolicyFeature(type=FeatureType.VISUAL, shape=(3, 480, 640)),
            "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(28,)),
        },
        output_features={
            "action": PolicyFeature(type=FeatureType.ACTION, shape=(28,)),
        },
        chunk_size=100,
        n_action_steps=100,
        vision_backbone="resnet18",
        dim_model=512,
        use_vae=True,
        device=DEVICE,
    )

    policy = ACTPolicy(cfg, dataset_stats=ds.meta.stats)
    ckpt = torch.load(CKPT_PATH, map_location="cpu")
    policy.load_state_dict(ckpt["model_state_dict"])
    policy.to(DEVICE)
    policy.eval()
    print(f"체크포인트 로드 완료: step={ckpt.get('step', 'N/A')}  device={DEVICE}")
    return policy


def build_qpos_indices(model: mujoco.MjModel) -> list[int]:
    """JOINT_NAMES_28 → qpos 인덱스 리스트 (초기화 시 1회만 호출)"""
    indices = []
    for name in JOINT_NAMES_28:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if jid == -1:
            raise ValueError(f"joint '{name}' not found in model")
        indices.append(model.jnt_qposadr[jid])
    return indices


def get_obs(
    data: mujoco.MjData,
    qpos_indices: list[int],
    renderer: mujoco.Renderer,
) -> dict:
    """MuJoCo sim → policy 입력 dict"""
    # state 28D
    state = np.array([data.qpos[i] for i in qpos_indices], dtype=np.float32)

    # 카메라 렌더링 (Renderer는 루프 밖에서 한 번만 생성)
    renderer.update_scene(data, camera=CAMERA_KEY.split(".")[-1])  # "cam_left_high"
    img = renderer.render().copy()   # RGB (H, W, 3) uint8

    img_t = torch.from_numpy(img).float() / 255.0
    img_t = img_t.permute(2, 0, 1).unsqueeze(0).to(DEVICE)   # (1,3,H,W)

    return {
        "observation.state": torch.from_numpy(state).unsqueeze(0).to(DEVICE),
        CAMERA_KEY: img_t,
    }


def run_rollout() -> None:
    policy = load_policy()

    print(f"\nMuJoCo 로드: {MJCF_PATH}")
    model = mujoco.MjModel.from_xml_path(MJCF_PATH)
    data  = mujoco.MjData(model)
    print(f"actuator 수: {model.nu}  |  qpos 수: {model.nq}")

    # qpos 인덱스 초기화 (1회)
    qpos_indices = build_qpos_indices(model)
    print(f"joint qpos indices: {qpos_indices}")

    # Renderer 1회 생성 (루프 밖)
    renderer = mujoco.Renderer(model, height=480, width=640)

    success_count = 0

    with mujoco.viewer.launch_passive(model, data) as viewer:
        for ep in range(N_EPISODES):
            print(f"\n{'='*50}")
            print(f"Episode {ep+1}/{N_EPISODES}")
            mujoco.mj_resetData(model, data)
            policy.reset()

            for step in range(STEPS_PER_EP):
                t0 = time.perf_counter()

                obs = get_obs(data, qpos_indices, renderer)

                with torch.no_grad():
                    action = policy.select_action(obs)

                action_np = action.cpu().numpy().flatten()   # (28,)

                # policy 28D → actuator ctrl
                for i, act_idx in enumerate(POLICY_TO_ACTUATOR_IDX):
                    data.ctrl[act_idx] = action_np[i]

                mujoco.mj_step(model, data)
                viewer.sync()

                if step % 30 == 0:
                    state_np = np.array([data.qpos[i] for i in qpos_indices])
                    print(
                        f"  ep={ep+1} step={step:3d}  "
                        f"arm_l[:3]={np.array2string(state_np[:3], precision=3, floatmode='fixed')}  "
                        f"action_max={float(np.max(np.abs(action_np))):.4f}"
                    )

                elapsed = time.perf_counter() - t0
                time.sleep(max(0.0, 1.0 / FREQ - elapsed))

            print(f"Episode {ep+1} 완료")

    renderer.close()
    print(f"\n완료. 총 {N_EPISODES}회 실행")


if __name__ == "__main__":
    run_rollout()
