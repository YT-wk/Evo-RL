# Evo-RL for reBot B601-DM

This fork of Evo-RL adds a LeRobot-compatible hardware path for the reBot Arm
B601-DM follower and the reBot Arm 102 leader. It supports single and dual
arms, calibration, teleoperation, recording, replay, human-in-the-loop policy
rollouts, value learning, advantage-conditioned policy training, and SwanLab
visualization.

The B601-DM MIT control path computes the SDK URDF's Pinocchio gravity vector
from the latest measured joint positions and sends it as non-zero torque
feedforward for the six arm joints. The gripper remains on its own configured
control path.

## Supported devices

| LeRobot type | Device | Action / observation layout |
| --- | --- | --- |
| `rebot_b601_follower` | One B601-DM follower | 7 joints |
| `bi_rebot_b601_follower` | Two B601-DM followers | `left_*` then `right_*`, 14 joints |
| `rebot_102_leader` | One Arm 102 leader | 7 joints |
| `bi_rebot_102_leader` | Two Arm 102 leaders | `left_*` then `right_*`, 14 joints |

The Arm 102 maps its calibrated joint directions and gripper travel to the
same joint convention and limits as the B601-DM. The standard LeRobot dataset
schema is unchanged, so recording, replay, and training use the normal
`action`, `observation.state`, camera, and episode fields.

### Arm 102 leader

<p align="center">
  <img src="website/assets/images/star-arm-102-hd-leader.png" alt="Fashion Star Arm 102-HD leader arm" width="560" />
</p>

The supported leader is the Fashion Star Star Arm 102-HD, connected through
its UART smart-servo bus.

## Installation

### 1. Create the environment

Use Python 3.10. The `requirements-rebot-pi05.txt` entry point is the tested
combination for this repository: reBot hardware dependencies, the patched
PI05/OpenPI Transformers build, and SwanLab.

```bash
git clone <YOUR_FORK_URL>
cd Evo-RL

conda create -y -n evo-rl python=3.10
conda activate evo-rl

python -m pip install --upgrade pip
python -m pip install -r requirements-rebot-pi05.txt
```

For an existing environment that previously installed a different
Transformers build, reinstall the tested extra so pip replaces incompatible
packages:

```bash
python -m pip install --upgrade --force-reinstall -e '.[rebot-pi05]'
python -m pip check
```

Do **not** install `requirements-ubuntu.txt`, `requirements-macos.txt`, or
`.[all]` into this environment. They target the broad upstream LeRobot feature
set and currently select incompatible requirements for this route (notably
Pinocchio 3.x and standard Transformers). `rebot-pi05` is intentionally
focused on the runnable B601-DM / Arm102 / PI05 workflow.

### 2. Verified dependency set

The following versions were used for the two-A100 value-training, value
inference, and PI05 policy-training smoke run:

| Package | Version |
| --- | --- |
| Python | 3.10 |
| `transformers` | patched `fix/lerobot_openpi` commit `dcddb970176382c0fcf4521b0c0e6fc15894dfe0` (`4.53.3`) |
| `tokenizers` | `0.21.4` |
| `huggingface-hub` | `0.36.2` |
| `pin` (Pinocchio) | `4.1.0` |
| `motorbridge` | `0.5.5` |
| `motorbridge-smart-servo` | `0.0.4` |
| `swanlab` | `0.10.1` |

The custom Transformers build is required by PI05. Using a recent stock
Transformers release can produce an `incorrect transformer version` error;
using a mismatched `huggingface-hub` can fail during import. The pinned extra
prevents both cases.

### 3. Configure B601 gravity feedforward

MIT mode requires the `ReBot_Arm_DM.urdf` from the reBot SDK. Set one of these
variables before connecting a B601-DM:

```bash
# Preferred: path to the SDK root containing urdf/DM/urdf/ReBot_Arm_DM.urdf
export REBOT_GRAVITY_SDK_ROOT=/path/to/reBotArm_control_py

# Or point directly at the URDF.
# export REBOT_GRAVITY_URDF=/path/to/ReBot_Arm_DM.urdf
```

If the model cannot be loaded, or valid joint feedback is unavailable, a MIT
command is rejected rather than silently sending zero gravity torque. Confirm
the gravity-torque signs with the arm mechanically supported before any normal
teleoperation or replay.

### 4. Optional SwanLab login

For cloud visualization, authenticate in the local user account. Do not put a
token in source files, shell history, or the repository.

```bash
swanlab login
```

This stores the login under the user's home directory. The training pipeline
also supports `--swanlab-mode local`, `offline`, or `disabled` when cloud
upload is not desired.

## Hardware setup

On Linux, verify stable device paths and grant the user serial-device access:

```bash
ls -l /dev/serial/by-id/
ls -l /dev/ttyACM* /dev/ttyUSB*
sudo usermod -aG dialout "$USER"
```

Log out and back in after changing group membership. The examples below use
temporary `/dev/ttyACM*` and `/dev/ttyUSB*` paths for clarity; use
`/dev/serial/by-id/...` in long-lived deployments.

### Calibration

Calibrate each follower and leader independently. The `id` becomes part of the
calibration-file identity, so use stable, distinct names for the two sides.

```bash
# B601-DM follower
lerobot-calibrate \
  --robot.type=rebot_b601_follower \
  --robot.port=/dev/ttyACM0 \
  --robot.id=b601_left

# Arm 102 leader
lerobot-calibrate \
  --teleop.type=rebot_102_leader \
  --teleop.port=/dev/ttyUSB0 \
  --teleop.id=arm102_left
```

### Dual-arm teleoperation

Keep the workspace clear for the first run. B601 MIT mode applies non-zero
gravity feedforward, including during the safe-home lifecycle.

```bash
lerobot-teleoperate \
  --robot.type=bi_rebot_b601_follower \
  --robot.left_arm_config.port=/dev/ttyACM0 \
  --robot.right_arm_config.port=/dev/ttyACM1 \
  --robot.id=evorl_b601 \
  --teleop.type=bi_rebot_102_leader \
  --teleop.left_arm_config.port=/dev/ttyUSB0 \
  --teleop.right_arm_config.port=/dev/ttyUSB1 \
  --teleop.id=evorl_arm102
```

For a single arm, replace the two `bi_*` types with `rebot_b601_follower` and
`rebot_102_leader`, and pass `--robot.port` and `--teleop.port`.

## Data collection and replay

### Record demonstrations

This dual-arm example stores a local dataset and associates one wrist camera
with each arm plus a front camera. Adapt the camera paths, task, and episode
settings to the workstation.

```bash
lerobot-record \
  --robot.type=bi_rebot_b601_follower \
  --robot.left_arm_config.port=/dev/ttyACM0 \
  --robot.right_arm_config.port=/dev/ttyACM1 \
  --robot.id=evorl_b601 \
  --robot.left_arm_config.cameras='{wrist: {type: opencv, index_or_path: "/dev/video-left", width: 640, height: 480, fps: 30, fourcc: "MJPG"}}' \
  --robot.right_arm_config.cameras='{wrist: {type: opencv, index_or_path: "/dev/video-right", width: 640, height: 480, fps: 30, fourcc: "MJPG"}, front: {type: opencv, index_or_path: "/dev/video-front", width: 640, height: 480, fps: 30, fourcc: "MJPG"}}' \
  --teleop.type=bi_rebot_102_leader \
  --teleop.left_arm_config.port=/dev/ttyUSB0 \
  --teleop.right_arm_config.port=/dev/ttyUSB1 \
  --teleop.id=evorl_arm102 \
  --dataset.repo_id=local/my_b601_task \
  --dataset.single_task="pick and place an object" \
  --dataset.num_episodes=20 \
  --dataset.episode_time_s=60 \
  --dataset.reset_time_s=5 \
  --dataset.push_to_hub=false \
  --display_data=false
```

### Human-in-the-loop rollout

Add a policy path to record policy rollouts and manual takeovers. The dataset
stores the executed action in `action`, policy output in
`complementary_info.policy_action`, and intervention source/state metadata in
the complementary fields.

```bash
lerobot-human-inloop-record \
  --robot.type=bi_rebot_b601_follower \
  --robot.left_arm_config.port=/dev/ttyACM0 \
  --robot.right_arm_config.port=/dev/ttyACM1 \
  --robot.id=evorl_b601 \
  --teleop.type=bi_rebot_102_leader \
  --teleop.left_arm_config.port=/dev/ttyUSB0 \
  --teleop.right_arm_config.port=/dev/ttyUSB1 \
  --teleop.id=evorl_arm102 \
  --dataset.repo_id=local/my_b601_rollouts \
  --dataset.single_task="pick and place an object" \
  --dataset.num_episodes=20 \
  --dataset.episode_time_s=60 \
  --dataset.reset_time_s=5 \
  --dataset.push_to_hub=false \
  --policy.path=/path/to/pi05_checkpoint
```

The default takeover flow is `i` to enter alignment, `Space` to confirm when
all leader/follower joint errors are within the configured threshold, and `i`
again to return to policy inference. Configure the hotkeys and alignment
threshold with `--intervention_toggle_key`, `--intervention_confirm_key`, and
`--intervention_max_joint_error_deg`; inspect all available options with
`lerobot-human-inloop-record --help`.

After an episode outcome, the B601 returns to the pose measured when the
recording command started. The Arm 102 is released rather than driven to a
reset pose.

### Replay

Replay invokes the same B601 action path as teleoperation and therefore keeps
MIT gravity feedforward enabled:

```bash
lerobot-replay \
  --robot.type=bi_rebot_b601_follower \
  --robot.left_arm_config.port=/dev/ttyACM0 \
  --robot.right_arm_config.port=/dev/ttyACM1 \
  --robot.id=evorl_b601 \
  --dataset.repo_id=local/my_b601_task \
  --dataset.root=/path/to/local/datasets/my_b601_task \
  --dataset.episode=0
```

## Offline Evo-RL training with SwanLab

`scripts/train_evo_rl_pipeline.sh` runs the three stages in order:

1. Value function training (`pistar06`)
2. Value inference and ACP annotation
3. PI05 policy training

It uses `accelerate` for multi-GPU execution, disables Hugging Face network
access for launched processes, writes ACP columns into a copy of the source
dataset by default, and saves checkpoints/logs under the selected output root.
Batch-size arguments are per GPU.

Place the required pretrained weights on local storage first: a PI05
checkpoint, the PaliGemma tokenizer/model directory, the SigLIP vision
backbone, and the Gemma language backbone. Then run, for example:

```bash
bash scripts/train_evo_rl_pipeline.sh \
  --dataset-root /data/datasets/my_b601_rollouts \
  --dataset-repo-id local/my_b601_rollouts \
  --pi05-path /data/checkpoints/pi05_base \
  --paligemma-path /data/weights/google_paligemma-3b-pt-224 \
  --value-vision-path /data/weights/siglip-so400m-patch14-384 \
  --value-language-path /data/weights/gemma-3-270m \
  --output-root /data/evo-rl-runs \
  --job-name b601-smoke \
  --gpus 0,1 \
  --num-processes 2 \
  --value-steps 100 \
  --value-batch-size 1 \
  --infer-batch-size 2 \
  --policy-steps 100 \
  --policy-batch-size 1 \
  --swanlab-mode cloud \
  --swanlab-project evo-rl
```

Use `--stages value_train`, `--stages value_infer`, or
`--stages policy_train` to run one stage. When skipping an earlier stage,
provide its artifact with `--value-checkpoint-path` or
`--annotated-dataset-root`. Run `bash scripts/train_evo_rl_pipeline.sh --help`
for every supported parameter, including `--dry-run`.

For one GPU, use `--gpus 0 --num-processes 1`. The script uses the equivalent
of `CUDA_VISIBLE_DEVICES=<ids> accelerate launch --multi_gpu
--num_processes=<n>` internally; it also handles the `--no_python` form
required by its Python-module launch command.

## Validation checklist

Before a hardware session:

1. Run `python -m pip check` and verify `python -c "import pinocchio, motorbridge, transformers"`.
2. Confirm the SDK URDF path resolves through `REBOT_GRAVITY_SDK_ROOT` or `REBOT_GRAVITY_URDF`.
3. Verify port assignments and calibration IDs for left and right arms.
4. With mechanical support and a clear workspace, check that B601 gravity torque is non-zero and has the correct sign.
5. Start with low-speed, short teleoperation before recording or replay.

## Repository hygiene

The reBot examples do not contain a server address, login token, password,
private key, or local credential. The repository ignores common local
credential files such as `.env.*`, `.netrc`, `.swanlab/`, and private-key
formats. Use command arguments and environment variables for machine-specific
paths.

## License

Apache-2.0. See [LICENSE](LICENSE).
