# Pretrained Weights Directory

This folder holds the lightweight, ready-to-use neural network checkpoints downloaded from Hugging Face and GitHub.

| File Name | Source / Repository | Size | Core Role in Coach App |
|:---|:---|:---|:---|
| `policy_bc_v4.pt` | `Cochon123/clash-royale-policy-bc-v4` | 2.27 MB | **Stage 1 Imitation Brain**: Pre-trained behavior cloning policy for card and grid placement. |
| `action_clock_v1.pt` | `Cochon123/clash-royale-action-clock-v1` | ~1.5 MB | **Human Latency Controller**: Delays AI reaction to win-conditions by 300–600ms to mimic human reflex. |
| `winner_predictor.pt` | `Cochon123/clash-royale-winner-predictor` | ~3.8 MB | **Win-Rate Graph Engine**: Generates the real-time $[0\%, 100\%]$ probability curve for post-match analysis. |
| `katacr_yolov8_detector.onnx` | `wty-yy/KataCR` | ~12 MB | **Non-Invasive Vision Engine**: Reads units, towers, and HP bars from the live Mac screen without clicking. |
| `pro_selfplay_ppo_latest.onnx` | Cloud Colab Export | ~5 MB | **Superhuman Sparring Partner**: Exported policy trained via self-play PPO for local sparring. |

### Download Command
```bash
python pretrained_weights/download_weights.py
```
