# CR-Pro-Coach 🏆
**顶级《皇室战争》AI 专属私人陪练与复盘教练 (Clash Royale Pro Coach App)**

专为 13,050+ 顶级天梯对局研发。融合学术界开源权重、轻量级高速模拟器、人类反应延迟模拟器以及无侵入式只读复盘引擎。

---

## 目录结构 (Architecture & File Tree)

```text
CR-Pro-Coach/
│
├── 📂 configs/                       # 【1. 动态版本配置中心 - 随时同步 16 级与更新】
│   ├── cards_stats_v2026.json        # 全卡牌基础数值 + 11级/16级缩放系数 (Level Scaling)
│   ├── mechanics_tags.json           # 原子化机制标签 (觉醒 Evolutions、英雄技能、冲锋等)
│   ├── meta_decks_top200.json        # 当前赛季天梯 Top 200 卡组库 (用于贝叶斯猜牌)
│   └── human_reaction_profile.yaml   # 人类生理极限参数 (突发延迟 450ms、悬停预判 150ms)
│
├── 📂 pretrained_weights/            # 【2. 从 Hugging Face 下载的现成模型权重】
│   ├── download_weights.py           # 一键自动拉取模型权重的 Python 脚本
│   ├── policy_bc_v4.pt               # 下载自 Cochon123/clash-royale-policy-bc-v4 (仅 2.3MB)
│   ├── action_clock_v1.pt            # 下载自 Cochon123/clash-royale-action-clock-v1 (人类延迟控制)
│   ├── winner_predictor.pt           # 下载自 Cochon123/clash-royale-winner-predictor (胜率曲线引擎)
│   ├── katacr_yolov8_detector.onnx   # 真实游戏画面只读识别模型 (用于捕获屏幕复盘)
│   └── pro_selfplay_ppo_latest.onnx  # 你在云端 Colab 训练好导回 Mac 的主力陪练模型
│
├── 📂 backend_engine/                # 【3. Python 核心大脑与模拟器 (跑在 Mac 后台，占内存 <500MB)】
│   ├── 📂 simulator/                 # 改编自 ClashAI / RoyaleSim 的高速逻辑内核
│   │   ├── arena_grid.py             # 18x32 竞技场网格与寻路碰撞逻辑 (36x64 半格精度)
│   │   ├── unit_ecs.py               # 实体组件系统 (符文巨人强化、小王子救援、地狱龙激光蓄力)
│   │   └── sync_meta.py              # 一键从 RoyaleAPI 拉取最新平衡性补丁的脚本
│   │
│   ├── 📂 perception/                # 严格遵守人类视角的感知层 (不作弊)
│   │   ├── opp_elixir_count.py       # 源自 ClashAI：根据下牌事件实时推算敌方圣水与卡序
│   │   ├── bayesian_deck_guesser.py  # 贝叶斯后验概率推算对手剩余未知手牌
│   │   └── screen_reader_cv.py       # 只读模式：Mac 屏幕捕获 + ONNX 识别 (绝对不发点击指令)
│   │
│   ├── 📂 brain/                     # 决策与教练复盘核心
│   │   ├── human_delay_buffer.py     # 结合 action_clock_v1 的 0.5s 人类反应缓冲队列
│   │   ├── ppo_agent.py              # 加载策略模型 + Action Masking (屏蔽违规位置与不够费的牌)
│   │   ├── fast_mcts_rollout.py      # 关键团战向前推演 6 秒的蒙特卡洛搜索器
│   │   └── coach_analyzer.py         # 赛后复盘：计算每一步的胜率差 (Blunder 检测) 与最佳落点
│   │
│   └── server_bridge.py              # 本地 WebSocket / JSON-RPC 服务 (连接 Unity 前端)
│
├── 📂 cloud_training/                # 【4. 云端炼丹脚本 (丢到 Google Colab 跑，不消耗 Mac 寿命)】
│   ├── train_imitation_bc.ipynb      # 阶段1：读取高玩 Replay 录像进行行为克隆
│   ├── train_selfplay_ppo.ipynb      # 阶段2：基于 ClashAI rl_royale.py 的自博弈强化学习
│   └── gauntlet_eval.py              # 源自 ClashAI e1_eval.py：新旧模型车轮战晋级测试
│
└── 📂 frontend_app/                  # 【5. Unity (C#) 独立客户端界面】
    ├── Assets/Scripts/ArenaUI/       # 18x32 棋盘渲染、拖拽下牌、11级/16级满级一键切换按钮
    ├── Assets/Scripts/DeckBuilder/   # 全卡牌自由组卡器 (支持直接粘贴皇室战争卡组链接)
    └── Assets/Scripts/CoachOverlay/  # 底部实时胜率曲线、幽灵最优解虚影 (Ghost Tile)、一键接管重打按钮
```

---

## 快速上手 (Quickstart)

### 1. 下载开源模型资产
```bash
python pretrained_weights/download_weights.py
```

### 2. 启动后台大脑服务
```bash
python backend_engine/server_bridge.py
```

### 3. 测试 18x32 网格与符文巨人战斗系统
```bash
python -c "from backend_engine.simulator.arena_grid import ArenaGrid; print('Grid loaded:', ArenaGrid.legal_placement_mask()[:5])"
python -c "from backend_engine.perception.bayesian_deck_guesser import BayesianDeckGuesser; g = BayesianDeckGuesser(); print('Bayesian decks:', g.infer(['battle_ram', 'bandit'])[0][:2])"
```
