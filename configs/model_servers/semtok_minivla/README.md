---
smoke_config: p0_stabilized_robotwin.yaml
---

# SemTok MiniVLA

Local SemTok/OpenVLA-Mini bridge for RoboTwin 2.0 evaluation.

## Configs

| File | Benchmark | Policy |
|------|-----------|--------|
| `p0_stabilized_robotwin.yaml` | RoboTwin 2.0 | SemTok P0 stabilized |

## Local Smoke

```bash
vla-eval test --server semtok_minivla --timeout 120
```

For a short local RoboTwin preflight without Docker, run the server and then
the benchmark from the RoboTwin simulator environment:

```bash
vla-eval serve -c configs/model_servers/semtok_minivla/p0_stabilized_robotwin.yaml --address 127.0.0.1:8010

ROBOTWIN_ROOT=/home/shkim_rllab/Desktop/RoboTwin \
  /home/shkim_rllab/Desktop/RoboTwin/.venv/bin/vla-eval run \
  -c configs/benchmarks/robotwin/p0_pick_diverse_bottles_smoke.yaml \
  --no-docker --server-url ws://127.0.0.1:8010 --no-save
```
