# Examples

| Directory | What it does | Hardware |
|---|---|---|
| `quickstart-openvla/` | OpenVLA 7B LoRA fine-tune on Bridge V2 + Robosuite Lift eval. | 24 GB GPU |

More quickstarts (Octo, Pi0.5) arrive in later releases. See the
publication plan for the cadence.

## Validating an example

Once `lovell-odyssey` is installed:

```bash
odyssey validate examples/quickstart-openvla/mission.yaml
```

`odyssey run` is not implemented yet in v0.0.x.
