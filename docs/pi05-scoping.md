# Scoping de pi0.5 (issue #74)

## Licencia y pesos

### Resumen ejecutivo

Los checkpoints de **π0.5 (pi05)** publicados por Physical Intelligence en el
repositorio oficial `openpi` se distribuyen bajo **licencia Apache 2.0**. Apache
2.0 es una licencia permisiva que **permite explícitamente el uso comercial**,
la modificación, la distribución y el uso en despliegues privados/industriales,
sin regalías, siempre que se conserven el aviso de copyright, el texto de la
licencia y el fichero `NOTICE` (si existe) en las redistribuciones. **No hay
cláusula de "solo investigación" ni restricción de uso comercial.**

→ Para el uso previsto (**despliegue industrial comercial**), la licencia **no
supone un bloqueo**.

### Checkpoints disponibles abiertamente

Physical Intelligence libera pesos de π0.5 por dos vías equivalentes: Google
Cloud Storage (`gs://openpi-assets/...`, la fuente que consume el repo `openpi`)
y el Hub de Hugging Face (mirror bajo la organización `lerobot`).

| Checkpoint | Ruta / repo | Uso previsto |
| --- | --- | --- |
| **π0.5 base** | `gs://openpi-assets/checkpoints/pi05_base` · HF: [`lerobot/pi05_base`](https://huggingface.co/lerobot/pi05_base) | Fine-tuning (modelo generalista pre-entrenado en 10k+ h de datos de robot) |
| **π0.5-LIBERO** | `gs://openpi-assets/checkpoints/pi05_libero` · HF: [`lerobot/pi05_libero`](https://huggingface.co/lerobot/pi05_libero) | Inferencia / benchmark LIBERO (SOTA) |
| **π0.5-DROID** | `gs://openpi-assets/checkpoints/pi05_droid` | Inferencia / fine-tuning (fine-tuned sobre DROID con *knowledge insulation*, inferencia rápida y seguimiento de lenguaje) |

Notas:
- π0.5 es la evolución de π0 orientada a **generalización open-world** (publicado
  en septiembre de 2025).
- El código del repo `openpi` está igualmente bajo Apache 2.0; la implementación
  en LeRobot (`policy.type=pi05`) deriva del mismo repositorio y declara la misma
  licencia Apache 2.0 para el modelo.
- `chunk_size` por defecto de π0.5 = 50 (relevante para la pasarela de *chunking*
  de la pilot).

### Diligencia recomendada antes de producción

Aunque la licencia es permisiva, para un despliegue comercial conviene:
1. **Conservar `LICENSE`/`NOTICE`** de openpi en cualquier artefacto redistribuido
   o imagen de contenedor que incluya los pesos.
2. Verificar la **model card** de cada checkpoint en el Hub en el momento de la
   descarga: Apache 2.0 aplica hoy, pero una tarjeta concreta podría añadir
   términos de dataset (p. ej. DROID) o *acceptable use* — revisar antes de
   fijar el pin.
3. Comprobar la licencia de los **datasets** usados si se hace fine-tuning propio
   (los pesos base son Apache 2.0, pero los datos de fine-tuning que aporte el
   proyecto tienen su propia procedencia).

### Fuentes

- Physical-Intelligence/openpi — repositorio y checkpoints (LICENSE = Apache 2.0):
  https://github.com/Physical-Intelligence/openpi
- LICENSE (Apache License, Version 2.0):
  https://raw.githubusercontent.com/Physical-Intelligence/openpi/main/LICENSE
- Physical Intelligence — "Open Sourcing π0" (blog):
  https://www.pi.website/blog/openpi
- Physical Intelligence — π0.5 (open-world generalization):
  https://www.physicalintelligence.company/blog/pi05
- LeRobot docs — π0.5 (Pi05) Policy, "This model follows the Apache 2.0 License":
  https://huggingface.co/docs/lerobot/pi05
- HF checkpoints: [`lerobot/pi05_base`](https://huggingface.co/lerobot/pi05_base),
  [`lerobot/pi05_libero`](https://huggingface.co/lerobot/pi05_libero)

**Conclusión:** π0.5 se publica bajo **Apache 2.0**, que autoriza el uso
comercial e industrial. **No se añade línea de BLOQUEO** — la licencia permite el
uso previsto.

## FAST tokenizer

### Resumen ejecutivo

**FAST (Frequency-space Action Sequence Tokenization) está disponible como
librería reusable — NO requiere reimplementación.** Physical Intelligence lo
publica de dos formas complementarias, ambas bajo **Apache 2.0**:

1. **Como HuggingFace `AutoProcessor`** (checkpoint `physical-intelligence/fast`),
   consumible en 3 líneas vía `transformers` + `scipy`, sin dependencia de openpi.
2. **Wrapper `FASTTokenizer`** dentro de `openpi`
   (`src/openpi/models/tokenizer.py`), que a su vez carga el mismo `AutoProcessor`
   por debajo. Es la vía que usa el propio π0-FAST / π0.5 en entrenamiento.

→ Para la pilot de π0.5, la tokenización de acciones **se reutiliza tal cual**; el
único trabajo es de integración (normalización a `[-1, 1]`, wiring del chunk), no
de reimplementar el algoritmo (DCT + cuantización estilo JPEG).

### Vía A — librería universal vía `transformers` (independiente de openpi)

Requisitos: `pip install transformers scipy`. Checkpoint: `physical-intelligence/fast`
(tokenizer **FAST+** universal, entrenado sobre 1M de secuencias de acción reales).

```python
from transformers import AutoProcessor
import numpy as np

# trust_remote_code=True: el algoritmo FAST viaja como código en el repo del Hub
tokenizer = AutoProcessor.from_pretrained("physical-intelligence/fast", trust_remote_code=True)

action_data = np.random.rand(256, 50, 14)  # (batch, time_horizon, action_dim), normalizado a [-1, 1]
tokens = tokenizer(action_data)             # -> list[int]  (codificación con compresión)
decoded = tokenizer.decode(tokens)          # -> acciones reconstruidas
```

Notas de uso:
- Recomendado para *chunks* de ~1 s **pre-normalizados a `[-1, 1]`**.
- Encode/decode soportan **inferencia por lotes** (batched).
- Se puede **entrenar un tokenizer propio** para un dataset concreto con
  `tokenizer.fit(action_data)`, seguido de `save_pretrained(...)` /
  `push_to_hub(...)`. Útil si las estadísticas de acción del embodiment objetivo
  difieren mucho de FAST+.

### Vía B — wrapper `FASTTokenizer` en openpi (la que usa π0.5)

En `openpi`, la clase `FASTTokenizer` (`src/openpi/models/tokenizer.py`) envuelve
el mismo `AutoProcessor` y añade el ensamblado con el tokenizer de lenguaje del VLM:

```python
# openpi (resumido)
self._fast_tokenizer = AutoProcessor.from_pretrained(fast_tokenizer_path, trust_remote_code=True)
#   fast_tokenizer_path por defecto = "physical-intelligence/fast"

action_tokens = self._fast_tokenizer(actions[None])[0]                    # tokenize()
... = self._fast_tokenizer.decode([action_tokens.tolist()],              # extract_actions()
                                  time_horizon=action_horizon, action_dim=action_dim)[0]
```

Implicación para la pilot: si se apoya en el stack de `openpi` para π0.5, la
tokenización llega **gratis** con el repo (mismo checkpoint del Hub por debajo);
no hay que traer una dependencia extra ni reimplementar nada.

### Cómo funciona (por qué "frequency-space")

FAST comprime cada secuencia de acción mediante: (1) normalización, (2) **DCT
(Discrete Cosine Transform)** por dimensión de acción, y (3) cuantización que
redondea/descarta coeficientes poco significativos — el mismo principio de
compresión que JPEG (imagen) o MP3 (audio). Esto permite entrenar VLA
**autorregresivas** sobre acciones de alta frecuencia/destreza, alcanzando la
destreza de flow-matching/diffusion con ~5× menos tiempo de entrenamiento.

### Fuentes

- Modelo/tokenizer universal en el Hub (Apache 2.0), con ejemplo de uso y `.fit()`:
  https://huggingface.co/physical-intelligence/fast
  · README: https://huggingface.co/physical-intelligence/fast/blob/main/README.md
- Wrapper `FASTTokenizer` en openpi:
  https://github.com/Physical-Intelligence/openpi/blob/main/src/openpi/models/tokenizer.py
- Paper "FAST: Efficient Action Tokenization for Vision-Language-Action Models":
  https://arxiv.org/abs/2501.09747 · PDF: https://www.pi.website/download/fast.pdf
- Página de research (resumen + figuras): https://www.pi.website/research/fast
- Integración en LeRobot (π0-FAST policy, misma licencia Apache 2.0):
  https://huggingface.co/docs/lerobot/pi0fast
  · mirror del tokenizer: https://huggingface.co/lerobot/fast-action-tokenizer

**Conclusión:** FAST **es reusable como librería** (`AutoProcessor` del Hub o el
wrapper `FASTTokenizer` de openpi), Apache 2.0. **No se requiere
reimplementación** — solo integración (normalización + wiring del chunk). **No se
añade línea de BLOQUEO.**

## VRAM footprint (estimado)

> ⚠️ **NO VERIFICADO EN HARDWARE.** Todas las cifras de esta sección son una
> estimación **analítica** (parámetros × precisión + KV cache + overhead de
> runtime). **No** proceden de una medición real en GPU (`nvidia-smi`, perfilado
> de memoria, ni OOM observados). Sirven para dimensionar *a priori* la
> co-residencia; hay que confirmarlas en una L4 real antes de fijar cualquier
> decisión de despliegue.

### Objetivo

¿Caben **π0.5 (pilot)** y un **Specialist tipo Gemma** (juez/grounding,
cuantizado int4) **co-residentes** en una única **NVIDIA L4 de 24 GB**?

### Supuestos de partida

| Parámetro | Valor asumido | Base del supuesto |
| --- | --- | --- |
| Arquitectura π0.5 | PaliGemma-3B (SigLIP ViT ~0.4B + Gemma-2B) + *action expert* ~0.3B | Misma familia que π0/π0.5 en openpi; ≈ **3.3B parámetros** totales |
| Precisión de inferencia π0.5 | **bf16** (2 bytes/parám) | openpi/JAX corre en bf16 por defecto |
| Contexto (tokens de prefijo) | ~1–2k tokens (256 tok/imagen × ~2–3 cámaras + lenguaje) | Config típica de π0.5 multi-cámara |
| `chunk_size` | 50 | Ya documentado arriba (default π0.5) |
| Specialist | Gemma **int4** (≈4B) VLM juez | Nota de memoria: `check_done` = "Gemma int4" |
| Precisión Specialist | int4 (~0.5 byte/parám efectivo + escalas) | Cuantización GPTQ/AWQ-style |

### Desglose analítico

**1. Pesos de π0.5 (bf16)**
- 3.3B × 2 bytes ≈ **6.6 GB**
- (Si se corriese en fp32 se duplicaría a ~13.2 GB → la co-residencia dejaría de
  ser cómoda; **usar bf16/fp16 es un requisito**, no una optimización.)

**2. KV cache de π0.5** — *despreciable*
- Gemma-2B usa MQA (≈1 KV head, head_dim 256, 18 capas).
- Por token: `2 (K,V) × 18 capas × 1 KV head × 256 × 2 bytes ≈ 18 KB/token`.
- Con ~2k tokens de contexto: **~0.04 GB**. Incluso con márgenes generosos, < 0.1 GB.
- Razón: a diferencia de un chat LLM, el contexto es corto y fijo (prefijo de
  imágenes + instrucción); no crece por decodificación larga.

**3. Activaciones / workspace de inferencia (batch 1)**
- Encoder de visión (SigLIP) + atención del backbone + muestreo del *action
  expert* (varios pasos de denoising que reutilizan buffers): **~1–2 GB**.

**4. Overhead de runtime (contexto CUDA + XLA/cuDNN)**
- Contexto CUDA, kernels, cuDNN/cuBLAS workspaces de **dos** frameworks
  co-residentes (JAX para π0.5 + PyTorch para el Specialist): **~1–1.5 GB**.

**5. Specialist Gemma int4 (≈4B)**
- Pesos: 4B × 0.5 byte ≈ 2.0 GB + escalas/zeros del esquema int4 (~+10–15%) ≈
  **~2.3 GB**.
- KV cache + activaciones + (si es VLM) tokens de imagen: **~0.5–0.7 GB**.
- Subtotal Specialist: **~3.0 GB**.
- (Si el Specialist fuese un Gemma-2B int4, bajaría a ~1.2 GB de pesos → ~1.7 GB
  totales.)

### Total estimado

| Componente | VRAM (GB) |
| --- | --- |
| π0.5 pesos (bf16) | 6.6 |
| π0.5 KV cache | ~0.05 |
| π0.5 activaciones/workspace | 1.5 |
| Overhead runtime (CUDA + XLA/cuDNN, 2 frameworks) | 1.5 |
| Specialist Gemma int4 (~4B) | 3.0 |
| **Total** | **≈ 12.6 GB** |

**Sobre L4 (24 GB): ~12.6 GB usados → ~11 GB de holgura (~52% libre).**

Rango con incertidumbre (peor caso de activaciones/overhead y Specialist 4B):
**~11–15 GB**. La co-residencia **cabe con margen cómodo** en la estimación; el
riesgo no es el *tamaño* estático sino la **gestión de asignación** (ver abajo).

### Caveats críticos para la co-residencia (riesgos operativos, no de tamaño)

1. **Preasignación de JAX/XLA (riesgo #1).** Por defecto JAX reserva el **75–90%
   de la VRAM** al arrancar, lo que **mataría por OOM** al Specialist de PyTorch
   aunque el footprint real quepa. **Obligatorio** fijar
   `XLA_PYTHON_CLIENT_PREALLOCATE=false` o
   `XLA_PYTHON_CLIENT_MEM_FRACTION=~0.5` para acotar a π0.5.
2. **Fragmentación entre dos allocators.** JAX y PyTorch mantienen pools de
   memoria independientes; la suma nominal puede caber pero la fragmentación
   reduce el máximo asignable contiguo. Considerar
   `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.
3. **Precisión.** El cálculo asume bf16 para π0.5; en fp32 no cabría con holgura.
4. **Tamaño real del Specialist sin confirmar.** "Gemma int4" no fija variante
   (2B vs 4B vs 3-4B VLM) ni esquema de cuantización — cambia el subtotal en
   ±1.3 GB. Confirmar el checkpoint exacto.
5. **Picos transitorios en carga.** La carga de checkpoints (dequant int4,
   conversión de dtype) puede exigir un pico temporal por encima del estado
   estacionario; secuenciar las cargas (no cargar ambos modelos en paralelo).

### Cómo verificarlo en hardware (pendiente)

- `nvidia-smi --query-gpu=memory.used --format=csv -l 1` durante un rollout
  co-residente.
- `torch.cuda.max_memory_allocated()` para el Specialist y el *memory profiler*
  de JAX (`jax.profiler` / `XLA_FLAGS=--xla_dump...`) para π0.5.
- Prueba de estrés: rollout completo (chunk 50) con ambos modelos activos y
  medición de pico, no de media.

## Mapeo de espacio de acción

### Resumen ejecutivo

La pregunta operativa es: **¿cuál es el `unnorm_key` de π0.5?** Respuesta corta:
π0.5 **no expone un `unnorm_key` en la config de la misión**. El papel que en
OpenVLA juega el string `unnorm_key` (seleccionar, en tiempo de inferencia, qué
estadísticas de des-normalización aplicar a la acción) lo cubre en `openpi` un
fichero **`norm_stats`** que viene **empaquetado dentro del propio checkpoint**
(bajo `assets/<asset_id>/`) y se selecciona por el **`repo_id`/`asset_id`** con el
que se entrenó — no por una clave que el usuario pase en la config. Para
`pi05_libero` ese `asset_id` es `physical-intelligence/libero`.

→ Consecuencia práctica para la pilot: donde OpenVLA lleva
`config.unnorm_key: libero_object`, **π0.5-LIBERO no lleva ninguna clave
equivalente** en el YAML. La normalización correcta viaja con los pesos. El
"knob" análogo solo existe en *entrenamiento* (elegir el `repo_id`/`norm_stats`
del embodiment), no en *evaluación*.

### El análogo exacto de `unnorm_key`

| | OpenVLA (`predict_action`) | π0.5 / `openpi` |
| --- | --- | --- |
| Qué normaliza | La acción de salida (7-D), de espacio normalizado → unidades físicas del robot | Estado de entrada **y** acción de salida (chunk), vía `Normalize`/`Unnormalize` |
| Dónde viven las estadísticas | `dataset_statistics` dentro del `config.json` del checkpoint (varios datasets a la vez) | `norm_stats.json` bajo `assets/<asset_id>/` del checkpoint (uno por embodiment) |
| Cómo se selecciona | String `unnorm_key` pasado en inferencia (`predict_action(..., unnorm_key=...)`) | Implícito: `asset_id = assets.asset_id or repo_id`, fijado al entrenar |
| Superficie en la misión | `config.unnorm_key` (obligatorio, debe casar suite/checkpoint) | **ninguna** — el checkpoint es autocontenido |
| Fallo típico | `unnorm_key` mal → acciones a escala equivocada, arm errático | (no aplica en eval) checkpoint/repo_id equivocado al entrenar |

Referencia en el repo del lado OpenVLA: `src/odyssey/runners/models/openvla.py:481`
(`unnorm_key = cfg.get("unnorm_key", "bridge_orig")`) y su paso a
`model.predict_action(..., unnorm_key=self._unnorm_key)`
(`openvla.py:674`).

### Espacio de observación de π0.5-LIBERO

`openpi` normaliza la observación LIBERO al mismo *embodiment* Franka Panda que
consume el resto del stack (robosuite/LIBERO), a través de `LiberoInputs`:

- **Imágenes** (3 entradas del modelo, familia PaliGemma/Pi0):
  - `base_0_rgb` ← vista cenital de tercera persona (`observation/image`, `agentview`).
  - `left_wrist_0_rgb` ← cámara de muñeca (`observation/wrist_image`).
  - `right_wrist_0_rgb` ← **rellena a ceros** (LIBERO es de un brazo; se enmascara
    con `image_mask`).
  - Formato normalizado a `uint8 (H, W, C)`.
- **Estado propioceptivo, 8-D** (idéntico a la convención Franka Panda de
  robosuite/LIBERO):

  ```python
  state = np.concatenate([
      obs["robot0_eef_pos"],                    # (3) posición EEF x,y,z
      quat2axisangle(obs["robot0_eef_quat"]),   # (3) orientación EEF: quat xyzw -> eje-ángulo
      obs["robot0_gripper_qpos"],               # (2) qpos de las dos falanges del gripper
  ])                                            # -> 8-D
  ```

  Este es exactamente el mismo vector de estado que construye el eval de LIBERO de
  NVIDIA para GR00T (`quat_xyzw_to_axis_angle` en
  `src/odyssey/runners/evals/gr00t_transforms.py:172`) y que OpenVLA asume
  implícitamente. La conversión clave de convención es
  **quaternion `xyzw` (robosuite) → eje-ángulo** vía `quat2axisangle`.
- **Padding a la dimensión del modelo:** el estado 8-D se rellena con ceros hasta
  `action_dim` del modelo (**32** en la familia Pi0) mediante `pad_to_dim` /
  `PadStatesAndActions`, y luego se **normaliza** con `norm_stats`. El *prompt* de
  lenguaje se toma de la tarea (`prompt_from_task=True`).

### Espacio de acción de π0.5-LIBERO

El modelo emite un **chunk** de acciones normalizadas de forma
`(action_horizon, 32)`; `LiberoOutputs` lo des-normaliza (con el mismo
`norm_stats`) y **recorta las 32 dims al 7-DoF de LIBERO**:

```python
# openpi LiberoOutputs
return {"actions": np.asarray(data["actions"][..., :7])}   # el resto es padding
```

Las 7 dims son la acción **OSC_POSE** nativa de LIBERO/Franka Panda:
`[dx, dy, dz, droll, dpitch, dyaw, gripper]` (delta de pose EEF + gripper), que se
aplica **directamente** a `env.step(action.tolist())`.

**Diferencia crítica frente a OpenVLA — el gripper NO se re-procesa.** En OpenVLA
el eval de odyssey aplica `_libero_action` (`src/odyssey/runners/evals/libero.py:159`):
re-escala el gripper `[0,1]→[-1,1]`, lo **binariza** y lo **invierte** para casar
el signo de LIBERO. En π0.5/`openpi` **no hay** ese fix-up en evaluación: el
checkpoint se entrenó sobre datos ya en la convención de gripper de LIBERO y las
`norm_stats` fijan la escala, así que la salida se pasa tal cual a `env.step`
(igual criterio que `examples/libero/main.py` de openpi, sin `binarize`/`invert`).
Contrasta también con GR00T-N1.7-LIBERO, que **sí** aplica
`normalize_gripper_action` + `invert_gripper_action`
(`gr00t_transforms.py:218-228`) porque su checkpoint emite el gripper en `[0,1]`.

> ⚠️ **Verificar la polaridad del gripper en el primer rollout en GPU.** Aunque la
> teoría dice "sin fix-up", un signo de gripper invertido es un fallo silencioso
> clásico (el brazo se mueve y se aproxima pero nunca agarra). Es el mismo footgun
> anotado para GR00T; confirmarlo contra el servidor π0.5 antes de fijar la config.

### Tabla comparativa de los tres pilots sobre LIBERO/Franka Panda

| | OpenVLA-7B | GR00T-N1.7-LIBERO | **π0.5-LIBERO** |
| --- | --- | --- | --- |
| Estado de entrada | (implícito) | 8-D: eef_pos + quat→axis-angle + 2 gripper qpos | 8-D: eef_pos + quat→axis-angle + 2 gripper qpos |
| Acción emitida | 1 paso 7-D | chunk 7-D (`action.*`, absoluta) | **chunk** `(H, 32)` → recorte 7-D |
| Espacio de acción | OSC_POSE 7-DoF | OSC_POSE 7-DoF | OSC_POSE 7-DoF `[dx,dy,dz,droll,dpitch,dyaw,g]` |
| Selección de norm-stats | `unnorm_key` en config | (baked en checkpoint) | `norm_stats`/`asset_id` baked (`physical-intelligence/libero`) |
| Fix-up de gripper en eval | binariza + invierte (`_libero_action`) | normaliza + invierte | **ninguno** (baked en datos + norm_stats) |
| `action_horizon` (chunk) | 1 | 40 (`ACTION_HORIZON`) | 10 (`pi05_libero`), 50 default de π0.5 |

Nota sobre el horizonte: el default general de π0.5 es `chunk_size = 50` (ver
sección de licencia/pesos), pero el `TrainConfig` `pi05_libero` de openpi usa
`action_horizon=10`. Relevante para la pasarela de *chunking* de la pilot: el nº
de acciones consumidas por consulta al pilot depende del checkpoint concreto, no
de una constante global.

### Implicaciones para la integración en odyssey

1. **No añadir `unnorm_key` al YAML de π0.5.** El checkpoint `pi05_libero` es
   autocontenido; la des-normalización viaja en `assets/`. Cualquier clave de
   normalización en la config de una misión π0.5 sería inerte o engañosa.
2. **El adaptador de acción de π0.5 recorta a 7-D y aplica sin fix-up de gripper**
   — a diferencia del `VLARuntime`/`_libero_action` de OpenVLA. El
   `ChunkPilotAdapter` (ver nota de multiagente) debe tratar π0.5 como
   *chunk-emitting* y **no** reintroducir binarización/inversión de gripper salvo
   que el smoke en GPU demuestre polaridad invertida.
3. **La convención de estado es la misma Franka Panda** (8-D, quat xyzw→axis-angle,
   2 gripper qpos), así que la construcción de observación se puede compartir con
   el path de GR00T-LIBERO (`build_gr00t_libero_obs`) reordenando a las claves que
   espera `openpi` (`observation/image`, `observation/wrist_image`,
   `observation/state`), sin re-derivar la cinemática.

### Fuentes

- `openpi` — transforms LIBERO (`LiberoInputs`/`LiberoOutputs`, estado 8-D, recorte 7-D):
  https://github.com/Physical-Intelligence/openpi/blob/main/src/openpi/policies/libero_policy.py
- `openpi` — eval LIBERO (construcción de estado desde robosuite, `env.step` sin
  inversión de gripper):
  https://github.com/Physical-Intelligence/openpi/blob/main/examples/libero/main.py
- `openpi` — config de entrenamiento (`LeRobotLiberoDataConfig`,
  `asset_id = assets.asset_id or repo_id`, `norm_stats`, `pi05_libero`):
  https://github.com/Physical-Intelligence/openpi/blob/main/src/openpi/training/config.py
- Convención Franka Panda / gripper en el repo: `src/odyssey/runners/evals/libero.py:159`
  (`_libero_action`), `src/odyssey/runners/evals/gr00t_transforms.py:172,218-270`
  (estado y acción LIBERO_PANDA), `src/odyssey/runners/models/openvla.py:481,674`
  (`unnorm_key`).

**Conclusión:** el análogo de `unnorm_key` en π0.5 es el par
`asset_id`/`norm_stats` **empaquetado en el checkpoint** — no una clave de
misión. La observación (8-D Franka Panda: eef_pos + quat→axis-angle + 2 gripper
qpos) y la acción (7-DoF OSC_POSE, recorte de las 32 dims del modelo) casan con la
convención LIBERO/robosuite que ya usan OpenVLA y GR00T; la única diferencia
operativa es que π0.5 **no requiere fix-up de gripper en evaluación** (pendiente
de confirmar polaridad en GPU). **No se añade línea de BLOQUEO.**
