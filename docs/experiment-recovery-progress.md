# Experimento de recovery closed-loop — diario de progreso

> Documento de seguimiento (en castellano) para no perder de vista el experimento.
> El documento técnico de diseño (en inglés, con todo el detalle) es
> [`experiment-recovery-design-vla.md`](experiment-recovery-design-vla.md).
> Rama: `experiment-vla-recovery-closed-loop` (off develop). Última actualización: 2026-08-12.

## La pregunta del experimento

¿Puede un agente SPECIALIST (un VLM servido) que observa la simulación *mientras*
el PILOT ejecuta action chunks, detectar que el brazo está atascado y disparar
una recuperación — volver al último chunk bueno y reintentar — que convierta
fallos en éxitos? ¿Y qué añade la semántica del VLM sobre señales más baratas
(pura cinemática)?

Marco: integra el paper **VLA-Corrector** (arXiv 2607.01804,
github.com/ZJU-OmniAI/vla-corrector) en Odyssey. Su *truncación* ya está
implementada; su detector entrenado (LVM) será el PR 2; su OGG nunca (necesita
gradientes, imposible con pilots servidos).

## Cómo funciona (la mecánica en 10 líneas)

GR00T emite chunks de 16 acciones que se ejecutan a ciegas. Nuestro sistema se
engancha a las **fronteras entre chunks** (~cada 0.8 s) y en cada una:

```
frontera de chunk
├── ChunkLedger:     snapshot (pose del brazo + frame) → los puntos de retorno
├── StuckMonitor:    física: ¿el brazo empuja pero no se mueve? (tier 1, gratis)
└── SpecialistGate:  2 frames al VLM en background: "¿brazo congelado entre
                     estos dos instantes?" (async — la sim nunca espera)
```

Al disparar, dos medicinas: **flush** (tirar el resto del chunk y re-preguntar
a la política desde donde está — la truncación de VLA-Corrector, vía
`ChunkPilotAdapter.flush()`) o **rollback** (conducir el brazo al último
snapshot con veredictos limpios y re-preguntar desde ahí). Todo tras flags
default-off: apagado = comportamiento byte-idéntico al de siempre.

## Qué se ha hecho (línea temporal)

| Fecha | Hito | Resultado |
| --- | --- | --- |
| 2026-08-11 | **PR 1 implementado** (9 commits): núcleo puro (`recovery.py`, `specialist_gate.py`), `flush()`, recipe GR00T migrado a `ChunkPilotAdapter` (equivalencia pinneada), ambos recipes LIBERO cableados, runner + artefactos, 2 ejemplos | ~90 tests nuevos, suite en verde, mypy strict limpio. PR #96 cerrado a propósito: el trabajo vive como rama |
| 2026-08-11 | **Fase 0** — calibración del judge en la H100 (Cosmos3-Nano servido por vLLM, sobre vídeos grabados) | La pregunta de stuck de **1 frame es inútil** (NO a todo — "atascado" es temporal, un frame no lo muestra). El compare de **2 frames** discrimina limpio (activo→NO 8/8, atascado→YES 10/10) |
| 2026-08-12 | **Commit de dos frames**: el specialist envía pares (frame anterior de frontera, frame actual); judge multi-imagen; prompt frozen-compare validado; knob `stuck_pair_gap_chunks` | El hallazgo de fase 0 convertido en código (`4f38954`) |
| 2026-08-12 | **Fase 2 shadow, episodios limpios** (GR00T auto-servido, `libero_object` task 0, 10 eps; specialist = **RoboBrain2.5-8B-NV** reutilizando el vLLM ya servido en la caja — el judge es agnóstico al modelo) | 10/10 PASS ambos runs. Con gap 1 (~0.8 s): 10 falsos positivos — el juez confunde la **micro-pausa del grasp** con atasco. Con **gap 2 (~1.6 s): 0 FP** al mismo ritmo de consulta. ~40 llamadas async al judge, 0 errores, la sim nunca se bloqueó |
| 2026-08-12 | **Fase 2 shadow, fallos inducidos** (`object_dx: 0.10` — el pose-cliff; knob de perturbación cherry-picked de `feat/libero-object-perturbation`) | 0/10 (cliff reproducido). **Recall 10/10 episodios**, 126 triggers (~12.6/ep). Specialist = detector de volumen (105); cinemático = confirmador de precisión (21 — mudo en limpios, solo despierta con atasco real). **Primer trigger al 15–28% del episodio** → ~380 steps de margen para recuperar |

**Veredicto de la fase 2**: detector caracterizado — 0 FP en limpios, 100%
recall en fallos, detección temprana. Luz verde para la fase 3.

## Los brazos experimentales y su estado

"Brazo" = condición experimental. "Vivo" = sin shadow: las decisiones se
ejecutan de verdad y cambian lo que hace el robot.

| Brazo | Detector | ¿Interviene? | Estado |
| --- | --- | --- | --- |
| **A — baseline** | apagado | no | ✅ tenemos el suelo: 10/10 limpio, 0/10 en el cliff |
| **B — shadow** | encendido | no (solo apunta) | ✅ hecho — es como se midió FP/recall sin riesgo |
| **C — cinemático vivo** | solo física | sí → flush | ⏳ pendiente |
| **D — delegación completa** | física + VLM | sí → flush + rollback | ⏳ pendiente (el titular) |

A vs C vs D responde la pregunta; **C vs D aísla el valor del VLM**.

Expectativa honesta para la fase 3: en el cliff de 10 cm el fallo es OOD
*espacial* (el objeto está donde la política no sabe operar) — el rollback
puede no bastar ahí. La zona interesante es la intermedia (dx 4–8 cm, política
degradada pero no muerta), donde un reintento desde otra pose sí puede
convertir. Sea cual sea el resultado, ese es el dato.

## Dónde vive todo

- **Rama**: `experiment-vla-recovery-closed-loop` (publicada; sin PR abierto a
  propósito — se abrirá cuando haya historia completa que revisar).
- **Local (Mac)**: worktree `../odyssey-recovery` — ⚠️ el checkout principal
  (`odyssey/`) es para el trabajo de cosmos3-training; el experimento NO se
  toca desde ahí.
- **H100** (`ubuntu@192.222.52.169`): worktree `~/odyssey-recovery`; venv de
  eval `~/odyssey-recovery/env_pilot_libero` (python 3.10 — obligatorio;
  mujoco 2.3.7 pinneado); misión machine-local `~/recovery_shadow_mission.yaml`
  (server GR00T auto-servido en :5599, specialist RoboBrain en :8002,
  `served_model_path` apuntando al subdir `libero_object` del checkpoint).
- **Artefactos de los runs**: `~/.odyssey/runs/<mission>/<task>/recovery/`
  (`recovery_events.jsonl` + `rollouts/*.npz` — el corpus del futuro LVM, ya
  ~270 MB entre limpios y perturbados) + `videos/`.
- **Fase 0**: `~/phase0_*.json`, `~/phase0_sweep_*.jsonl` en la H100.

## Próximos pasos

1. **Fase 3 — brazos vivos C y D**: mismos seeds, `shadow_mode: false` +
   `recovery: true` (C sin `specialist_base_url`, D con él). Correr en el
   cliff (dx 10) y en la zona intermedia (dx 4–8).
2. **PR 2 — el LVM de VLA-Corrector**: entrenar su corrector (~40M) sobre el
   corpus npz que ya se está acumulando, enchufarlo como tier 3.
3. **Abrir PR de revisión** cuando la fase 3 tenga números.
4. (Iteración 2, aparcada con diseño hecho: Cosmos3-WAM × RoboLab.)
