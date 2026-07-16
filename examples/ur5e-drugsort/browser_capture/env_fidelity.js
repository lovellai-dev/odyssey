// ENVIRONMENT-FIDELITY DIAGNOSTIC for the in-browser UR5e drug-sorting eval harness.
//
// QUESTION: every policy evaluated through eval_browser_groot.js scores 0/15 with
// lift_height ~1e-5 and place_dist pinned at ~0.3 m (the vial never moves). Is that a
// real policy failure, or does the EVAL harness's action-application path simply fail
// to drive the arm?
//
// The two harnesses differ ONLY in how a control target reaches mjData.ctrl:
//   browser_harness.js  RUN_EPISODE  -> smooth per-physics-step interp, ONE mj_step per
//                                       ctrl write; records an action every `decim` steps.
//   eval_browser_groot.js RUN_ATTEMPT -> ZERO-ORDER HOLD: setArm(a); 25x mj_step.
//
// DECISIVE EXPERIMENT, over the SAME plans, with byte-identical resets
// (mj_resetDataKeyframe(home) + plan.vial_qpos override + qvel=0 + noslip_iterations=20
// + mj_forward + 180-step SETTLE at home, arm-collision freed once in SETUP):
//
//   A1 "expert-native"        — run the expert FSM exactly as RUN_EPISODE does, and
//                               record the decimated action sequence exactly as data-gen
//                               does. POSITIVE CONTROL: this is how the 151 demos were
//                               made, so it MUST succeed. If A1 fails, nothing else here
//                               is interpretable.
//   A2 "expert-through-eval"  — replay A1's OWN recorded actions open-loop through the
//                               EVAL semantics (ZOH, `decim` physics steps per action).
//
// Identical physics, identical reset, identical action VALUES. The ONLY variable is the
// action-application path -> any A1/A2 gap is a harness artifact, not a policy property.
//
// If A2 fails we sweep the replay hold (1/5/10/25 physics steps per action) and a
// LERP variant that reconstructs the expert's intra-step ramp, to identify the correct
// deploy semantics.
//
// Rendering is deliberately omitted: the Three.js capture path is read-only w.r.t.
// MuJoCo state (syncVisuals pushes mjData -> Three, never the reverse), so dropping it
// leaves the dynamics bit-identical while making the sweep affordable.
//
// SAFE: OWN Chrome, UNIQUE --user-data-dir, PID captured to <OUT>/fidelity_pid.txt and
// killed BY PID. Read-only page load, pure client-side physics — never touches the
// /api/groot bridge, so no GPU and no GR00T server are required and no running service
// is disturbed.
//
// Env: PLANS OUT PORT N CHROME PUPPETEER_CORE DISPLAY
const fs = require('fs');
const path = require('path');
const puppeteer = require(process.env.PUPPETEER_CORE || 'puppeteer-core');

const PLANS = JSON.parse(fs.readFileSync(process.env.PLANS || 'plans_eval.json', 'utf8'));
const OUT = process.env.OUT || path.resolve('env_fidelity_out');
const PORT = process.env.PORT || '8042';
const N = process.env.N ? parseInt(process.env.N) : 5;
const MAX_STEPS = 30000;          // browser_harness.js MAX_STEPS
const DECIM = 25;                 // (1/20 s) / 0.002 s -> 20 Hz control, both harnesses
const CHROME = process.env.CHROME || '/usr/bin/google-chrome-stable';
const PROFILE = path.join(OUT, 'chrome-udd-fidelity-' + Date.now());
const GL_ARGS = ['--use-gl=angle', '--use-angle=gl-egl', '--ignore-gpu-blocklist', '--enable-webgl', '--enable-gpu-rasterization'];
const sleep = (ms) => new Promise(r => setTimeout(r, ms));
fs.mkdirSync(OUT, { recursive: true });

// ── One-time page setup. Mirrors BOTH harnesses' SETUP exactly (same ids, same
//    pm.enabled=false, same arm-collision freeing) + resolves gr_pinch for telemetry.
const SETUP = () => {
  const v = window.viewer, pm = v.physics, mj = pm.mj, m = pm.mjModel;
  pm.enabled = false;   // stop the rAF auto-stepper corrupting the rollout
  const OBJ_BODY = 1, OBJ_JNT = 3, OBJ_ACT = 19, OBJ_SITE = 6, OBJ_GEOM = 5, OBJ_KEY = 24;
  const ANAMES = ['shoulder_pan', 'shoulder_lift', 'elbow', 'wrist_1', 'wrist_2', 'wrist_3'];
  const armAct = ANAMES.map(n => mj.mj_name2id(m, OBJ_ACT, n));
  const gripAct = mj.mj_name2id(m, OBJ_ACT, 'gr_fingers_actuator');
  const armQadr = ANAMES.map(n => { const j = mj.mj_name2id(m, OBJ_JNT, n + '_joint'); return j >= 0 ? m.jnt_qposadr[j] : -1; });
  const gj = mj.mj_name2id(m, OBJ_JNT, 'gr_right_driver_joint');
  const gripQadr = gj >= 0 ? m.jnt_qposadr[gj] : -1;
  const vialBody = mj.mj_name2id(m, OBJ_BODY, 'vial_0');
  let vialQ = -1, vialDof = -1;
  for (let j = 0; j < m.njnt; j++) { if (m.jnt_bodyid[j] === vialBody && m.jnt_type[j] === 0) { vialQ = m.jnt_qposadr[j]; vialDof = m.jnt_dofadr[j]; break; } }
  const pocketSite = mj.mj_name2id(m, OBJ_SITE, 'pocket_0');
  const pinchSite = mj.mj_name2id(m, OBJ_SITE, 'gr_pinch');
  const homeKey = mj.mj_name2id(m, OBJ_KEY, 'home');
  const ARM_BODIES = new Set(['base', 'shoulder_link', 'upper_arm_link', 'forearm_link', 'wrist_1_link', 'wrist_2_link', 'wrist_3_link']);
  let freed = 0;
  for (let g = 0; g < m.ngeom; g++) {
    const bn = mj.mj_id2name(m, OBJ_BODY, m.geom_bodyid[g]) || '';
    const gn = mj.mj_id2name(m, OBJ_GEOM, g) || '';
    if ((ARM_BODIES.has(bn) || bn.startsWith('gr_')) && !gn.includes('pad')) { m.geom_contype[g] = 0; m.geom_conaffinity[g] = 0; freed++; }
  }
  window.__fid = { armAct, gripAct, armQadr, gripQadr, vialBody, vialQ, vialDof, pocketSite, pinchSite, homeKey };
  return { freed, armAct, gripAct, vialBody, vialQ, vialDof, pocketSite, pinchSite, homeKey, gripQadr, timestep: m.opt.timestep, nq: m.nq, nu: m.nu };
};

// ── Shared in-page kernel: accessors, identical reset, telemetry, scoring.
//    Injected as a string into both A1 and A2 so the two paths CANNOT drift.
const KERNEL = `
  const H = window.__fid, v = window.viewer, pm = v.physics, mj = pm.mj, m = pm.mjModel;
  const GRIP_CLOSE = 255, GRIP_RANGE = 0.8, SETTLE = 180;
  // FRESH accessors: WASM heap views detach when the heap grows — re-read pm.mjData.* every call.
  const setArm = q => { const c = pm.mjData.ctrl; for (let i = 0; i < 6; i++) if (H.armAct[i] >= 0) c[H.armAct[i]] = q[i]; };
  const setGrip = g => { if (H.gripAct >= 0) pm.mjData.ctrl[H.gripAct] = g; };
  const readArm = () => H.armQadr.map(a => a >= 0 ? pm.mjData.qpos[a] : 0);
  const gripMeas = () => Math.max(0, Math.min(1, pm.mjData.qpos[H.gripQadr] / GRIP_RANGE));
  const vialPose = () => { const b = H.vialBody, x = pm.mjData.xpos, q = pm.mjData.xquat; return [x[b*3], x[b*3+1], x[b*3+2], q ? q[b*4] : 1, q ? q[b*4+1] : 0, q ? q[b*4+2] : 0, q ? q[b*4+3] : 0]; };
  const pocketPose = () => { const s = H.pocketSite, sp = pm.mjData.site_xpos; return [sp[s*3], sp[s*3+1], sp[s*3+2]]; };
  const pinchPos = () => { const s = H.pinchSite, sp = pm.mjData.site_xpos; return [sp[s*3], sp[s*3+1], sp[s*3+2]]; };
  const dist3 = (a, b) => Math.hypot(a[0]-b[0], a[1]-b[1], a[2]-b[2]);
  const armNorm = (a, b) => { let s = 0; for (let i = 0; i < 6; i++) { const d = a[i]-b[i]; s += d*d; } return Math.sqrt(s); };

  // IDENTICAL reset for every arm of the experiment: home keyframe -> plan vial pose ->
  // zero vial qvel -> noslip 20 -> mj_forward -> 180-step settle holding home.
  const resetTo = (vialQpos, homeQ) => {
    mj.mj_resetDataKeyframe(m, pm.mjData, H.homeKey);
    { const qp = pm.mjData.qpos; for (let i = 0; i < 7; i++) qp[H.vialQ + i] = vialQpos[i]; }
    { const dv = pm.mjData.qvel; for (let i = 0; i < 6; i++) dv[H.vialDof + i] = 0; }
    m.opt.noslip_iterations = 20;
    mj.mj_forward(m, pm.mjData);
    for (let k = 0; k < SETTLE; k++) { setArm(homeQ); setGrip(0); mj.mj_step(m, pm.mjData); }
  };

  // Telemetry accumulator — sampled at the SAME cadence in A1 and A2 so they compare.
  const mkTel = (homeQ) => ({
    homeQ, prevQ: null, arm_travel: 0, max_dev: 0, min_pad: Infinity, gripMax: 0,
    sample(step) {
      const q = readArm();
      if (this.prevQ) this.arm_travel += armNorm(q, this.prevQ);
      this.prevQ = q;
      const dev = armNorm(q, this.homeQ); if (dev > this.max_dev) this.max_dev = dev;
      const d = dist3(pinchPos(), vialPose()); if (d < this.min_pad) this.min_pad = d;
    },
  });

  // The EVAL harness's own scoring criteria, applied verbatim to every arm.
  const score = (z0, zMax, tel, steps, extra) => {
    const vp = vialPose(), pp = pocketPose();
    const placeDist = Math.hypot(vp[0]-pp[0], vp[1]-pp[1]);
    const lifted = (zMax - z0) > 0.02;
    const seated = placeDist < 0.05 && vp[2] > 0.20;
    return Object.assign({
      success: (lifted && seated), lifted, seated,
      lift_height: zMax - z0, place_dist: placeDist,
      gripMax: tel.gripMax,
      arm_travel: tel.arm_travel,
      max_arm_dev_from_home: tel.max_dev,
      min_pad_to_vial_dist: (tel.min_pad === Infinity ? null : tel.min_pad),
      sim_steps: steps,
      vial_final_xyz: [vp[0], vp[1], vp[2]],
    }, extra || {});
  };
`;

// ── A1 "expert-native": browser_harness.js RUN_EPISODE, verbatim FSM + verbatim
//    decimated action recording. Frame capture removed (physics-neutral).
const RUN_A1 = new Function('plan', 'cfg', `return (async () => {
  ${KERNEL}
  const MOVE = 800, CONV = 6e-3, CONV_MAX = 2200;
  const ease = t => t < 0.5 ? 2*t*t : 1 - Math.pow(-2*t + 2, 2)/2;
  const lerp = (a, b, t) => a + (b - a)*t;
  const armErr = (q, meas) => { let mx = 0; for (let i = 0; i < 6; i++) { const e = Math.abs(q[i]-meas[i]); if (e > mx) mx = e; } return mx; };

  resetTo(plan.vial_qpos, plan.home_q);

  const phases = plan.phases;
  let prev = readArm(), phase = 0, s = 0, step = 0;
  const states = [], actions = [];
  const z0 = vialPose()[2];
  let zMax = z0;
  const tel = mkTel(plan.home_q);
  tel.sample(0);

  while (step < cfg.maxSteps) {
    const measured = readArm();
    let armCtrl, gripCtrl, finished = false;
    if (phase >= phases.length) { finished = true; const ph = phases[phases.length-1]; armCtrl = ph.q; gripCtrl = ph.grip; }
    else {
      const ph = phases[phase];
      const same = ph.q.every((x, i) => Math.abs(x - prev[i]) < 1e-3);
      const phaseMove = (ph.move_steps != null) ? ph.move_steps : MOVE;
      const moveDur = same ? 0 : phaseMove;
      armCtrl = (s < moveDur) ? ph.q.map((x, i) => lerp(prev[i], x, ease(s/moveDur))) : ph.q;
      gripCtrl = ph.grip;
      s++;
      if (s >= moveDur) {
        const held = s - moveDur;
        const done = (ph.hold === 'conv') ? (armErr(ph.q, measured) < CONV || held > CONV_MAX) : (held >= ph.hold);
        if (done) { prev = ph.q.slice(); phase++; s = 0; }
      }
    }
    // EXPERT SEMANTICS: write the smoothly-interpolated target, ONE mj_step per write.
    setArm(armCtrl); setGrip(gripCtrl);
    mj.mj_step(m, pm.mjData);
    step++;
    const vz = vialPose()[2]; if (vz > zMax) zMax = vz;
    tel.gripMax = Math.max(tel.gripMax, gripCtrl / GRIP_CLOSE);

    if (step % cfg.decim === 0) {
      states.push([measured[0], measured[1], measured[2], measured[3], measured[4], measured[5], gripMeas()]);
      actions.push([armCtrl[0], armCtrl[1], armCtrl[2], armCtrl[3], armCtrl[4], armCtrl[5], gripCtrl / GRIP_CLOSE]);
      tel.sample(step);
    }
    if (finished) break;
  }
  return Object.assign(score(z0, zMax, tel, step), { actions, num_actions: actions.length, n_states: states.length });
})();`);

// ── A2 "expert-through-eval-path": eval_browser_groot.js RUN_ATTEMPT semantics applied
//    to A1's OWN recorded actions, open-loop. cfg.hold = physics steps per action
//    (25 == the eval's decim). cfg.interp = reconstruct the intra-step ramp by LERPing
//    from the previous action to the current one across the hold (1 mj_step per write,
//    i.e. the expert's own write cadence at the policy's 20 Hz action rate).
const RUN_A2 = new Function('plan', 'actions', 'cfg', `return (async () => {
  ${KERNEL}
  resetTo(plan.vial_qpos, plan.home_q);

  const z0 = vialPose()[2];
  let zMax = z0, step = 0, ticks = 0;
  const tel = mkTel(plan.home_q);
  tel.sample(0);
  let prevA = readArm().concat([0]);   // ramp origin for the interp variant

  for (let k = 0; k < actions.length; k++) {
    if (cfg.maxTicks && ticks >= cfg.maxTicks) break;
    const a = actions[k];
    const g = Math.max(0, Math.min(1, a[6]));
    tel.gripMax = Math.max(tel.gripMax, g);
    if (cfg.interp) {
      // LERP the target across the hold: same 20 Hz action stream, expert write cadence.
      for (let s = 1; s <= cfg.hold; s++) {
        const t = s / cfg.hold;
        const q = [];
        for (let i = 0; i < 6; i++) q.push(prevA[i] + (a[i] - prevA[i]) * t);
        setArm(q); setGrip(g * GRIP_CLOSE);
        mj.mj_step(m, pm.mjData); step++;
        const vz = vialPose()[2]; if (vz > zMax) zMax = vz;
      }
    } else {
      // EVAL SEMANTICS (verbatim): one setArm/setGrip, then a ZERO-ORDER HOLD of
      // cfg.hold physics steps with the target frozen.
      setArm(a.slice(0, 6)); setGrip(g * GRIP_CLOSE);
      for (let s = 0; s < cfg.hold; s++) { mj.mj_step(m, pm.mjData); step++; }
      const vz = vialPose()[2]; if (vz > zMax) zMax = vz;
    }
    prevA = a;
    ticks++;
    tel.sample(step);
  }
  return score(z0, zMax, tel, step, { ticks, hold: cfg.hold, interp: !!cfg.interp });
})();`);

(async () => {
  const browser = await puppeteer.launch({
    executablePath: CHROME, headless: true, userDataDir: PROFILE, protocolTimeout: 1800000,
    env: { ...process.env, DISPLAY: process.env.DISPLAY || ':1' },
    args: ['--no-sandbox', '--disable-dev-shm-usage', '--disable-gpu-sandbox', '--window-size=1400,950', ...GL_ARGS],
  });
  const pid = browser.process().pid;
  fs.writeFileSync(path.join(OUT, 'fidelity_pid.txt'), String(pid));
  console.log('FIDELITY_CHROME_PID=' + pid + ' port=' + PORT + ' N=' + N);
  const page = await browser.newPage();
  await page.setViewport({ width: 1400, height: 950 });
  page.on('console', m => { const t = m.text(); if (/error/i.test(t)) console.log('  [page]', t.slice(0, 140)); });
  page.on('pageerror', e => console.log('  [pageerror]', String(e).slice(0, 200)));

  const url = `http://localhost:${PORT}/robot-playground.html?demo=drugsorting`;
  console.log('GOTO', url);
  await page.goto(url, { waitUntil: 'load', timeout: 120000 });
  await page.waitForFunction(() => window.viewer && window.viewer.physics && window.viewer.physics.mj && window.viewer.physics.mjModel, { timeout: 120000 });
  await page.waitForFunction(() => {
    const pm = window.viewer.physics, mj = pm.mj; if (!pm.mjModel) return false;
    return mj.mj_name2id(pm.mjModel, 1, 'vial_0') >= 0;
  }, { timeout: 120000 });
  await sleep(1500);
  const setup = await page.evaluate(SETUP);
  console.log('SETUP ' + JSON.stringify(setup));
  if (setup.pinchSite < 0) console.log('WARN: gr_pinch site not found — min_pad_to_vial_dist will be null');

  const plans = PLANS.plans.slice(0, N);
  const perPlan = [];
  for (const plan of plans) {
    const t0 = Date.now();
    const a1 = await page.evaluate(RUN_A1, plan, { decim: DECIM, maxSteps: MAX_STEPS });
    const actions = a1.actions; delete a1.actions;
    const a2 = await page.evaluate(RUN_A2, plan, actions, { hold: DECIM, interp: false, maxTicks: 0 });
    const secs = (Date.now() - t0) / 1000;
    perPlan.push({ episode: plan.episode, num_actions: a1.num_actions, A1: a1, A2: a2, seconds: +secs.toFixed(1), _actions: actions });
    const f = (r) => `${r.success ? 'SUCCESS' : 'fail'} lift=${(r.lift_height*100).toFixed(2)}cm place=${(r.place_dist*100).toFixed(1)}cm travel=${r.arm_travel.toFixed(2)} dev=${r.max_arm_dev_from_home.toFixed(2)} pad=${r.min_pad_to_vial_dist == null ? 'n/a' : (r.min_pad_to_vial_dist*100).toFixed(1)+'cm'} grip=${r.gripMax.toFixed(2)}`;
    console.log(`ep${String(plan.episode).padStart(3,'0')} nact=${a1.num_actions}\n   A1 ${f(a1)}\n   A2 ${f(a2)}   (${secs.toFixed(1)}s)`);
  }

  const nA1 = perPlan.filter(p => p.A1.success).length;
  const nA2 = perPlan.filter(p => p.A2.success).length;
  console.log(`A1=${nA1}/${perPlan.length}  A2=${nA2}/${perPlan.length}`);

  // ── Sweep: only meaningful when the eval path (A2) loses ground the expert (A1) holds.
  let sweep = null;
  if (nA2 < nA1) {
    console.log('A2 < A1 -> sweeping replay hold + interp to find the semantics that restore expert success');
    sweep = [];
    const variants = [
      { name: 'hold1', hold: 1, interp: false },
      { name: 'hold5', hold: 5, interp: false },
      { name: 'hold10', hold: 10, interp: false },
      { name: 'hold25_zoh', hold: 25, interp: false },
      { name: 'hold25_lerp', hold: 25, interp: true },
      { name: 'hold5_lerp', hold: 5, interp: true },
    ];
    for (const vr of variants) {
      const rows = [];
      for (const p of perPlan) {
        const plan = plans.find(x => x.episode === p.episode);
        const r = await page.evaluate(RUN_A2, plan, p._actions, { hold: vr.hold, interp: vr.interp, maxTicks: 0 });
        rows.push({ episode: p.episode, ...r });
      }
      const ns = rows.filter(r => r.success).length;
      sweep.push({
        variant: vr.name, hold: vr.hold, interp: vr.interp,
        n_success: ns, success: `${ns}/${rows.length}`,
        mean_arm_travel: +(rows.reduce((a, r) => a + r.arm_travel, 0) / rows.length).toFixed(3),
        mean_lift_height: +(rows.reduce((a, r) => a + r.lift_height, 0) / rows.length).toFixed(5),
        mean_place_dist: +(rows.reduce((a, r) => a + r.place_dist, 0) / rows.length).toFixed(4),
        results: rows,
      });
      console.log(`  SWEEP ${vr.name.padEnd(12)} ${ns}/${rows.length}  travel=${(rows.reduce((a,r)=>a+r.arm_travel,0)/rows.length).toFixed(2)} lift=${(rows.reduce((a,r)=>a+r.lift_height,0)/rows.length*100).toFixed(2)}cm`);
    }
  }

  const mean = (f) => +(perPlan.reduce((a, p) => a + f(p), 0) / perPlan.length).toFixed(5);
  let verdict;
  if (nA1 < perPlan.length) verdict = 'POSITIVE_CONTROL_BROKEN: A1 (expert-native) did not reproduce the demo success rate — nothing downstream is interpretable';
  else if (nA2 === nA1) verdict = 'EVAL_ACTION_PATH_FAITHFUL: the expert survives the eval action path unchanged, so the in-browser 0/15 is NOT an action-application artifact';
  else verdict = 'EVAL_HARNESS_IS_THE_BUG: the expert succeeds natively but fails through the eval action path — the in-browser 0/15 is a harness artifact';

  for (const p of perPlan) delete p._actions;
  const result = {
    experiment: 'env_fidelity_A1_expert_native_vs_A2_expert_through_eval_path',
    generated_utc: new Date().toISOString(),
    port: +PORT, decim: DECIM, n_plans: perPlan.length,
    plans_source: process.env.PLANS || 'plans_eval.json',
    episodes: perPlan.map(p => p.episode),
    eval_harness_max_ticks_default: 900,
    expert_num_actions: perPlan.map(p => p.num_actions),
    A1_expert_native: {
      n_success: nA1, success: `${nA1}/${perPlan.length}`,
      mean_arm_travel: mean(p => p.A1.arm_travel),
      mean_max_arm_dev_from_home: mean(p => p.A1.max_arm_dev_from_home),
      mean_min_pad_to_vial_dist: mean(p => p.A1.min_pad_to_vial_dist || 0),
      mean_lift_height: mean(p => p.A1.lift_height),
      mean_place_dist: mean(p => p.A1.place_dist),
    },
    A2_expert_through_eval_path: {
      n_success: nA2, success: `${nA2}/${perPlan.length}`,
      mean_arm_travel: mean(p => p.A2.arm_travel),
      mean_max_arm_dev_from_home: mean(p => p.A2.max_arm_dev_from_home),
      mean_min_pad_to_vial_dist: mean(p => p.A2.min_pad_to_vial_dist || 0),
      mean_lift_height: mean(p => p.A2.lift_height),
      mean_place_dist: mean(p => p.A2.place_dist),
    },
    verdict,
    sweep,
    per_plan: perPlan,
  };
  fs.writeFileSync(path.join(OUT, 'env_fidelity.json'), JSON.stringify(result, null, 2));
  fs.writeFileSync('/home/daniel/env_fidelity_result.json', JSON.stringify(result, null, 2));
  console.log('VERDICT ' + verdict);

  await browser.close();
  try { fs.rmSync(PROFILE, { recursive: true, force: true }); } catch (e) {}
  console.log('FIDELITY_DONE');
})().catch(e => { console.error('FIDELITY_ERROR', e && e.stack || e); process.exit(1); });
