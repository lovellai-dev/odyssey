// In-browser closed-loop GR00T eval for the render-gap close.
//
// Drives the REAL deployment path: loads the Lovell AI Robot Playground in headless
// Chrome and, per attempt, POSTs the exterior+wrist Three.js frames + proprio to the
// SAME same-origin proxy the deployed ?agents=groot pilot uses
// (/api/groot/get_action -> agent-service -> GR00T HTTP/ZMQ bridge -> served
// checkpoint), applies the returned absolute joint-target chunk, and scores
// grasp+place success from the MuJoCo-WASM GT vial/pocket pose. Each attempt starts
// from a randomized vial pose (from the eval plans.json) so baseline and new
// checkpoint see the SAME initial poses -> a fair success RATE.
//
// This is the measurement that matters: the NEW in-browser success rate after
// training on deployment-matched Three.js frames, vs the ~0 render-gap baseline.
//
// SAFE: own Chrome, UNIQUE --user-data-dir, PID captured + cleaned by PID. Read-only
// page load; the only network write is the deployment proxy (exactly what deploy does).
//
// Env: PLANS OUT PORT(8021) AGENTS(groot|groot-observer) N N_ACTION_STEPS(8)
//      MAX_TICKS(1000) CHROME PUPPETEER_CORE DISPLAY
const fs = require('fs');
const path = require('path');
const puppeteer = require(process.env.PUPPETEER_CORE || 'puppeteer-core');

const PLANS = JSON.parse(fs.readFileSync(process.env.PLANS || 'plans_eval.json', 'utf8'));
const OUT = process.env.OUT || path.resolve('eval_out');
const PORT = process.env.PORT || '8021';
const AGENTS = process.env.AGENTS || 'groot';                 // groot | groot-observer
const N = process.env.N ? parseInt(process.env.N) : PLANS.plans.length;
const N_ACTION_STEPS = process.env.N_ACTION_STEPS ? parseInt(process.env.N_ACTION_STEPS) : 8;
const MAX_TICKS = process.env.MAX_TICKS ? parseInt(process.env.MAX_TICKS) : 900;
const CHROME = process.env.CHROME || '/usr/bin/google-chrome-stable';
const PROFILE = path.join(OUT, 'chrome-udd-eval-' + Date.now());
const GL_ARGS = ['--use-gl=angle', '--use-angle=gl-egl', '--ignore-gpu-blocklist', '--enable-webgl', '--enable-gpu-rasterization'];
const sleep = (ms) => new Promise(r => setTimeout(r, ms));
fs.mkdirSync(OUT, { recursive: true });

// One-time page setup (mirrors browser_harness SETUP): ids, free arm collision,
// deployment sensor cameras + capture, disable the rAF auto-stepper.
const SETUP = () => {
  const v = window.viewer, pm = v.physics, mj = pm.mj, m = pm.mjModel;
  pm.enabled = false;
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
  const pinchSite = mj.mj_name2id(m, OBJ_SITE, 'gr_pinch');   // near-miss telemetry (as probe_browser_groot.js)
  const homeKey = mj.mj_name2id(m, OBJ_KEY, 'home');
  const ARM_BODIES = new Set(['base', 'shoulder_link', 'upper_arm_link', 'forearm_link', 'wrist_1_link', 'wrist_2_link', 'wrist_3_link']);
  for (let g = 0; g < m.ngeom; g++) {
    const bn = mj.mj_id2name(m, OBJ_BODY, m.geom_bodyid[g]) || '';
    const gn = mj.mj_id2name(m, OBJ_GEOM, g) || '';
    if ((ARM_BODIES.has(bn) || bn.startsWith('gr_')) && !gn.includes('pad')) { m.geom_contype[g] = 0; m.geom_conaffinity[g] = 0; }
  }
  const G = window.MultiAgentGroot;
  const camExt = G.makeMjCamera(pm, mj, 'room');
  const camWrist = G.makeMjCamera(pm, mj, 'wrist');
  const cap = G.makeCapture(v.renderer);
  window.__ev = { armAct, gripAct, armQadr, gripQadr, vialBody, vialQ, vialDof, pocketSite, pinchSite, homeKey, camExt, camWrist, cap };
  return { armAct, gripAct, vialBody, vialQ, pocketSite, pinchSite, homeKey };
};

// One in-browser GR00T attempt: randomized start -> closed-loop query/act -> score.
const RUN_ATTEMPT = async (vialQpos, homeQ, cfg) => {
  const H = window.__ev, v = window.viewer, pm = v.physics, mj = pm.mj, m = pm.mjModel, G = window.MultiAgentGroot;
  const GRIP_CLOSE = 255, GRIP_RANGE = 0.8, SETTLE = 180;
  const setArm = q => { const c = pm.mjData.ctrl; for (let i = 0; i < 6; i++) if (H.armAct[i] >= 0) c[H.armAct[i]] = q[i]; };
  const setGrip = g => { if (H.gripAct >= 0) pm.mjData.ctrl[H.gripAct] = g; };   // g in 0..255
  const readArm = () => H.armQadr.map(a => a >= 0 ? pm.mjData.qpos[a] : 0);
  const vialPose = () => { const b = H.vialBody, x = pm.mjData.xpos, q = pm.mjData.xquat; return [x[b * 3], x[b * 3 + 1], x[b * 3 + 2], q ? q[b * 4] : 1, q ? q[b * 4 + 1] : 0, q ? q[b * 4 + 2] : 0, q ? q[b * 4 + 3] : 0]; };
  const pocketPose = () => { const s = H.pocketSite, sp = pm.mjData.site_xpos; return [sp[s * 3], sp[s * 3 + 1], sp[s * 3 + 2]]; };
  // Telemetry-only (same as probe_browser_groot.js): closest gr_pinch approach to the vial.
  const pinchPos = () => { const s = H.pinchSite, sp = pm.mjData.site_xpos; return s >= 0 ? [sp[s * 3], sp[s * 3 + 1], sp[s * 3 + 2]] : null; };
  const dist3 = (a, b) => Math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2]);
  const decim = Math.max(1, Math.round((1 / 20) / (m.opt.timestep || 0.002)));

  // Randomized reset (same override the data-gen harness uses).
  mj.mj_resetDataKeyframe(m, pm.mjData, H.homeKey);
  { const qp = pm.mjData.qpos; for (let i = 0; i < 7; i++) qp[H.vialQ + i] = vialQpos[i]; }
  { const dv = pm.mjData.qvel; for (let i = 0; i < 6; i++) dv[H.vialDof + i] = 0; }
  m.opt.noslip_iterations = 20;
  mj.mj_forward(m, pm.mjData);
  for (let k = 0; k < SETTLE; k++) { setArm(homeQ); setGrip(0); mj.mj_step(m, pm.mjData); }

  const z0 = vialPose()[2];
  let zMax = z0, ticks = 0, queries = 0, gripCmd = 0, gripMax = 0, err = null, minPad = Infinity;
  const url = cfg.agents === 'groot-observer' ? '/api/groot/get_action_observer' : '/api/groot/get_action';
  const sid = cfg.agents === 'groot-observer' ? ('obs-' + Date.now() + '-' + Math.floor(Math.random() * 1e6)) : null;

  while (ticks < cfg.maxTicks) {
    const measured = readArm();
    const state = [measured[0], measured[1], measured[2], measured[3], measured[4], measured[5], Math.max(0, Math.min(1, gripCmd))];
    if (pm.syncVisuals) pm.syncVisuals();
    const img = H.cap(v.scene, H.camExt.cam);
    G.syncMjCamera(H.camWrist.cam, pm, mj, H.camWrist.id);
    const imgWrist = H.cap(v.scene, H.camWrist.cam);
    const body = { image_b64: img, state, instruction: 'pick up the vial and place it in the rack', image_b64_wrist: imgWrist };
    if (sid) body.sid = sid;
    let res;
    try {
      const resp = await fetch(url, { method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify(body) });
      if (!resp.ok) { err = 'proxy ' + resp.status + ': ' + (await resp.text()).slice(0, 120); break; }
      res = await resp.json();
    } catch (e) { err = String(e); break; }
    queries++;
    const chunkQ = (res.chunk_q && res.chunk_q.length) ? res.chunk_q : [res.q];
    const chunkG = (res.chunk_grip && res.chunk_grip.length) ? res.chunk_grip : [res.grip];
    const n = Math.min(cfg.nActionSteps, chunkQ.length);
    for (let i = 0; i < n; i++) {
      gripCmd = Math.max(0, Math.min(1, chunkG[i]));
      gripMax = Math.max(gripMax, gripCmd);
      setArm(chunkQ[i]); setGrip(gripCmd * GRIP_CLOSE);
      // Hold the action target for `decim` (=25) physics steps = one 20 Hz control
      // step (mujoco-wasm mj_step takes no nstep arg, so loop explicitly — matches
      // browser_harness + the deploy bridge.step(decim)).
      for (let s = 0; s < decim; s++) mj.mj_step(m, pm.mjData);
      const vz = vialPose()[2]; if (vz > zMax) zMax = vz;
      const pinch = pinchPos(); if (pinch) { const d = dist3(pinch, vialPose()); if (d < minPad) minPad = d; }
      ticks++;
      if (ticks >= cfg.maxTicks) break;
    }
  }
  const vp = vialPose(), pp = pocketPose();
  const placeDist = Math.hypot(vp[0] - pp[0], vp[1] - pp[1]);
  const lifted = (zMax - z0) > 0.02;
  const seated = placeDist < 0.05 && vp[2] > 0.20;
  return { success: (lifted && seated), lifted, seated, lift_height: zMax - z0, place_dist: placeDist, gripMax, queries, ticks, min_pad_to_vial: (minPad === Infinity ? null : minPad), error: err };
};

(async () => {
  const browser = await puppeteer.launch({
    executablePath: CHROME, headless: true, userDataDir: PROFILE, protocolTimeout: 900000,
    env: { ...process.env, DISPLAY: process.env.DISPLAY || ':1' },
    args: ['--no-sandbox', '--disable-dev-shm-usage', '--disable-gpu-sandbox', '--window-size=1400,950', ...GL_ARGS],
  });
  const pid = browser.process().pid;
  fs.writeFileSync(path.join(OUT, 'eval_pid.txt'), String(pid));
  console.log('EVAL_CHROME_PID=' + pid + ' agents=' + AGENTS + ' port=' + PORT + ' N=' + N);
  const page = await browser.newPage();
  await page.setViewport({ width: 1400, height: 950 });
  page.on('console', m => { const t = m.text(); if (/\[groot\]|error|observer/i.test(t)) console.log('  [page]', t.slice(0, 160)); });
  // Load WITHOUT &agents= so the page's own runGrootAgents() never auto-fires; our
  // RUN_ATTEMPT drives the /api/groot proxy directly (AGENTS only selects the endpoint).
  const url = `http://localhost:${PORT}/robot-playground.html?demo=drugsorting`;
  console.log('GOTO', url);
  await page.goto(url, { waitUntil: 'load', timeout: 120000 });
  await page.waitForFunction(() => window.viewer && window.viewer.physics && window.viewer.physics.mj && window.viewer.physics.mjModel, { timeout: 120000 });
  await page.waitForFunction(() => {
    const pm = window.viewer.physics, mj = pm.mj; if (!pm.mjModel) return false;
    return mj.mj_name2id(pm.mjModel, 1, 'vial_0') >= 0 && !!window.MultiAgentGroot;
  }, { timeout: 120000 });
  await sleep(1200);
  await page.evaluate(SETUP);

  // Confirm the deployment proxy is reachable/healthy before scoring.
  const health = await page.evaluate(async () => { try { const r = await fetch('/api/groot/health'); return await r.json(); } catch (e) { return { error: String(e) }; } });
  console.log('HEALTH ' + JSON.stringify(health));

  const results = [];
  const t0all = Date.now();
  const plans = PLANS.plans.slice(0, N);
  for (const plan of plans) {
    const t0 = Date.now();
    const r = await page.evaluate(RUN_ATTEMPT, plan.vial_qpos, plan.home_q, { maxTicks: MAX_TICKS, nActionSteps: N_ACTION_STEPS, agents: AGENTS });
    const secs = (Date.now() - t0) / 1000;
    results.push({ episode: plan.episode, ...r, seconds: +secs.toFixed(1) });
    console.log(`ATT ep${String(plan.episode).padStart(3, '0')}: ${r.success ? 'SUCCESS' : (r.error ? 'ERROR:' + r.error : 'fail')} lifted=${r.lifted} seated=${r.seated} lift=${(r.lift_height * 100).toFixed(1)}cm place=${(r.place_dist * 100).toFixed(1)}cm pad=${r.min_pad_to_vial == null ? 'n/a' : (r.min_pad_to_vial * 100).toFixed(1) + 'cm'} grip=${r.gripMax.toFixed(2)} q=${r.queries} ${secs.toFixed(1)}s`);
  }
  const nSucc = results.filter(r => r.success).length;
  const nLift = results.filter(r => r.lifted).length;
  const nSeat = results.filter(r => r.seated).length;
  const summary = {
    agents: AGENTS, port: PORT, n_action_steps: N_ACTION_STEPS, max_ticks: MAX_TICKS,
    n_attempts: results.length, n_success: nSucc, n_lifted: nLift, n_seated: nSeat,
    success_rate: +(nSucc / results.length).toFixed(3), success: `${nSucc}/${results.length}`,
    total_seconds: +((Date.now() - t0all) / 1000).toFixed(1), results,
  };
  fs.writeFileSync(path.join(OUT, `eval_${AGENTS}.json`), JSON.stringify(summary, null, 2));
  console.log('EVAL_SUMMARY ' + JSON.stringify({ agents: AGENTS, success: summary.success, rate: summary.success_rate, lifted: nLift, seated: nSeat }));
  await browser.close();
  try { fs.rmSync(PROFILE, { recursive: true, force: true }); } catch (e) {}
  console.log('EVAL_DONE');
})().catch(e => { console.error('EVAL_ERROR', e && e.stack || e); process.exit(1); });
