// Phase-1 inference-input integrity probe (PLAN_MULTIAGENT.md) — an INSTRUMENTED
// in-browser GR00T eval. Drives the exact deployment path eval_browser_groot.js
// drives (?agents=groot: browser -> /api/groot/get_action -> conditioner -> tunnel
// -> bridge -> ZMQ GR00T), but records per query:
//   - the state actually sent + FNV-1a hashes of both frame dataURLs
//   - the full returned action chunk + injected grasp_target + conditioner mode
//   - post-chunk telemetry: achieved qpos, measured vs commanded grip, per-joint
//     |ctrl - qpos| tracking error, vial GT pose, pinch-to-vial distance
// and dumps two raw frame+state samples (ep0 q0 / ep0 q30) for the offline
// controlled-payload battery (probe_inference_path.py).
//
// SAFE: own Chrome, UNIQUE --user-data-dir, PID captured + killed by PID.
// NEVER pkill chrome.
//
// Env: PLANS OUT PORT N(3) N_ACTION_STEPS(8) MAX_TICKS(900) CHROME PUPPETEER_CORE DISPLAY
const fs = require('fs');
const path = require('path');
const puppeteer = require(process.env.PUPPETEER_CORE || 'puppeteer-core');

const PLANS = JSON.parse(fs.readFileSync(process.env.PLANS || 'plans_eval.json', 'utf8'));
const OUT = process.env.OUT || path.resolve('probe_out');
const PORT = process.env.PORT || '8032';
const N = process.env.N ? parseInt(process.env.N) : 3;
const N_ACTION_STEPS = process.env.N_ACTION_STEPS ? parseInt(process.env.N_ACTION_STEPS) : 8;
const MAX_TICKS = process.env.MAX_TICKS ? parseInt(process.env.MAX_TICKS) : 900;
const CHROME = process.env.CHROME || '/usr/bin/google-chrome-stable';
const PROFILE = path.join(OUT, 'chrome-udd-probe-' + Date.now());
const GL_ARGS = ['--use-gl=angle', '--use-angle=gl-egl', '--ignore-gpu-blocklist', '--enable-webgl', '--enable-gpu-rasterization'];
const sleep = (ms) => new Promise(r => setTimeout(r, ms));
fs.mkdirSync(OUT, { recursive: true });

// Same one-time setup as eval_browser_groot.js + gr_pinch site for near-miss telemetry.
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
  const pinchSite = mj.mj_name2id(m, OBJ_SITE, 'gr_pinch');
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
  window.__pr = { armAct, gripAct, armQadr, gripQadr, vialBody, vialQ, vialDof, pocketSite, pinchSite, homeKey, camExt, camWrist, cap };
  return { armAct, gripAct, pinchSite, homeKey };
};

// One instrumented attempt. Identical control flow to eval_browser_groot.js
// RUN_ATTEMPT; the additions are telemetry-only (log array + frame dumps).
const RUN_ATTEMPT = async (vialQpos, homeQ, cfg) => {
  const H = window.__pr, v = window.viewer, pm = v.physics, mj = pm.mj, m = pm.mjModel, G = window.MultiAgentGroot;
  const GRIP_CLOSE = 255, GRIP_RANGE = 0.8, SETTLE = 180;
  const setArm = q => { const c = pm.mjData.ctrl; for (let i = 0; i < 6; i++) if (H.armAct[i] >= 0) c[H.armAct[i]] = q[i]; };
  const setGrip = g => { if (H.gripAct >= 0) pm.mjData.ctrl[H.gripAct] = g; };
  const readArm = () => H.armQadr.map(a => a >= 0 ? pm.mjData.qpos[a] : 0);
  const gripMeas = () => Math.max(0, Math.min(1, pm.mjData.qpos[H.gripQadr] / GRIP_RANGE));
  const vialPose = () => { const b = H.vialBody, x = pm.mjData.xpos; return [x[b * 3], x[b * 3 + 1], x[b * 3 + 2]]; };
  const pinchPos = () => { const s = H.pinchSite, sp = pm.mjData.site_xpos; return s >= 0 ? [sp[s * 3], sp[s * 3 + 1], sp[s * 3 + 2]] : null; };
  const fnv = s => { let h = 0x811c9dc5; for (let i = 0; i < s.length; i++) { h ^= s.charCodeAt(i); h = (h * 0x01000193) >>> 0; } return h.toString(16).padStart(8, '0'); };
  const dist3 = (a, b) => Math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2]);
  const decim = Math.max(1, Math.round((1 / 20) / (m.opt.timestep || 0.002)));

  mj.mj_resetDataKeyframe(m, pm.mjData, H.homeKey);
  { const qp = pm.mjData.qpos; for (let i = 0; i < 7; i++) qp[H.vialQ + i] = vialQpos[i]; }
  { const dv = pm.mjData.qvel; for (let i = 0; i < 6; i++) dv[H.vialDof + i] = 0; }
  m.opt.noslip_iterations = 20;
  mj.mj_forward(m, pm.mjData);
  for (let k = 0; k < SETTLE; k++) { setArm(homeQ); setGrip(0); mj.mj_step(m, pm.mjData); }

  const z0 = vialPose()[2];
  let zMax = z0, ticks = 0, queries = 0, gripCmd = 0, gripMax = 0, err = null, minPad = Infinity;
  const log = [], chunks = [], dumps = [];

  while (ticks < cfg.maxTicks) {
    const measured = readArm();
    const state = [measured[0], measured[1], measured[2], measured[3], measured[4], measured[5], Math.max(0, Math.min(1, gripCmd))];
    if (pm.syncVisuals) pm.syncVisuals();
    const img = H.cap(v.scene, H.camExt.cam);
    G.syncMjCamera(H.camWrist.cam, pm, mj, H.camWrist.id);
    const imgWrist = H.cap(v.scene, H.camWrist.cam);
    if (cfg.dumpFrames && (queries === 0 || queries === 30)) {
      dumps.push({ query: queries, state: state.slice(), ext: img, wrist: imgWrist });
    }
    const body = { image_b64: img, state, instruction: 'pick up the vial and place it in the rack', image_b64_wrist: imgWrist };
    let res;
    try {
      const resp = await fetch('/api/groot/get_action', { method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify(body) });
      if (!resp.ok) { err = 'proxy ' + resp.status + ': ' + (await resp.text()).slice(0, 120); break; }
      res = await resp.json();
    } catch (e) { err = String(e); break; }
    queries++;
    const chunkQ = (res.chunk_q && res.chunk_q.length) ? res.chunk_q : [res.q];
    const chunkG = (res.chunk_grip && res.chunk_grip.length) ? res.chunk_grip : [res.grip];
    if (queries <= 40 || queries % 10 === 0) chunks.push(chunkQ.map(r => r.slice()));
    const n = Math.min(cfg.nActionSteps, chunkQ.length);
    let lastQ = null;
    for (let i = 0; i < n; i++) {
      // Clamp the grip index (chunk_grip may be shorter than chunk_q) and fail
      // loudly on non-finite values instead of writing NaN into mjData.ctrl.
      gripCmd = Math.max(0, Math.min(1, chunkG[Math.min(i, chunkG.length - 1)]));
      if (!Number.isFinite(gripCmd) || chunkQ[i].some(x => !Number.isFinite(x))) { err = 'non-finite action at query ' + queries; break; }
      gripMax = Math.max(gripMax, gripCmd);
      lastQ = chunkQ[i];
      setArm(chunkQ[i]); setGrip(gripCmd * GRIP_CLOSE);
      for (let s = 0; s < decim; s++) mj.mj_step(m, pm.mjData);
      const vz = vialPose()[2]; if (vz > zMax) zMax = vz;
      const pp = pinchPos(); if (pp) { const d = dist3(pp, vialPose()); if (d < minPad) minPad = d; }
      ticks++;
      if (ticks >= cfg.maxTicks) break;
    }
    if (err) break;
    const achieved = readArm();
    log.push({
      tick: ticks, query: queries,
      state_sent: state.map(x => +x.toFixed(6)),
      ext_sha: fnv(img), wrist_sha: fnv(imgWrist), ext_len: img.length, wrist_len: imgWrist.length,
      q0: chunkQ[0].map(x => +x.toFixed(5)), grip0: +(+chunkG[0]).toFixed(4),
      chunk_len: chunkQ.length,
      grasp_target: res.grasp_target ? res.grasp_target.map(x => +x.toFixed(5)) : null,
      cond_mode: res.conditioner ? res.conditioner.mode : null,
      post: {
        qpos: achieved.map(x => +x.toFixed(5)),
        grip_meas: +gripMeas().toFixed(4), grip_cmd: +gripCmd.toFixed(4),
        track_err: lastQ ? lastQ.map((q, j) => +Math.abs(q - achieved[j]).toFixed(5)) : null,
        vial_xyz: vialPose().map(x => +x.toFixed(5)),
      },
    });
  }
  const vp = vialPose();
  const lifted = (zMax - z0) > 0.02;
  return {
    lifted, lift_height: zMax - z0, gripMax, queries, ticks, error: err,
    min_pad_to_vial: (minPad === Infinity ? null : minPad),
    vial_final: vp, log, chunks, dumps,
  };
};

(async () => {
  const browser = await puppeteer.launch({
    executablePath: CHROME, headless: true, userDataDir: PROFILE, protocolTimeout: 900000,
    env: { ...process.env, DISPLAY: process.env.DISPLAY || ':1' },
    args: ['--no-sandbox', '--disable-dev-shm-usage', '--disable-gpu-sandbox', '--window-size=1400,950', ...GL_ARGS],
  });
  const pid = browser.process().pid;
  fs.writeFileSync(path.join(OUT, 'probe_pid.txt'), String(pid));
  console.log('PROBE_CHROME_PID=' + pid + ' port=' + PORT + ' N=' + N);
  const page = await browser.newPage();
  await page.setViewport({ width: 1400, height: 950 });
  page.on('console', mm => { const t = mm.text(); if (/error/i.test(t)) console.log('  [page]', t.slice(0, 160)); });
  const url = `http://localhost:${PORT}/robot-playground.html?demo=drugsorting`;
  console.log('GOTO', url);
  await page.goto(url, { waitUntil: 'load', timeout: 120000 });
  await page.waitForFunction(() => window.viewer && window.viewer.physics && window.viewer.physics.mj && window.viewer.physics.mjModel, { timeout: 120000 });
  await page.waitForFunction(() => {
    const pm = window.viewer.physics, mj = pm.mj; if (!pm.mjModel) return false;
    return mj.mj_name2id(pm.mjModel, 1, 'vial_0') >= 0 && !!window.MultiAgentGroot;
  }, { timeout: 120000 });
  await sleep(1200);
  const setup = await page.evaluate(SETUP);
  if (setup.pinchSite < 0) console.log('WARN: gr_pinch site not found — min_pad_to_vial will be null');

  const health = await page.evaluate(async () => { try { const r = await fetch('/api/groot/health'); return await r.json(); } catch (e) { return { error: String(e) }; } });
  console.log('HEALTH ' + JSON.stringify(health));
  if (!(health && health.bridge_health && health.bridge_health.ok)) {
    console.error('PROBE_ERROR bridge unhealthy: ' + JSON.stringify(health)); process.exit(1);
  }

  const episodes = [];
  const plans = PLANS.plans.slice(0, N);
  for (let e = 0; e < plans.length; e++) {
    const plan = plans[e];
    const t0 = Date.now();
    const r = await page.evaluate(RUN_ATTEMPT, plan.vial_qpos, plan.home_q,
      { maxTicks: MAX_TICKS, nActionSteps: N_ACTION_STEPS, dumpFrames: e === 0 });
    const secs = (Date.now() - t0) / 1000;
    // Persist per-episode telemetry + dumped raw frames for the offline battery.
    const lines = r.log.map(x => JSON.stringify(x)).join('\n');
    fs.writeFileSync(path.join(OUT, `probe_ep${e}.jsonl`), lines + '\n');
    fs.writeFileSync(path.join(OUT, `probe_ep${e}_chunks.json`), JSON.stringify(r.chunks));
    for (const d of r.dumps) {
      fs.writeFileSync(path.join(OUT, `frame_q${d.query}_ext.b64`), d.ext);
      fs.writeFileSync(path.join(OUT, `frame_q${d.query}_wrist.b64`), d.wrist);
      fs.writeFileSync(path.join(OUT, `frame_q${d.query}_state.json`), JSON.stringify(d.state));
    }
    episodes.push({
      episode: plan.episode, lifted: r.lifted, lift_height: r.lift_height,
      gripMax: r.gripMax, queries: r.queries, ticks: r.ticks, error: r.error,
      min_pad_to_vial: r.min_pad_to_vial, vial_final: r.vial_final,
      grasp_target_first: r.log.length ? r.log[0].grasp_target : null,
      cond_mode_first: r.log.length ? r.log[0].cond_mode : null,
      seconds: +secs.toFixed(1),
    });
    console.log(`EP${e}: queries=${r.queries} lifted=${r.lifted} pad=${r.min_pad_to_vial == null ? 'n/a' : (r.min_pad_to_vial * 100).toFixed(1) + 'cm'} grip=${r.gripMax.toFixed(2)} tgt=${JSON.stringify(episodes[e].grasp_target_first)} mode=${episodes[e].cond_mode_first} err=${r.error} ${secs.toFixed(1)}s`);
    if (r.error) { console.error('PROBE_ERROR episode errored: ' + r.error); await browser.close(); process.exit(1); }
  }
  fs.writeFileSync(path.join(OUT, 'probe_episodes.json'), JSON.stringify({ episodes }, null, 2));
  console.log('PROBE_EPISODES ' + JSON.stringify(episodes.map(e => ({ ep: e.episode, pad_cm: e.min_pad_to_vial == null ? null : +(e.min_pad_to_vial * 100).toFixed(1), lifted: e.lifted }))));
  await browser.close();
  try { fs.rmSync(PROFILE, { recursive: true, force: true }); } catch (e) {}
  console.log('PROBE_BROWSER_DONE');
})().catch(e => { console.error('PROBE_ERROR', e && e.stack || e); process.exit(1); });
