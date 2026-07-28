// DAgger-on-browser rollout + capture harness for the UR5e / Robotiq drug-sort cell.
//
// This is the ONE new integration that makes DAgger run on the DEPLOYMENT renderer.
// It MERGES two existing harnesses:
//   * eval_browser_groot.js  — drives the CURRENT served GR00T through the REAL
//     deployment path (POST exterior+wrist Three.js frames + proprio to the
//     same-origin /api/groot/get_action proxy -> agent-service -> GR00T bridge ->
//     served checkpoint) and applies the returned absolute-joint-target chunk.
//   * browser_harness.js     — captures, per 20 Hz control step, the exterior +
//     wrist frames rendered by the Playground's REAL Three.js deployment sensor
//     path (window.MultiAgentGroot.makeCapture over the room/wrist MJCF cameras)
//     paired with proprio + ground-truth vial/pocket/pinch poses.
//
// So: we roll out the policy the policy ACTUALLY visits at deploy time, and at
// every visited (often failing) state we record the Three.js frame + proprio +
// the GT geometry the IK-expert relabeler needs. dagger_relabel_assemble.py then
// relabels each visited state with the corrective IK expert's correct ABSOLUTE
// joint target -> the DAgger aggregation data, on deployment-matched pixels.
//
// Per episode -> <OUT>/raw/epNNN/{exterior,wrist}/fNNNN.png + meta.json carrying:
//   states  (N x 7 proprio: 6 arm qpos + measured grip closure)
//   gt      (N x {pinch:[3], vial:[7], pocket:[3], grip})  — relabeler inputs
//   vial_xyz / pocket_xyz (post-settle anchors for the IK solve), home_q, vial_qpos
//   rollout diagnostics (success/lifted/seated/lift_height/place_dist/queries)
//
// Note on the grip channel (matches the proven harnesses):
//   * the RECORDED proprio state grip = MEASURED closure (qpos/GRIP_RANGE) —
//     identical to browser_harness.js + dagger_ur5e_drugsort.py training data.
//   * the grip in the obs SENT to the policy = COMMANDED grip — identical to
//     eval_browser_groot.js, i.e. exactly what the deployed pilot feeds GR00T,
//     so the rollout visits the true deploy-time state distribution.
//
// SAFE: OWN Chrome, UNIQUE --user-data-dir, PID captured to <OUT>/rollout_pid.txt,
// cleaned up by PID (NEVER pkill). The only network write is the deployment proxy
// (exactly what the real ?agents=groot pilot does).
//
// Env: PLANS OUT PORT(8031) N N_ACTION_STEPS(8) MAX_TICKS(400)
//      CHROME PUPPETEER_CORE DISPLAY INSTRUCTION
const fs = require('fs');
const path = require('path');
const puppeteer = require(process.env.PUPPETEER_CORE || 'puppeteer-core');

const PLANS = JSON.parse(fs.readFileSync(process.env.PLANS || 'plans_dagger.json', 'utf8'));
const OUT = process.env.OUT || path.resolve('dagger_rollout_out');
const PORT = process.env.PORT || '8031';
const N = process.env.N ? parseInt(process.env.N) : PLANS.plans.length;
const N_ACTION_STEPS = process.env.N_ACTION_STEPS ? parseInt(process.env.N_ACTION_STEPS) : 8;
const MAX_TICKS = process.env.MAX_TICKS ? parseInt(process.env.MAX_TICKS) : 400;
const INSTRUCTION = process.env.INSTRUCTION || 'pick up the vial and place it in the rack';
const CHROME = process.env.CHROME || '/usr/bin/google-chrome-stable';
const RAW = path.join(OUT, 'raw');
const PROFILE = path.join(OUT, 'chrome-udd-dagger-' + Date.now());
// GL override args removed: on the current driver/Chrome-150 stack the
// angle/gl-egl combination breaks WebGL context creation entirely
// (webgl2:false -> viewer never initializes). Plain headless works (3.9s boot).
const GL_ARGS = (process.env.GL_ARGS || '').split(' ').filter(Boolean);
const sleep = (ms) => new Promise(r => setTimeout(r, ms));

// One-time in-page setup: ids (incl. gr_pinch), free arm collision, deployment
// sensor cameras + capture fn, disable the rAF auto-stepper. Stored on window.__dg.
const SETUP = () => {
  const v = window.viewer, pm = v.physics, mj = pm.mj, m = pm.mjModel;
  pm.enabled = false;   // CRITICAL: stop the animate-loop auto-stepper corrupting our rollout
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
  const G = window.MultiAgentGroot;
  const camExt = G.makeMjCamera(pm, mj, 'room');
  const camWrist = G.makeMjCamera(pm, mj, 'wrist');
  const cap = G.makeCapture(v.renderer);
  window.__dg = { armAct, gripAct, armQadr, gripQadr, vialBody, vialQ, vialDof, pocketSite, pinchSite, homeKey, camExt, camWrist, cap };
  return { freed, pinchSite, vialQ, pocketSite, homeKey };
};

// One in-browser DAgger rollout: randomized start -> closed-loop query/act while
// capturing every visited state's Three.js frames + proprio + GT geometry.
const RUN_ROLLOUT = async (plan, cfg) => {
  const H = window.__dg, v = window.viewer, pm = v.physics, mj = pm.mj, m = pm.mjModel, G = window.MultiAgentGroot;
  const GRIP_CLOSE = 255, GRIP_RANGE = 0.8, SETTLE = 180;
  const setArm = q => { const c = pm.mjData.ctrl; for (let i = 0; i < 6; i++) if (H.armAct[i] >= 0) c[H.armAct[i]] = q[i]; };
  const setGrip = g => { if (H.gripAct >= 0) pm.mjData.ctrl[H.gripAct] = g; };
  const readArm = () => H.armQadr.map(a => a >= 0 ? pm.mjData.qpos[a] : 0);
  const gripMeas = () => Math.max(0, Math.min(1, pm.mjData.qpos[H.gripQadr] / GRIP_RANGE));
  const vialPose = () => { const b = H.vialBody, x = pm.mjData.xpos, q = pm.mjData.xquat; return [x[b * 3], x[b * 3 + 1], x[b * 3 + 2], q ? q[b * 4] : 1, q ? q[b * 4 + 1] : 0, q ? q[b * 4 + 2] : 0, q ? q[b * 4 + 3] : 0]; };
  const pocketPose = () => { const s = H.pocketSite, sp = pm.mjData.site_xpos; return [sp[s * 3], sp[s * 3 + 1], sp[s * 3 + 2]]; };
  const pinchPos = () => { const s = H.pinchSite, sp = pm.mjData.site_xpos; return [sp[s * 3], sp[s * 3 + 1], sp[s * 3 + 2]]; };
  const decim = Math.max(1, Math.round((1 / 20) / (m.opt.timestep || 0.002)));

  // Randomized reset (same override the data-gen + eval harnesses use).
  mj.mj_resetDataKeyframe(m, pm.mjData, H.homeKey);
  { const qp = pm.mjData.qpos; for (let i = 0; i < 7; i++) qp[H.vialQ + i] = plan.vial_qpos[i]; }
  { const dv = pm.mjData.qvel; for (let i = 0; i < 6; i++) dv[H.vialDof + i] = 0; }
  m.opt.noslip_iterations = 20;
  mj.mj_forward(m, pm.mjData);
  const homeQ = plan.home_q;
  for (let k = 0; k < SETTLE; k++) { setArm(homeQ); setGrip(0); mj.mj_step(m, pm.mjData); }

  // Post-settle anchors for the IK-expert solve (matches dagger _build_expert).
  const vp0 = vialPose();
  const vialXYZ = [vp0[0], vp0[1], vp0[2]];
  const pocketXYZ = pocketPose();
  const z0 = vialXYZ[2];
  let zMax = z0, ticks = 0, queries = 0, gripCmd = 0, gripMax = 0, err = null;
  const states = [], gt = [];
  // Fresh per-episode session id: the steering sidecar keys its per-session
  // state (phase machine, t_norm tick counter, hold-last target) by sid.
  const sid = 'dg-' + Date.now() + '-' + Math.floor(Math.random() * 1e6);
  let chunkQ = null, chunkG = null, ci = 0, chunkLen = 0;

  while (ticks < cfg.maxTicks) {
    const measured = readArm();
    const gm = gripMeas();
    // --- record THIS visited state (proprio grip = MEASURED, training convention) ---
    states.push([measured[0], measured[1], measured[2], measured[3], measured[4], measured[5], gm]);
    if (pm.syncVisuals) pm.syncVisuals();   // push MuJoCo-WASM state into the Three.js scene before rendering
    const ext = H.cap(v.scene, H.camExt.cam);
    G.syncMjCamera(H.camWrist.cam, pm, mj, H.camWrist.id);   // wrist cam moves with the arm
    const wr = H.cap(v.scene, H.camWrist.cam);
    const pad = String(ticks).padStart(4, '0');
    await window.__saveFrame(cfg.epDir + '/exterior/f' + pad + '.png', ext.replace(/^data:image\/png;base64,/, ''));
    await window.__saveFrame(cfg.epDir + '/wrist/f' + pad + '.png', wr.replace(/^data:image\/png;base64,/, ''));
    gt.push({ pinch: pinchPos(), vial: vialPose(), pocket: pocketPose(), grip: gm });

    // --- query the SERVED policy per chunk (obs grip = COMMANDED, deploy convention) ---
    if (chunkQ === null || ci >= chunkLen) {
      // 10-D deploy-matched obs: proprio7 + grasp_target3 (base frame = negated
      // world x,y). The dr-generation checkpoints are observer-conditioned; the
      // bridge maps state[7:10] to the grasp_target modality. GT vial stands in
      // for the observer estimate (measured median gap 0.25-0.34cm).
      const vpNow = vialPose();
      const obsState = [measured[0], measured[1], measured[2], measured[3], measured[4], measured[5], Math.max(0, Math.min(1, gripCmd)),
                        -vpNow[0], -vpNow[1], vpNow[2]];
      const body = { image_b64: ext, state: obsState, instruction: cfg.instruction, image_b64_wrist: wr, sid };
      let res;
      try {
        // 45s abort per query + one retry: a wedged tunnel/proxy must fail the
        // episode, never hang the whole run past the CDP protocol timeout.
        const q = async () => {
          const ac = new AbortController();
          const t = setTimeout(() => ac.abort(), 45000);
          try {
            const resp = await fetch('/api/groot/get_action', { method: 'POST',
              headers: { 'content-type': 'application/json' },
              body: JSON.stringify(body), signal: ac.signal });
            if (!resp.ok) throw new Error('proxy ' + resp.status + ': ' + (await resp.text()).slice(0, 120));
            return await resp.json();
          } finally { clearTimeout(t); }
        };
        try { res = await q(); } catch (e1) { res = await q(); }
      } catch (e) { err = String(e); break; }
      queries++;
      chunkQ = (res.chunk_q && res.chunk_q.length) ? res.chunk_q : [res.q];
      chunkG = (res.chunk_grip && res.chunk_grip.length) ? res.chunk_grip : [res.grip];
      chunkLen = Math.min(cfg.nActionSteps, chunkQ.length);
      ci = 0;
    }

    // --- apply the policy action for this tick, hold it for `decim` physics steps ---
    gripCmd = Math.max(0, Math.min(1, chunkG[ci]));
    gripMax = Math.max(gripMax, gripCmd);
    setArm(chunkQ[ci]); setGrip(gripCmd * GRIP_CLOSE);
    for (let s = 0; s < decim; s++) mj.mj_step(m, pm.mjData);
    const vz = vialPose()[2]; if (vz > zMax) zMax = vz;
    ci++; ticks++;
  }

  const vp = vialPose(), pp = pocketPose();
  const placeDist = Math.hypot(vp[0] - pp[0], vp[1] - pp[1]);
  const lifted = (zMax - z0) > 0.02;
  const seated = placeDist < 0.05 && vp[2] > 0.20;
  return {
    states, gt, vial_xyz: vialXYZ, pocket_xyz: pocketXYZ,
    success: (lifted && seated), lifted, seated, lift_height: zMax - z0, place_dist: placeDist,
    num_frames: states.length, queries, grip_max: gripMax, error: err,
  };
};

(async () => {
  // KEEP_RAW=1 preserves prior episodes (per-episode driver invocations, Stage C):
  // the unconditional wipe silently destroyed 19/20 collected episodes when the
  // driver called this script once per episode.
  if (process.env.KEEP_RAW !== '1') fs.rmSync(RAW, { recursive: true, force: true });
  fs.mkdirSync(RAW, { recursive: true });
  const browser = await puppeteer.launch({
    executablePath: CHROME, headless: true, userDataDir: PROFILE, protocolTimeout: 900000,
    env: { ...process.env, DISPLAY: process.env.DISPLAY || ':1' },
    args: ['--no-sandbox', '--disable-dev-shm-usage', '--disable-gpu-sandbox', '--window-size=1400,950', ...GL_ARGS],
  });
  const pid = browser.process().pid;
  fs.writeFileSync(path.join(OUT, 'rollout_pid.txt'), String(pid));
  console.log('DAGGER_ROLLOUT_CHROME_PID=' + pid + ' port=' + PORT + ' N=' + N + ' max_ticks=' + MAX_TICKS);
  const page = await browser.newPage();
  await page.setViewport({ width: 1400, height: 950 });
  page.on('console', m => { const t = m.text(); if (/\[groot\]|error|MJCF loaded/i.test(t)) console.log('  [page]', t.slice(0, 150)); });
  await page.exposeFunction('__saveFrame', (rel, b64) => { fs.writeFileSync(path.join(RAW, rel), Buffer.from(b64, 'base64')); });

  const url = `http://localhost:${PORT}/robot-playground.html?demo=drugsorting${process.env.ARM ? `&arm=${process.env.ARM}` : ``}`;
  console.log('GOTO', url);
  await page.goto(url, { waitUntil: "load", timeout: 300000 });
  await page.waitForFunction(() => window.viewer && window.viewer.physics && window.viewer.physics.mj && window.viewer.physics.mjModel, { timeout: 300000 });
  await page.waitForFunction(() => {
    const pm = window.viewer.physics, mj = pm.mj; if (!pm.mjModel) return false;
    return mj.mj_name2id(pm.mjModel, 1, 'vial_0') >= 0 && !!window.MultiAgentGroot;
  }, { timeout: 300000 });
  await sleep(1200);
  const setup = await page.evaluate(SETUP);
  console.log('SETUP ' + JSON.stringify(setup));

  // Confirm the deployment proxy (current served ckpt) is reachable before rolling out.
  const health = await page.evaluate(async () => { try { const r = await fetch('/api/groot/health'); return await r.json(); } catch (e) { return { error: String(e) }; } });
  console.log('HEALTH ' + JSON.stringify(health));
  if (!health || health.bridge_reachable === false) {
    console.error('DAGGER_ROLLOUT_ERROR bridge not reachable: ' + JSON.stringify(health));
    await browser.close(); process.exit(2);
  }

  const results = [];
  const t0all = Date.now();
  const plans = PLANS.plans.slice(0, N);
  for (const plan of plans) {
    const ep = plan.episode;
    const epRel = 'ep' + String(ep).padStart(3, '0');
    fs.mkdirSync(path.join(RAW, epRel, 'exterior'), { recursive: true });
    fs.mkdirSync(path.join(RAW, epRel, 'wrist'), { recursive: true });
    const t0 = Date.now();
    const r = await page.evaluate(RUN_ROLLOUT, plan, { maxTicks: MAX_TICKS, nActionSteps: N_ACTION_STEPS, epDir: epRel, instruction: INSTRUCTION });
    const secs = (Date.now() - t0) / 1000;
    fs.writeFileSync(path.join(RAW, epRel, 'meta.json'), JSON.stringify({
      episode: ep, nominal: plan.nominal, vial_qpos: plan.vial_qpos, home_q: plan.home_q,
      vial_xyz: r.vial_xyz, pocket_xyz: r.pocket_xyz, ik_max_err_mm: plan.ik_max_err_mm || 0.0,
      states: r.states, gt: r.gt,
      success: r.success, lifted: r.lifted, seated: r.seated, lift_height: r.lift_height,
      place_dist: r.place_dist, num_frames: r.num_frames, queries: r.queries, grip_max: r.grip_max,
      error: r.error, seconds: secs,
    }));
    results.push({ episode: ep, success: r.success, lifted: r.lifted, frames: r.num_frames, seconds: +secs.toFixed(1) });
    console.log(`ROLL ep${String(ep).padStart(3, '0')}: ${r.success ? 'SUCCESS' : (r.error ? 'ERROR:' + r.error : 'fail')} lifted=${r.lifted} seated=${r.seated} lift=${(r.lift_height * 100).toFixed(1)}cm place=${(r.place_dist * 100).toFixed(1)}cm grip=${r.grip_max.toFixed(2)} frames=${r.num_frames} q=${r.queries} ${secs.toFixed(1)}s`);
  }
  const totalSecs = (Date.now() - t0all) / 1000;
  const nSucc = results.filter(r => r.success).length;
  const nLift = results.filter(r => r.lifted).length;
  const totFrames = results.reduce((a, r) => a + r.frames, 0);
  const summary = {
    port: PORT, num_episodes: results.length, rollout_success: nSucc, rollout_lifted: nLift,
    total_frames: totFrames, max_ticks: MAX_TICKS, n_action_steps: N_ACTION_STEPS,
    total_seconds: +totalSecs.toFixed(1), results,
  };
  fs.writeFileSync(path.join(OUT, 'rollout_summary.json'), JSON.stringify(summary, null, 2));
  console.log('ROLLOUT_SUMMARY ' + JSON.stringify({ eps: results.length, rollout_success: nSucc, lifted: nLift, frames: totFrames }));
  await browser.close();
  try { fs.rmSync(PROFILE, { recursive: true, force: true }); } catch (e) {}
  console.log('DAGGER_ROLLOUT_DONE');
})().catch(e => { console.error('DAGGER_ROLLOUT_ERROR', e && e.stack || e); process.exit(1); });
