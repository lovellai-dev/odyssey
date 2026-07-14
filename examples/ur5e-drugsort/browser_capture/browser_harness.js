// Browser-frame (Three.js) data-generation harness for the AseptiPack UR5e cell.
//
// This is the middle stage of the browser-frame harness that closes the RENDER GAP:
// GR00T + the Observer are trained on MuJoCo `Renderer` frames but DEPLOY on the
// Lovell AI Robot Playground's Three.js renderer, so both degrade in-browser. To
// retrain on the deployment appearance we capture training frames from the real
// deployment renderer here.
//
// Per episode it replays the adaptive-IK expert (precomputed in plans.json by
// precompute_plans.py) INSIDE the browser Playground: it steps the real MuJoCo-WASM
// physics and captures, per 20 Hz control step, the exterior + wrist frames rendered
// by the Playground's REAL deployment sensor path (multiagent/groot-pilot.js
// `makeCapture` -> the SAME THREE.WebGLRenderer + scene the served GR00T policy
// receives each tick) paired with proprio (6 qpos + grip closure), the ground-truth
// vial + nest pose (from MuJoCo-WASM state, for the Observer's labels), and the
// expert action (6 absolute joint targets + grip 0..1). Raw output ->
// <OUT>/raw/epNNN/{exterior,wrist}/*.png + meta.json, assembled into a LeRobot v2.1
// dataset by assemble_lerobot.py.
//
// Why browser-drive (not a standalone headless-Three.js renderer): fidelity. These
// frames must match what the policy actually deploys on. Rendering `window.viewer.scene`
// from the MJCF `room`/`wrist` cameras through the Playground's own
// `window.viewer.renderer` (with its PBR "cell-upgrade" materials — glass vials,
// polished metal, wood worktop) reproduces the deployment appearance exactly; a
// separate Three scene would risk material/lighting drift.
//
// SAFE isolation: OWN Chrome, UNIQUE --user-data-dir, PID captured to
// <OUT>/harness_pid.txt, cleaned up by PID (NEVER pkill). Read-only page loads on the
// Playground static server — the EXPERT path is pure client-side physics + Three.js
// render; it never touches the GR00T /api bridge, so it disturbs no running service.
//
// Config via env vars:
//   PLANS           plans.json path                       (default ./plans.json)
//   OUT             output dir (raw/, summary)            (default ./browser_capture_out)
//   PORT            Playground server port                (default 8021)
//   CHROME          Chrome/Chromium executable            (default /usr/bin/google-chrome-stable)
//   PUPPETEER_CORE  puppeteer-core module path/name       (default 'puppeteer-core')
//   DISPLAY         X display for headed GL               (default :1)
const fs = require('fs');
const path = require('path');
const puppeteer = require(process.env.PUPPETEER_CORE || 'puppeteer-core');

const PLANS_PATH = process.env.PLANS || path.resolve('plans.json');
const OUT = process.env.OUT || path.resolve('browser_capture_out');
const PORT = process.env.PORT || '8021';
const CHROME = process.env.CHROME || '/usr/bin/google-chrome-stable';
const PLANS = JSON.parse(fs.readFileSync(PLANS_PATH, 'utf8'));
const RAW = path.join(OUT, 'raw');
const PROFILE = path.join(OUT, 'chrome-udd-harness-' + Date.now());
const DECIM = 25;          // (1/20 s) / 0.002 s timestep  -> 20 Hz control/record (matches the headless generator)
const MAX_STEPS = 30000;
// Headless WebGL via ANGLE (software/GPU EGL) so the Three.js renderer runs off-screen.
const GL_ARGS = ['--use-gl=angle', '--use-angle=gl-egl', '--ignore-gpu-blocklist', '--enable-webgl', '--enable-gpu-rasterization'];
const sleep = (ms) => new Promise(r => setTimeout(r, ms));

// ── One-time in-page setup: resolve ids, free arm collision, build the deployment
//    sensor cameras + capture fn, disable the rAF auto-stepper. Stored on window.__hns.
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
  const homeKey = mj.mj_name2id(m, OBJ_KEY, 'home');
  // Free arm-link + non-pad gripper collisions (arm swings free above worktop; only pads grip).
  const ARM_BODIES = new Set(['base', 'shoulder_link', 'upper_arm_link', 'forearm_link', 'wrist_1_link', 'wrist_2_link', 'wrist_3_link']);
  let freed = 0;
  for (let g = 0; g < m.ngeom; g++) {
    const bn = mj.mj_id2name(m, OBJ_BODY, m.geom_bodyid[g]) || '';
    const gn = mj.mj_id2name(m, OBJ_GEOM, g) || '';
    if ((ARM_BODIES.has(bn) || bn.startsWith('gr_')) && !gn.includes('pad')) { m.geom_contype[g] = 0; m.geom_conaffinity[g] = 0; freed++; }
  }
  // Deployment sensor path (identical to what the served GR00T policy receives each tick).
  const G = window.MultiAgentGroot;
  const camExt = G.makeMjCamera(pm, mj, 'room');
  const camWrist = G.makeMjCamera(pm, mj, 'wrist');
  const cap = G.makeCapture(v.renderer);
  window.__hns = { armAct, gripAct, armQadr, gripQadr, vialBody, vialQ, vialDof, pocketSite, homeKey, camExt, camWrist, cap };
  return { freed, armAct, gripAct, vialBody, vialQ, vialDof, pocketSite, homeKey, gripQadr, extCamId: camExt.id, wrCamId: camWrist.id };
};

// ── Run ONE episode in-page. Replays the FSM (a direct port of PickPlaceController)
//    against the plan's adaptive phases, streaming frames to disk via window.__saveFrame.
const RUN_EPISODE = async (plan, cfg) => {
  const H = window.__hns, v = window.viewer, pm = v.physics, mj = pm.mj, m = pm.mjModel, G = window.MultiAgentGroot;
  const GRIP_CLOSE = 255, MOVE = 800, CONV = 6e-3, CONV_MAX = 2200, SETTLE = 180, GRIP_RANGE = 0.8;
  const ease = t => t < 0.5 ? 2 * t * t : 1 - Math.pow(-2 * t + 2, 2) / 2;
  const lerp = (a, b, t) => a + (b - a) * t;
  // FRESH accessors: WASM heap views detach when it grows — re-read pm.mjData.* every call.
  const setArm = q => { const c = pm.mjData.ctrl; for (let i = 0; i < 6; i++) if (H.armAct[i] >= 0) c[H.armAct[i]] = q[i]; };
  const setGrip = g => { if (H.gripAct >= 0) pm.mjData.ctrl[H.gripAct] = g; };
  const readArm = () => H.armQadr.map(a => a >= 0 ? pm.mjData.qpos[a] : 0);
  const armErr = (q, meas) => { let mx = 0; for (let i = 0; i < 6; i++) { const e = Math.abs(q[i] - meas[i]); if (e > mx) mx = e; } return mx; };
  const gripMeas = () => Math.max(0, Math.min(1, pm.mjData.qpos[H.gripQadr] / GRIP_RANGE));
  const vialPose = () => { const b = H.vialBody, x = pm.mjData.xpos, q = pm.mjData.xquat; return [x[b * 3], x[b * 3 + 1], x[b * 3 + 2], q ? q[b * 4] : 1, q ? q[b * 4 + 1] : 0, q ? q[b * 4 + 2] : 0, q ? q[b * 4 + 3] : 0]; };
  const pocketPose = () => { const s = H.pocketSite, sp = pm.mjData.site_xpos; return [sp[s * 3], sp[s * 3 + 1], sp[s * 3 + 2]]; };

  // Reset to home keyframe, override the vial to the plan's randomized pose, enable noslip
  // (prevents the stiff pad/vial contact ejecting the grasped vial on lift — matches the
  // headless generator's model.opt.noslip_iterations=20).
  mj.mj_resetDataKeyframe(m, pm.mjData, H.homeKey);
  { const qp = pm.mjData.qpos; for (let i = 0; i < 7; i++) qp[H.vialQ + i] = plan.vial_qpos[i]; }
  { const dv = pm.mjData.qvel; for (let i = 0; i < 6; i++) dv[H.vialDof + i] = 0; }
  m.opt.noslip_iterations = 20;
  mj.mj_forward(m, pm.mjData);

  const homeQ = plan.home_q;
  for (let k = 0; k < SETTLE; k++) { setArm(homeQ); setGrip(0); mj.mj_step(m, pm.mjData); }

  const phases = plan.phases;
  let prev = readArm(), phase = 0, s = 0, step = 0, frameIdx = 0;
  const states = [], actions = [], gt = [];
  const z0 = vialPose()[2];
  let zMax = z0;
  const capOne = (camObj, refresh) => { if (refresh) G.syncMjCamera(camObj.cam, pm, mj, camObj.id); return H.cap(v.scene, camObj.cam); };

  while (step < cfg.maxSteps) {
    const measured = readArm();
    let armCtrl, gripCtrl, finished = false;
    if (phase >= phases.length) { finished = true; const ph = phases[phases.length - 1]; armCtrl = ph.q; gripCtrl = ph.grip; }
    else {
      const ph = phases[phase];
      const same = ph.q.every((x, i) => Math.abs(x - prev[i]) < 1e-3);
      const phaseMove = (ph.move_steps != null) ? ph.move_steps : MOVE;
      const moveDur = same ? 0 : phaseMove;
      armCtrl = (s < moveDur) ? ph.q.map((x, i) => lerp(prev[i], x, ease(s / moveDur))) : ph.q;
      gripCtrl = ph.grip;
      s++;
      if (s >= moveDur) {
        const held = s - moveDur;
        const done = (ph.hold === 'conv') ? (armErr(ph.q, measured) < CONV || held > CONV_MAX) : (held >= ph.hold);
        if (done) { prev = ph.q.slice(); phase++; s = 0; }
      }
    }
    setArm(armCtrl); setGrip(gripCtrl);
    mj.mj_step(m, pm.mjData);
    step++;
    const vz = vialPose()[2]; if (vz > zMax) zMax = vz;

    if (step % cfg.decim === 0) {
      const gm = gripMeas();
      states.push([measured[0], measured[1], measured[2], measured[3], measured[4], measured[5], gm]);
      actions.push([armCtrl[0], armCtrl[1], armCtrl[2], armCtrl[3], armCtrl[4], armCtrl[5], gripCtrl / GRIP_CLOSE]);
      gt.push({ vial: vialPose(), pocket: pocketPose(), phase: (phase < phases.length ? phases[phase].name : 'finished') });
      if (pm.syncVisuals) pm.syncVisuals();   // push MuJoCo-WASM state into the Three.js scene before rendering
      const ext = capOne(H.camExt, false);
      const wr = capOne(H.camWrist, true);    // wrist cam moves with the arm -> refresh its pose each frame
      const pad = String(frameIdx).padStart(4, '0');
      await window.__saveFrame(cfg.epDir + '/exterior/f' + pad + '.png', ext.replace(/^data:image\/png;base64,/, ''));
      await window.__saveFrame(cfg.epDir + '/wrist/f' + pad + '.png', wr.replace(/^data:image\/png;base64,/, ''));
      frameIdx++;
    }
    if (finished) break;
  }

  const vp = vialPose(), pp = pocketPose();
  const placeDist = Math.hypot(vp[0] - pp[0], vp[1] - pp[1]);
  const lifted = (zMax - z0) > 0.02;
  const seated = placeDist < 0.05 && vp[2] > 0.20;
  return { states, actions, gt, success: (lifted && seated), lifted, seated, lift_height: zMax - z0, place_dist: placeDist, num_frames: frameIdx, sim_steps: step };
};

(async () => {
  fs.rmSync(RAW, { recursive: true, force: true }); fs.mkdirSync(RAW, { recursive: true });
  const browser = await puppeteer.launch({
    executablePath: CHROME, headless: true, userDataDir: PROFILE,
    protocolTimeout: 900000, env: { ...process.env, DISPLAY: process.env.DISPLAY || ':1' },
    args: ['--no-sandbox', '--disable-dev-shm-usage', '--disable-gpu-sandbox', '--window-size=1400,950', ...GL_ARGS],
  });
  const pid = browser.process().pid;
  fs.writeFileSync(path.join(OUT, 'harness_pid.txt'), String(pid));
  console.log('HARNESS_CHROME_PID=' + pid + ' profile=' + PROFILE);
  const page = await browser.newPage();
  await page.setViewport({ width: 1400, height: 950 });
  page.on('console', m => { const t = m.text(); if (/error|MJCF loaded|cell-upgrade/i.test(t)) console.log('  [page]', t.slice(0, 140)); });

  // Stream frames straight to disk (keeps page memory bounded across ~900 frames/episode).
  await page.exposeFunction('__saveFrame', (rel, b64) => { fs.writeFileSync(path.join(RAW, rel), Buffer.from(b64, 'base64')); });

  const url = `http://localhost:${PORT}/robot-playground.html?demo=drugsorting`;
  console.log('GOTO', url);
  await page.goto(url, { waitUntil: 'load', timeout: 120000 });
  await page.waitForFunction(() => window.viewer && window.viewer.physics && window.viewer.physics.mj && window.viewer.physics.mjModel, { timeout: 120000 });
  await page.waitForFunction(() => {
    const pm = window.viewer.physics, mj = pm.mj; if (!pm.mjModel) return false;
    return mj.mj_name2id(pm.mjModel, 1, 'vial_0') >= 0 && !!window.MultiAgentGroot && (typeof window.AseptipackPickPlace === 'function');
  }, { timeout: 120000 });
  await sleep(1200);
  const setup = await page.evaluate(SETUP);
  console.log('SETUP ' + JSON.stringify(setup));

  const results = [];
  const t0all = Date.now();
  for (const plan of PLANS.plans) {
    const ep = plan.episode;
    const epRel = 'ep' + String(ep).padStart(3, '0');
    fs.mkdirSync(path.join(RAW, epRel, 'exterior'), { recursive: true });
    fs.mkdirSync(path.join(RAW, epRel, 'wrist'), { recursive: true });
    const t0 = Date.now();
    const r = await page.evaluate(RUN_EPISODE, plan, { decim: DECIM, maxSteps: MAX_STEPS, epDir: epRel });
    const secs = (Date.now() - t0) / 1000;
    fs.writeFileSync(path.join(RAW, epRel, 'meta.json'), JSON.stringify({
      episode: ep, nominal: plan.nominal, vial_qpos: plan.vial_qpos, home_q: plan.home_q,
      ik_max_err_mm: plan.ik_max_err_mm, states: r.states, actions: r.actions, gt: r.gt,
      success: r.success, lifted: r.lifted, seated: r.seated, lift_height: r.lift_height,
      place_dist: r.place_dist, num_frames: r.num_frames, sim_steps: r.sim_steps, seconds: secs,
    }));
    results.push({ episode: ep, success: r.success, frames: r.num_frames, seconds: +secs.toFixed(1) });
    console.log(`EP ${String(ep).padStart(2, '0')}: ${r.success ? 'SUCCESS' : 'FAIL'} frames=${r.num_frames} sim_steps=${r.sim_steps} lift=${(r.lift_height * 100).toFixed(1)}cm place=${(r.place_dist * 100).toFixed(1)}cm ${secs.toFixed(1)}s`);
  }
  const totalSecs = (Date.now() - t0all) / 1000;
  const nSucc = results.filter(r => r.success).length;
  const avgSecs = results.reduce((a, r) => a + r.seconds, 0) / results.length;
  const avgFrames = results.reduce((a, r) => a + r.frames, 0) / results.length;
  const summary = {
    port: PORT, num_episodes: results.length, success: nSucc, total_seconds: +totalSecs.toFixed(1),
    avg_seconds_per_episode: +avgSecs.toFixed(1), avg_frames_per_episode: Math.round(avgFrames),
    est_full_300_hours: +((avgSecs * 300) / 3600).toFixed(2), results,
  };
  fs.writeFileSync(path.join(OUT, 'harness_summary.json'), JSON.stringify(summary, null, 2));
  console.log('SUMMARY ' + JSON.stringify(summary));
  await browser.close();
  try { fs.rmSync(PROFILE, { recursive: true, force: true }); } catch (e) {}
  console.log('HARNESS_DONE');
})().catch(e => { console.error('HARNESS_ERROR', e && e.stack || e); process.exit(1); });
