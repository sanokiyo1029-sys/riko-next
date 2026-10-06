// Riko's "will": she decides what to do next on her own and moves around the stage.
//
//  - Needs (energy, boredom, curiosity, social) drift over time.
//  - Every few seconds she scores each action against her needs + personality
//    traits + what's happening (cursor, conversation), then picks one with a
//    weighted random choice, so she is driven but not predictable.
//  - Movement uses "arrive" steering (accelerates, slows down near the target).
//  - The conversation always wins: when you talk she comes to you, when she
//    talks she stays put, when she thinks she paces.
//  - The LLM can also choose actions ([act:wander] tags -> perform()).
//
// Works with both the 2D face and the VRM avatar: it only moves the container.
// Riko Next adaptations: real (non-square) box size for the full-body VRM, a drag on the avatar
// (turning the 3D camera) is not a poke, and mirroring can be turned off (`mirror: false`) so the
// 3D model turns her body via `facing`/`speed` instead of being flipped.

const clamp = (v, a, b) => Math.max(a, Math.min(b, v));
const rand = (a, b) => a + Math.random() * (b - a);

function weightedPick(entries) {
  const total = entries.reduce((s, [, w]) => s + w, 0);
  let r = Math.random() * total;
  for (const [k, w] of entries) if ((r -= w) <= 0) return k;
  return entries[0][0];
}

export const ACTION_NAMES = ['idle', 'wander', 'inspect', 'follow', 'nap', 'hop', 'edge', 'come'];

export function createBehavior(stage, el, { traits = {}, onActivity, onPoke, mirror = true } = {}) {
  const T = { energy: 0.6, curiosity: 0.7, clinginess: 0.5, ...traits };
  const reduced = matchMedia('(prefers-reduced-motion: reduce)').matches;

  const s = {
    t: 0, x: 0, vx: 0, y: 0, vy: 0, facing: 1,
    target: null, activity: '', until: 0, hop: 0, last: '',
    needs: { energy: 0.8, boredom: 0.3, curiosity: 0.5, social: 0.4 },
    mouse: { x: 0, y: 0, movedAt: -1e9, inside: false },
    override: null,   // conversation-driven activity
  };

  // ------------------------------------------------------------------ geometry
  const size = () => el.offsetWidth || 220;
  const height = () => el.offsetHeight || size();
  function bounds() {
    const w = stage.clientWidth, h = stage.clientHeight, sz = size();
    const maxX = Math.max(0, w - sz), maxY = Math.max(0, h - height() - 110); // keep clear of controls/subtitles
    return { minX: 0, maxX, minY: Math.min(h * 0.04, maxY), maxY };
  }
  const randomPoint = () => { const b = bounds(); return { x: rand(b.minX, b.maxX), y: rand(b.minY, b.maxY) }; };
  const homePoint = () => { const b = bounds(); return { x: (b.minX + b.maxX) / 2, y: b.minY + (b.maxY - b.minY) * 0.55 }; };
  const nearMouse = () => {
    const sz = size(), side = s.mouse.x > s.x + sz / 2 ? -1 : 1;   // stop beside the cursor, not on it
    return { x: s.mouse.x - sz / 2 + side * sz * 0.6, y: s.mouse.y - height() * 0.6 };
  };
  const mouseActive = (sec) => s.mouse.inside && s.t - s.mouse.movedAt < sec;
  const dist = (p) => Math.hypot(p.x - s.x, p.y - s.y);

  // ------------------------------------------------------------------ actions
  // score(): how much she wants to do it now. start(): set it up. tick(dt) -> true when finished.
  const arrived = () => !s.target || dist(s.target) < 6;
  const A = {
    idle: {
      score: () => 0.35 + (1 - s.needs.energy) * 0.4,
      start: () => { s.target = null; s.until = s.t + rand(2.5, 6); },
      tick: () => s.t > s.until,
    },
    wander: {
      score: () => (0.2 + s.needs.boredom) * (0.4 + T.energy) * s.needs.energy,
      start: () => { s.target = randomPoint(); s.until = s.t + 8; },
      tick: () => arrived() || s.t > s.until,
    },
    inspect: {   // walk over to look at the cursor
      score: () => (mouseActive(3) ? 1.2 : 0) * s.needs.curiosity * T.curiosity,
      start: () => { s.target = nearMouse(); s.until = s.t + 6; s.needs.curiosity *= 0.3; },
      tick: () => arrived() || s.t > s.until,
    },
    follow: {    // tag along behind the cursor for a while
      score: () => (mouseActive(1.5) ? 1 : 0) * s.needs.social * T.clinginess,
      start: () => { s.until = s.t + rand(4, 8); },
      tick: () => { if (s.mouse.inside) s.target = nearMouse(); s.needs.social = Math.max(0, s.needs.social - 0.002); return s.t > s.until; },
    },
    nap: {
      score: () => (s.needs.energy < 0.25 ? 2 : 0),
      start: () => { s.target = null; s.until = s.t + rand(10, 20); el.classList.add('sleeping'); },
      tick: () => s.t > s.until || s.needs.energy > 0.95,
      end: () => el.classList.remove('sleeping'),
    },
    hop: {
      score: () => s.needs.boredom * 0.35 * T.energy * s.needs.energy,
      start: () => { s.target = null; s.hop = 0.001; },
      tick: () => s.hop === 0,
    },
    edge: {      // go peek at the side of the stage
      score: () => s.needs.curiosity * 0.35 * T.curiosity,
      start: () => { const b = bounds(); s.target = { x: Math.random() < 0.5 ? b.minX : b.maxX, y: rand(b.minY, b.maxY) }; s.until = s.t + 8; },
      tick: () => arrived() || s.t > s.until,
    },
    come: {      // come back to the middle
      score: () => 0,  // only chosen by the conversation or the LLM
      start: () => { s.target = homePoint(); s.until = s.t + 6; },
      tick: () => arrived() || s.t > s.until,
    },
    // conversation-only states
    attend: { start: () => { s.target = homePoint(); }, tick: () => false },
    talk:   { start: () => { s.target = null; }, tick: () => false },
    think:  {
      start: () => { s.pace = s.x; s.target = { x: s.x + 40, y: s.y }; },
      tick: () => { if (arrived()) s.target = { x: s.pace + (s.target.x > s.pace ? -40 : 40), y: s.y }; return false; },
    },
  };

  function begin(name) {
    if (s.activity && A[s.activity]?.end) A[s.activity].end();
    s.last = s.activity;
    s.activity = name;
    A[name].start();
    onActivity?.(name);
  }

  function choose() {
    const entries = ACTION_NAMES
      .map((n) => [n, A[n].score() * (n === s.last ? 0.5 : 1) * rand(0.8, 1.2)])
      .filter(([, w]) => w > 0)
      .map(([n, w]) => [n, w * w]);   // sharpen: strong wants win more often
    begin(entries.length ? weightedPick(entries) : 'idle');
  }

  // ------------------------------------------------------------------ input
  const onMove = (e) => {
    const r = stage.getBoundingClientRect();
    s.mouse.x = e.clientX - r.left; s.mouse.y = e.clientY - r.top;
    s.mouse.movedAt = s.t; s.mouse.inside = true;
  };
  const onLeave = () => { s.mouse.inside = false; };
  let down = null;
  const onDown = (e) => { down = { x: e.clientX, y: e.clientY }; };
  const onClick = (e) => {
    if (down && Math.hypot(e.clientX - down.x, e.clientY - down.y) > 5) return;   // that was a drag, not a poke
    s.needs.boredom = 0; s.needs.social = Math.min(1, s.needs.social + 0.3);
    if (!s.override) begin('hop');
    onPoke?.();
  };
  stage.addEventListener('pointermove', onMove);
  stage.addEventListener('pointerleave', onLeave);
  el.addEventListener('pointerdown', onDown);
  el.addEventListener('click', onClick);

  // start in the middle
  const h0 = homePoint(); s.x = h0.x; s.y = h0.y;
  begin('idle');

  // ------------------------------------------------------------------ per frame
  function update(dt, { convo = 'listening', userSpeaking = false, rikoSpeaking = false } = {}) {
    s.t += dt;

    // 1. conversation overrides autonomy
    const want = rikoSpeaking || convo === 'speaking' ? 'talk'
      : userSpeaking || convo === 'transcribing' ? 'attend'
      : convo === 'thinking' ? 'think' : null;
    if (want !== s.override) {
      s.override = want;
      if (want) begin(want); else { begin('idle'); s.until = s.t + rand(1, 3); }
    }

    // 2. needs drift
    const n = s.needs, moving = Math.hypot(s.vx, s.vy) > 10;
    n.energy = clamp(n.energy + (s.activity === 'nap' ? 0.06 : moving ? -0.012 : 0.004) * dt, 0, 1);
    n.boredom = clamp(n.boredom + (s.activity === 'idle' ? 0.05 : -0.03) * dt, 0, 1);
    n.curiosity = clamp(n.curiosity + 0.02 * dt, 0, 1);
    n.social = clamp(n.social + (userSpeaking ? -0.05 : 0.008) * dt, 0, 1);

    // 3. finish / choose
    if (A[s.activity].tick(dt) && !s.override) choose();

    // 4. steering ("arrive")
    const maxSpeed = (s.activity === 'follow' ? 230 : 150) * (0.55 + n.energy * 0.6) * (reduced ? 0.35 : 1);
    let ax = -s.vx * 4, ay = -s.vy * 4;   // damping when there is no target
    if (s.target) {
      const b = bounds();
      s.target.x = clamp(s.target.x, b.minX, b.maxX); s.target.y = clamp(s.target.y, b.minY, b.maxY);
      const dx = s.target.x - s.x, dy = s.target.y - s.y, d = Math.hypot(dx, dy) || 1;
      const speed = maxSpeed * Math.min(1, d / 120);
      ax = ((dx / d) * speed - s.vx) * 5; ay = ((dy / d) * speed - s.vy) * 5;
    }
    s.vx += ax * dt; s.vy += ay * dt;
    s.x += s.vx * dt; s.y += s.vy * dt;
    const b = bounds();
    s.x = clamp(s.x, b.minX, b.maxX); s.y = clamp(s.y, b.minY, b.maxY);

    // 5. look direction: toward movement, else toward the cursor, else front
    if (Math.abs(s.vx) > 20) s.facing = Math.sign(s.vx);
    else if (mouseActive(2)) s.facing = s.mouse.x > s.x + size() / 2 ? 1 : -1;

    // 6. draw: walking bob, lean, hop arc
    const speedNow = Math.hypot(s.vx, s.vy);
    const bob = speedNow > 10 ? -Math.abs(Math.sin(s.t * 11)) * 6 * Math.min(1, speedNow / 100) : Math.sin(s.t * 1.6) * 2;
    let hopY = 0;
    if (s.hop > 0) { s.hop += dt / 0.55; hopY = -Math.sin(Math.min(1, s.hop) * Math.PI) * (reduced ? 15 : 55); if (s.hop >= 1) s.hop = 0; }
    const lean = clamp(s.vx / 200, -1, 1) * 7;
    el.style.transform = `translate3d(${s.x}px, ${s.y + bob + hopY}px, 0) rotate(${lean}deg)`
      + (mirror ? ` scaleX(${s.facing})` : '');
    const sz = size();
    stage.style.setProperty('--rx', `${s.x + sz / 2}px`);
    stage.style.setProperty('--ry', `${s.y + height() / 2}px`);
  }

  return {
    update,
    /** Let something else (the LLM, a button) decide what she does next. */
    perform(name) { if (A[name] && ACTION_NAMES.includes(name) && !s.override) begin(name); },
    get activity() { return s.activity; },
    get facing() { return s.facing; },
    get speed() { return Math.hypot(s.vx, s.vy); },
    get needs() { return { ...s.needs }; },
    dispose() {
      stage.removeEventListener('pointermove', onMove);
      stage.removeEventListener('pointerleave', onLeave);
      el.removeEventListener('pointerdown', onDown);
      el.removeEventListener('click', onClick);
      el.style.transform = '';
    },
  };
}
