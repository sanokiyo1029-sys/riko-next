// Avatar layer. Optional: Riko works without it.
//  - VRM (three.js + @pixiv/three-vrm, loaded on demand from CDN): idle breathing,
//    blinking, lip sync from audio level, emotion expressions, talking head motion.
//  - Fallback: an original 2D face drawn in SVG with the same controller API.
const lerp = (a, b, t) => a + (b - a) * t;
// Respect the OS "reduce motion" setting: no idle sway or head motion, blinking and lip sync only.
const reducedMotion = () => window.matchMedia?.('(prefers-reduced-motion: reduce)').matches;

export async function createAvatar(container, { vrmUrl = '', enabled = true } = {}) {
  if (!enabled) return nullAvatar();
  if (vrmUrl) {
    try { return await createVRMAvatar(container, vrmUrl); }
    catch (e) { console.warn('[avatar] VRM failed, using 2D face:', e); }
  }
  return createFaceAvatar(container);
}

function nullAvatar() { return { setEmotion() {}, setMood() {}, update() {}, dispose() {}, kind: 'none' }; }

// ------------------------------------------------------------------ 2D fallback face
function createFaceAvatar(container) {
  container.innerHTML = `
  <svg class="face" viewBox="0 0 200 200" role="img" aria-label="Riko">
    <defs>
      <radialGradient id="cheek"><stop offset="0" stop-color="#ff86b0" stop-opacity=".55"/><stop offset="1" stop-color="#ff86b0" stop-opacity="0"/></radialGradient>
    </defs>
    <path d="M30 108 C30 50 62 22 100 22 C138 22 170 50 170 108 C170 150 140 178 100 178 C60 178 30 150 30 108Z" fill="#f6e9f1"/>
    <path d="M24 104 C22 42 58 12 100 12 C142 12 178 42 176 104 C164 70 148 54 128 50 C118 66 96 74 70 70 C52 78 36 90 24 104Z" fill="#5a3d8a"/>
    <g id="cheeks" opacity=".35"><ellipse cx="58" cy="128" rx="18" ry="10" fill="url(#cheek)"/><ellipse cx="142" cy="128" rx="18" ry="10" fill="url(#cheek)"/></g>
    <g id="browL"><path d="M50 82 Q66 74 82 80" stroke="#3e2a63" stroke-width="4" fill="none" stroke-linecap="round"/></g>
    <g id="browR"><path d="M118 80 Q134 74 150 82" stroke="#3e2a63" stroke-width="4" fill="none" stroke-linecap="round"/></g>
    <g id="eyes">
      <g id="eyeL" transform="translate(66 106)"><ellipse rx="13" ry="16" fill="#3e2a63"/><ellipse rx="9" ry="12" cy="2" fill="#c14f86"/><circle cx="-4" cy="-6" r="4" fill="#fff"/></g>
      <g id="eyeR" transform="translate(134 106)"><ellipse rx="13" ry="16" fill="#3e2a63"/><ellipse rx="9" ry="12" cy="2" fill="#c14f86"/><circle cx="-4" cy="-6" r="4" fill="#fff"/></g>
      <g id="happyEyes" opacity="0"><path d="M54 108 Q66 94 78 108" stroke="#3e2a63" stroke-width="5" fill="none" stroke-linecap="round"/><path d="M122 108 Q134 94 146 108" stroke="#3e2a63" stroke-width="5" fill="none" stroke-linecap="round"/></g>
    </g>
    <ellipse id="mouth" cx="100" cy="148" rx="9" ry="2" fill="#8a2c55"/>
  </svg>`;
  const $ = (id) => container.querySelector('#' + id);
  const eyes = [$('eyeL'), $('eyeR')], happy = $('happyEyes'), mouth = $('mouth');
  const browL = $('browL'), browR = $('browR'), cheeks = $('cheeks');
  let t = 0, nextBlink = 2 + Math.random() * 3, blink = 0, open = 0;
  let emo = { expression: 'neutral', intensity: 0.3, emotion: 'neutral' };
  let browTarget = 0, browCur = 0, lookY = 0;

  return {
    kind: '2d',
    setEmotion(e) {
      emo = e;
      browTarget = { angry: -10, sad: 8, surprised: -6, relaxed: 3 }[e.expression] ?? 0;
      cheeks.setAttribute('opacity', e.emotion === 'embarrassed' ? '1' : e.expression === 'happy' ? '.6' : '.35');
    },
    setMood() {},
    update(dt, { mouth: level = 0, speaking = false, thinking = false, arousal = 0.45 }) {
      t += dt;
      nextBlink -= dt;
      if (nextBlink <= 0) { blink = 0.14; nextBlink = (2 + Math.random() * 4) * (1.3 - arousal * 0.6); }
      blink = Math.max(0, blink - dt);
      const still = reducedMotion();
      lookY = lerp(lookY, thinking ? -4 : 0, Math.min(1, dt * 6));   // glance up while thinking
      const closed = blink > 0 ? 0.1 : 1;
      const useHappy = emo.expression === 'happy' && emo.intensity > 0.45 && !speaking;
      happy.setAttribute('opacity', useHappy ? '1' : '0');
      const sy = (emo.expression === 'surprised' ? 1.15 : 1) * closed;
      eyes.forEach((g, i) => {
        g.setAttribute('opacity', useHappy ? '0' : '1');
        const sway = still ? 0 : Math.sin(t * (1 + arousal)) * (0.5 + arousal);
        g.setAttribute('transform', `translate(${(i ? 134 : 66) + lookY * 0.4} ${106 + sway + lookY}) scale(1 ${sy})`);
      });
      browCur = lerp(browCur, browTarget, Math.min(1, dt * 8));
      browL.setAttribute('transform', `rotate(${browCur} 66 80)`);
      browR.setAttribute('transform', `rotate(${-browCur} 134 80)`);
      open = lerp(open, speaking ? level : 0, Math.min(1, dt * 18));
      const smile = emo.expression === 'happy' ? 4 : emo.expression === 'sad' ? 1 : 2;
      mouth.setAttribute('ry', (smile * 0.5 + open * 12).toFixed(2));
      mouth.setAttribute('rx', (9 + (emo.expression === 'happy' ? 3 : 0) - open * 2).toFixed(2));
    },
    dispose() { container.innerHTML = ''; },
  };
}

// ------------------------------------------------------------------ VRM
async function createVRMAvatar(container, url) {
  const THREE = await import('three');
  const { GLTFLoader } = await import('three/addons/loaders/GLTFLoader.js');
  const { VRMLoaderPlugin, VRMUtils } = await import('@pixiv/three-vrm');

  const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  renderer.outputColorSpace = THREE.SRGBColorSpace;
  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(26, 1, 0.1, 20);
  const key = new THREE.DirectionalLight(0xffffff, 2.2); key.position.set(1, 1.6, 1.4); scene.add(key);
  scene.add(new THREE.AmbientLight(0xd9ccff, 1.1));

  const loader = new GLTFLoader();
  loader.register((parser) => new VRMLoaderPlugin(parser));
  const gltf = await loader.loadAsync(url);
  const vrm = gltf.userData.vrm;
  if (!vrm) throw new Error('file is not a VRM model');
  VRMUtils.removeUnnecessaryVertices(gltf.scene);
  if (VRMUtils.combineSkeletons) VRMUtils.combineSkeletons(gltf.scene);
  VRMUtils.rotateVRM0(vrm);   // VRM 0.x faces -Z; normalise to VRM 1.0 orientation
  vrm.scene.traverse((o) => { o.frustumCulled = false; });
  scene.add(vrm.scene);
  container.innerHTML = '';
  container.appendChild(renderer.domElement);

  const bone = (n) => vrm.humanoid?.getNormalizedBoneNode(n);
  const head = bone('head'), neck = bone('neck'), spine = bone('spine'), chest = bone('chest');
  const lUpper = bone('leftUpperArm'), rUpper = bone('rightUpperArm');
  const lLower = bone('leftLowerArm'), rLower = bone('rightLowerArm');
  vrm.update(0);
  const hp = new THREE.Vector3(); head?.getWorldPosition(hp);
  const y = hp.y || 1.4;
  camera.position.set(0, y - 0.02, 1.15);
  camera.lookAt(0, y - 0.1, 0);

  const resize = () => {
    const w = container.clientWidth || 1, h = container.clientHeight || 1;
    renderer.setSize(w, h, false); camera.aspect = w / h; camera.updateProjectionMatrix();
  };
  const ro = new ResizeObserver(resize); ro.observe(container); resize();

  const em = vrm.expressionManager;
  const EXPR = ['happy', 'angry', 'sad', 'relaxed', 'surprised'];
  const target = Object.fromEntries(EXPR.map((k) => [k, 0]));
  const cur = { ...target, aa: 0, oh: 0, blink: 0 };
  let t = 0, nextBlink = 2, blinkT = 0;

  return {
    kind: 'vrm',
    setEmotion(e) {
      EXPR.forEach((k) => (target[k] = 0));
      if (e.expression && e.expression !== 'neutral' && e.expression in target) {
        target[e.expression] = Math.min(1, 0.35 + e.intensity * 0.6);
      }
    },
    setMood() {},
    update(dt, { mouth = 0, speaking = false, listening = false, thinking = false, arousal = 0.45 }) {
      t += dt;
      const amp = reducedMotion() ? 0 : 0.6 + arousal;
      // arms down from T-pose, gentle breathing and sway
      if (lUpper) lUpper.rotation.z = -1.15 + Math.sin(t * 1.1) * 0.015;
      if (rUpper) rUpper.rotation.z = 1.15 - Math.sin(t * 1.1) * 0.015;
      if (lLower) lLower.rotation.z = -0.12;
      if (rLower) rLower.rotation.z = 0.12;
      if (spine) spine.rotation.x = Math.sin(t * 1.2) * 0.015;
      if (chest) chest.rotation.x = Math.sin(t * 1.2 + 0.6) * 0.012;
      if (neck && amp === 0) { neck.rotation.set(0, 0, 0); } else if (neck) {
        neck.rotation.y = Math.sin(t * 0.45) * 0.06;
        neck.rotation.z = listening ? 0.06 : thinking ? -0.05 : Math.sin(t * 0.33) * 0.02 * amp;  // tilt: listening/thinking
        neck.rotation.x = thinking ? -0.08 : 0;   // glance up while thinking
      }
      if (head) head.rotation.x = speaking ? Math.sin(t * 7) * mouth * 0.05 : 0;
      // blinking
      nextBlink -= dt;
      if (nextBlink <= 0) { blinkT = 0.15; nextBlink = (2 + Math.random() * 4) * (1.3 - arousal * 0.6); }
      blinkT = Math.max(0, blinkT - dt);
      const blink = blinkT > 0 ? Math.sin((blinkT / 0.15) * Math.PI) : 0;
      // expressions, eased
      const k = Math.min(1, dt * 6);
      EXPR.forEach((n) => { cur[n] = lerp(cur[n], target[n], k); em?.setValue(n, cur[n]); });
      cur.aa = lerp(cur.aa, speaking ? Math.min(1, mouth * 1.4) : 0, Math.min(1, dt * 20));
      cur.oh = lerp(cur.oh, speaking ? Math.min(0.5, mouth * Math.abs(Math.sin(t * 9))) : 0, Math.min(1, dt * 12));
      em?.setValue('aa', cur.aa);
      em?.setValue('oh', cur.oh);
      em?.setValue('blink', cur.happy > 0.6 ? 0 : blink);
      vrm.update(dt);
      renderer.render(scene, camera);
    },
    dispose() { ro.disconnect(); renderer.dispose(); VRMUtils.deepDispose(vrm.scene); container.innerHTML = ''; },
  };
}
