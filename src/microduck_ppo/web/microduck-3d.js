import * as THREE from './vendor/three.module.js';
import { OrbitControls } from './vendor/OrbitControls.js';

const WORLD_LIMIT = 3.0;
function decode(data, dtype) {
  const bytes = Uint8Array.from(atob(data), (c) => c.charCodeAt(0));
  return new dtype(bytes.buffer);
}
function color(value) { return new THREE.Color(...value.slice(0, 3)); }
function makeLabel(text, tint) {
  const canvas = document.createElement('canvas');
  canvas.width = 420; canvas.height = 96;
  const ctx = canvas.getContext('2d');
  ctx.font = '600 34px Avenir Next, PingFang SC, sans-serif';
  const width = Math.min(390, Math.max(120, ctx.measureText(text).width + 36));
  ctx.fillStyle = 'rgba(20, 45, 58, .92)';
  ctx.beginPath(); ctx.roundRect((420 - width) / 2, 10, width, 72, 16); ctx.fill();
  ctx.strokeStyle = tint; ctx.lineWidth = 5; ctx.stroke();
  ctx.fillStyle = '#f8fbfc'; ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
  ctx.fillText(text, 210, 46);
  const sprite = new THREE.Sprite(new THREE.SpriteMaterial({ map: new THREE.CanvasTexture(canvas), transparent: true, depthTest: false }));
  sprite.scale.set(width / 420 * 0.78, 0.18, 1); sprite.renderOrder = 20;
  return sprite;
}

/** One DuckKing-style 3D scene used for both placement and live simulation. */
export function createMicroDuck3D(container, { onPlace = () => {}, onView = () => {} } = {}) {
  const scene = new THREE.Scene(); scene.background = new THREE.Color('#dbe6e8');
  const camera = new THREE.PerspectiveCamera(38, 1, 0.01, 30); camera.position.set(1.8, 1.45, 2.0);
  const renderer = new THREE.WebGLRenderer({ antialias: true });
  renderer.outputColorSpace = THREE.SRGBColorSpace; renderer.toneMapping = THREE.ACESFilmicToneMapping; renderer.toneMappingExposure = 1.1;
  renderer.shadowMap.enabled = true; renderer.shadowMap.type = THREE.PCFShadowMap;
  renderer.domElement.className = 'microduck-canvas'; renderer.domElement.setAttribute('aria-label', 'MicroDuck MuJoCo 三维仿真场景'); container.replaceChildren(renderer.domElement);
  const controls = new OrbitControls(camera, renderer.domElement);
  controls.target.set(0, 0.12, 0); controls.minDistance = 0.55; controls.maxDistance = 9; controls.maxPolarAngle = Math.PI / 2.02; controls.enableDamping = true; controls.update(); controls.saveState();
  function reportView() { const offset = camera.position.clone().sub(controls.target); const horizontal = Math.hypot(offset.x, offset.z); onView({ azimuth: (Math.atan2(offset.x, offset.z) * 180) / Math.PI + 180, elevation: (Math.atan2(offset.y, horizontal) * 180) / Math.PI, distance: offset.length() }); }
  controls.addEventListener('change', reportView);
  scene.add(new THREE.HemisphereLight(0xffffff, 0x71815a, 2.4));
  const sun = new THREE.DirectionalLight(0xfff6df, 3); sun.position.set(1, 2, 1); sun.castShadow = true; sun.shadow.mapSize.set(1024, 1024); scene.add(sun);
  const rim = new THREE.DirectionalLight(0xb8d9ff, 1.8); rim.position.set(-2, 1.5, -3); scene.add(rim);
  const floor = new THREE.Mesh(new THREE.PlaneGeometry(6, 6), new THREE.MeshStandardMaterial({ color: '#e9eff0', roughness: 0.95 })); floor.rotation.x = -Math.PI / 2; floor.receiveShadow = true; scene.add(floor);
  const grid = new THREE.GridHelper(6, 60, 0xc5d2d5, 0xd8e2e4); grid.position.y = 0.002; scene.add(grid);
  const world = new THREE.Group(); world.rotation.x = -Math.PI / 2; scene.add(world);
  const goal = new THREE.Mesh(new THREE.CylinderGeometry(0.13, 0.13, 0.008, 48), new THREE.MeshPhysicalMaterial({ color: '#176fc0', roughness: 0.24, metalness: 0.2, clearcoat: 0.35, transparent: true, opacity: 0.94 })); goal.position.y = 0.004; goal.userData.kind = 'goal'; scene.add(goal);
  const raycaster = new THREE.Raycaster(); const pointer = new THREE.Vector2();
  let placedGoal = { x: 1.4, y: 0 }, agents = [], model = null, modelStart = { x: 0, y: 0 }, live = false, dragging = null;
  function resize() { const { width, height } = container.getBoundingClientRect(); if (!width || !height) return; renderer.setPixelRatio(Math.min(devicePixelRatio, 2)); renderer.setSize(width, height); camera.aspect = width / height; camera.updateProjectionMatrix(); }
  new ResizeObserver(resize).observe(container); resize();
  function floorPoint(event) { const rect = renderer.domElement.getBoundingClientRect(); pointer.set(((event.clientX - rect.left) / rect.width) * 2 - 1, -((event.clientY - rect.top) / rect.height) * 2 + 1); raycaster.setFromCamera(pointer, camera); const hit = raycaster.intersectObject(floor)[0]; if (!hit) return null; return { x: Math.max(-WORLD_LIMIT, Math.min(WORLD_LIMIT, hit.point.x)), y: Math.max(-WORLD_LIMIT, Math.min(WORLD_LIMIT, -hit.point.z)) }; }
  function hitTarget(event) { const rect = renderer.domElement.getBoundingClientRect(); pointer.set(((event.clientX - rect.left) / rect.width) * 2 - 1, -((event.clientY - rect.top) / rect.height) * 2 + 1); raycaster.setFromCamera(pointer, camera); if (raycaster.intersectObject(goal, false).length) return { kind: 'goal' }; for (const agent of agents.slice().reverse()) if (raycaster.intersectObject(agent.group, true).length) return { kind: 'agent', id: agent.id }; return null; }
  function moveAgent(agent, point) { agent.start = { ...point }; const root = agent.root || modelStart; agent.group.position.set(point.x - root.x, point.y - root.y, 0); onPlace({ agentId: agent.id, start: { ...point }, goal: { ...placedGoal }, source: 'agent' }); }
  renderer.domElement.addEventListener('pointerdown', (event) => { const hit = hitTarget(event); if (!hit || live) return; dragging = hit; controls.enabled = false; renderer.domElement.setPointerCapture(event.pointerId); });
  renderer.domElement.addEventListener('pointermove', (event) => { if (!dragging) return; const point = floorPoint(event); if (!point) return; if (dragging.kind === 'goal') { placedGoal = point; goal.position.set(point.x, 0.004, -point.y); onPlace({ goal: { ...point }, source: 'goal' }); } else if (!live) { const agent = agents.find((item) => item.id === dragging.id); if (agent) moveAgent(agent, point); } });
  renderer.domElement.addEventListener('pointerup', () => { dragging = null; controls.enabled = true; }); renderer.domElement.addEventListener('pointercancel', () => { dragging = null; controls.enabled = true; });
  function setGoal(target) { placedGoal = { x: Number(target.x ?? target[0]), y: Number(target.y ?? target[1]) }; goal.position.set(placedGoal.x, 0.004, -placedGoal.y); }
  function createAgent(spec, index) {
    const group = new THREE.Group(); group.userData.agentId = spec.id; const tint = ['#c45c20', '#267659', '#7d55a8', '#b13d61', '#947020'][index % 5];
    for (const geom of model.geoms || []) { const geometry = model.meshes.get(Number(geom.mesh)); const c = color(geom.rgba); const luminance = 0.2126 * c.r + 0.7152 * c.g + 0.0722 * c.b; const mesh = new THREE.Mesh(geometry, new THREE.MeshPhysicalMaterial({ color: c, roughness: luminance < 0.30 ? 0.2 : 0.3, metalness: luminance < 0.30 ? 0.8 : 0.36, clearcoat: luminance < 0.30 ? 0.32 : 0.52, clearcoatRoughness: 0.15 })); mesh.castShadow = true; mesh.receiveShadow = true; group.add(mesh); }
    const label = makeLabel(spec.label, tint); group.add(label); world.add(group); return { ...spec, group, label, meshes: group.children.slice(0, model.geoms.length).map((mesh, i) => [model.geoms[i].id, mesh]), root: { ...modelStart } };
  }
  function setAgents(specs = []) { const incoming = specs.map((item, index) => ({ id: String(item.id || `agent-${index + 1}`), stage: item.stage || '', label: item.label || item.stage || `策略 ${index + 1}`, start: { x: Number(item.start?.x ?? item.start?.[0] ?? (-1 + index * 0.4)), y: Number(item.start?.y ?? item.start?.[1] ?? 0) }, yaw: Number(item.yaw || 0) })); if (!model) { agents = incoming; return; } const old = new Map(agents.map((agent) => [agent.id, agent])); for (const agent of agents) if (!incoming.some((item) => item.id === agent.id)) world.remove(agent.group); agents = incoming.map((item, index) => { const previous = old.get(item.id); const current = previous || createAgent(item, index); Object.assign(current, item); current.group.visible = true; current.root = current.root || { ...modelStart }; current.group.position.set(item.start.x - current.root.x, item.start.y - current.root.y, 0); current.label.position.set(current.root.x, current.root.y, current.root.z + 0.32); return current; }); }
  function loadModel(payload) { const specs = agents.map(({ id, stage, label, start, yaw }) => ({ id, stage, label, start, yaw })); world.clear(); agents = []; const meshes = new Map(); for (const [id, item] of Object.entries(payload.meshes || {})) { const geometry = new THREE.BufferGeometry(); geometry.setAttribute('position', new THREE.BufferAttribute(decode(item.vertices, Float32Array), 3)); geometry.setIndex(new THREE.BufferAttribute(decode(item.faces, Uint32Array), 1)); geometry.computeVertexNormals(); meshes.set(Number(id), geometry); } model = { ...payload, meshes }; if (payload.initial_state) modelStart = { x: Number(payload.initial_state.root?.[0] || 0), y: Number(payload.initial_state.root?.[1] || 0) }; setAgents(specs.length ? specs : [{ id: 'agent-1', stage: '', label: '初始策略', start: { x: -1.4, y: 0 }, yaw: 0 }]); if (payload.initial_state) updateState(payload.initial_state, { live: false }); }
  function updateState(state, { live: nextLive = true } = {}) { live = nextLive; let states = Array.isArray(state?.agents) ? state.agents : [state]; if (!nextLive && states.length === 1 && agents.length > 1) states = agents.map(() => states[0]); states.forEach((agentState, index) => { const agent = agents[index]; if (!agent) return; agent.root = { x: agentState.root?.[0] || 0, y: agentState.root?.[1] || 0, z: agentState.root?.[2] || 0 }; agent.group.position.set(nextLive ? 0 : agent.start.x - agent.root.x, nextLive ? 0 : agent.start.y - agent.root.y, 0); agent.label.position.set(agent.root.x, agent.root.y, agent.root.z + 0.32); for (const geom of agentState.geoms || []) { const mesh = agent.meshes.find(([id]) => id === geom.id)?.[1]; if (!mesh) continue; mesh.position.set(...geom.position); mesh.quaternion.setFromRotationMatrix(new THREE.Matrix4().set(geom.rotation[0][0], geom.rotation[0][1], geom.rotation[0][2], 0, geom.rotation[1][0], geom.rotation[1][1], geom.rotation[1][2], 0, geom.rotation[2][0], geom.rotation[2][1], geom.rotation[2][2], 0, 0, 0, 0, 1)); } }); goal.visible = true; }
  function setScene(start, target) { setGoal({ x: target[0], y: target[1] }); if (!agents.length) agents = [{ id: 'agent-1', stage: '', label: '初始策略', start: { x: start[0], y: start[1] } }]; else agents[0].start = { x: start[0], y: start[1] }; setAgents(agents); const center = new THREE.Vector3((start[0] + target[0]) / 2, 0.11, -(start[1] + target[1]) / 2); const span = Math.hypot(target[0] - start[0], target[1] - start[1]); camera.position.copy(center).add(new THREE.Vector3(1.75, 1.35, 1.9).normalize().multiplyScalar(Math.max(1.8, Math.min(7, 1.4 + 1.25 * span)))); controls.target.copy(center); controls.update(); controls.saveState(); reportView(); }
  function setPlacement(start, target) { setScene(start, target); }
  function resetCamera() { controls.reset(); reportView(); }
  function animate() { controls.update(); renderer.render(scene, camera); requestAnimationFrame(animate); } animate(); setGoal(placedGoal);
  return { element: renderer.domElement, setScene, setPlacement, setAgents, setGoal, loadModel, updateState, setPlacementMode: () => { live = false; }, setRunningMode: () => { live = true; }, resetCamera, getPlacement: () => ({ agents: agents.map((agent) => ({ id: agent.id, start: agent.start })), goal: placedGoal }), dispose: () => { controls.dispose(); renderer.dispose(); } };
}
