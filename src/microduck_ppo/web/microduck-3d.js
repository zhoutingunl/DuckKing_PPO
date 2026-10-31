import * as THREE from './vendor/three.module.js';
import { OrbitControls } from './vendor/OrbitControls.js';

const WORLD_LIMIT = 1.1;

function decode(data, dtype) {
  const bytes = Uint8Array.from(atob(data), (c) => c.charCodeAt(0));
  return new dtype(bytes.buffer);
}

function color(value) {
  return new THREE.Color(...value.slice(0, 3));
}

/**
 * DuckKing-style Three.js scene editor. It is deliberately a visual layer:
 * placement is sent to the MuJoCo session, while poses arrive as state frames.
 */
export function createMicroDuck3D(container, { onPlace = () => {}, onView = () => {} } = {}) {
  const scene = new THREE.Scene();
  scene.background = new THREE.Color('#dbe6e8');
  const camera = new THREE.PerspectiveCamera(38, 1, 0.01, 30);
  camera.position.set(1.8, 1.45, 2.0);
  const renderer = new THREE.WebGLRenderer({ antialias: true });
  renderer.outputColorSpace = THREE.SRGBColorSpace;
  renderer.toneMapping = THREE.ACESFilmicToneMapping;
  renderer.toneMappingExposure = 1.1;
  renderer.shadowMap.enabled = true;
  renderer.shadowMap.type = THREE.PCFShadowMap;
  renderer.domElement.setAttribute('aria-label', 'MicroDuck MuJoCo 三维仿真场景');
  renderer.domElement.className = 'microduck-canvas';
  container.replaceChildren(renderer.domElement);
  const controls = new OrbitControls(camera, renderer.domElement);
  controls.target.set(0, 0.12, 0);
  controls.minDistance = 0.55;
  controls.maxDistance = 5;
  controls.maxPolarAngle = Math.PI / 2.02;
  controls.enableDamping = true;
  controls.update();
  controls.saveState();
  function reportView() {
    const offset = camera.position.clone().sub(controls.target);
    const horizontal = Math.hypot(offset.x, offset.z);
    onView({
      azimuth: (Math.atan2(offset.x, offset.z) * 180) / Math.PI + 180,
      elevation: (Math.atan2(offset.y, horizontal) * 180) / Math.PI,
      distance: offset.length(),
    });
  }
  controls.addEventListener('change', reportView);

  scene.add(new THREE.HemisphereLight(0xffffff, 0x71815a, 2.4));
  const sun = new THREE.DirectionalLight(0xfff6df, 3);
  sun.position.set(1, 2, 1);
  sun.castShadow = true;
  sun.shadow.mapSize.set(1024, 1024);
  scene.add(sun);
  const floor = new THREE.Mesh(
    new THREE.PlaneGeometry(5, 5),
    new THREE.MeshStandardMaterial({ color: '#e9eff0', roughness: 0.95 }),
  );
  floor.rotation.x = -Math.PI / 2;
  floor.receiveShadow = true;
  floor.userData.kind = 'floor';
  scene.add(floor);
  const grid = new THREE.GridHelper(2.4, 24, 0xc5d2d5, 0xd8e2e4);
  grid.position.y = 0.002;
  scene.add(grid);
  const world = new THREE.Group();
  // Meshes keep MuJoCo coordinates; this group maps X/Y floor and Z-up to
  // Three.js X/Z floor and Y-up, exactly as DuckKing's physics viewer does.
  world.rotation.x = -Math.PI / 2;
  scene.add(world);
  const goal = new THREE.Mesh(
    new THREE.CylinderGeometry(0.1, 0.1, 0.008, 48),
    new THREE.MeshStandardMaterial({ color: '#176fc0', roughness: 0.7, transparent: true, opacity: 0.9 }),
  );
  goal.position.y = 0.004;
  goal.castShadow = true;
  goal.userData.kind = 'goal';
  scene.add(goal);
  const startMarker = new THREE.Mesh(
    new THREE.RingGeometry(0.105, 0.115, 48),
    new THREE.MeshBasicMaterial({ color: '#c45c20', side: THREE.DoubleSide }),
  );
  startMarker.rotation.x = -Math.PI / 2;
  startMarker.position.y = 0.006;
  scene.add(startMarker);
  const raycaster = new THREE.Raycaster();
  const pointer = new THREE.Vector2();
  let placedStart = { x: 0, y: 0 };
  let placedGoal = { x: 0.55, y: 0 };
  let modelStart = { x: 0, y: 0 };
  let meshEntries = new Map();
  let animation = 0;
  let active = false;
  let dragging = null;

  function resize() {
    const { width, height } = container.getBoundingClientRect();
    if (!width || !height) return;
    renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
    renderer.setSize(width, height);
    camera.aspect = width / height;
    camera.updateProjectionMatrix();
  }
  new ResizeObserver(resize).observe(container);
  resize();

  function placeFromEvent(event, kind) {
    const rect = renderer.domElement.getBoundingClientRect();
    pointer.set(((event.clientX - rect.left) / rect.width) * 2 - 1, -((event.clientY - rect.top) / rect.height) * 2 + 1);
    raycaster.setFromCamera(pointer, camera);
    const hit = raycaster.intersectObject(floor)[0];
    if (!hit) return;
    const x = Math.max(-WORLD_LIMIT, Math.min(WORLD_LIMIT, hit.point.x));
    const y = Math.max(-WORLD_LIMIT, Math.min(WORLD_LIMIT, -hit.point.z));
    if (kind === 'goal') placedGoal = { x, y };
    else placedStart = { x, y };
    updateMarkers();
    onPlace({ start: placedStart, goal: placedGoal, source: kind });
  }
  function hitMarker(event) {
    const rect = renderer.domElement.getBoundingClientRect();
    pointer.set(((event.clientX - rect.left) / rect.width) * 2 - 1, -((event.clientY - rect.top) / rect.height) * 2 + 1);
    raycaster.setFromCamera(pointer, camera);
    const hits = raycaster.intersectObjects([goal, startMarker], false);
    if (hits[0]?.object.userData.kind === 'goal') return 'goal';
    if (hits[0]) return 'duck';
    const duckHits = raycaster.intersectObjects([...meshEntries.values()], false);
    return duckHits.length ? 'duck' : null;
  }
  renderer.domElement.addEventListener('pointerdown', (event) => {
    const marker = hitMarker(event);
    if (marker) {
      dragging = marker;
      controls.enabled = false;
      renderer.domElement.setPointerCapture(event.pointerId);
    }
  });
  renderer.domElement.addEventListener('pointermove', (event) => {
    if (dragging) placeFromEvent(event, dragging);
  });
  renderer.domElement.addEventListener('pointerup', () => {
    dragging = null;
    controls.enabled = true;
  });
  renderer.domElement.addEventListener('dblclick', (event) => placeFromEvent(event, 'goal'));

  function updateMarkers() {
    goal.position.set(placedGoal.x, 0.004, -placedGoal.y);
    startMarker.position.set(placedStart.x, 0.006, -placedStart.y);
    if (!active) {
      world.position.set(placedStart.x - modelStart.x, 0, -(placedStart.y - modelStart.y));
    }
  }
  function setScene(start, target) {
    setPlacement(start, target);
    camera.position.set(1.75, 1.35, 1.9);
    controls.target.set((placedStart.x + placedGoal.x) / 2, 0.11, -(placedStart.y + placedGoal.y) / 2);
    controls.update();
    controls.saveState();
    reportView();
  }
  function setPlacement(start, target) {
    placedStart = { x: Number(start[0]), y: Number(start[1]) };
    placedGoal = { x: Number(target[0]), y: Number(target[1]) };
    updateMarkers();
  }
  function loadModel(model) {
    world.clear();
    meshEntries = new Map();
    const meshes = new Map();
    for (const [id, payload] of Object.entries(model.meshes || {})) {
      const geometry = new THREE.BufferGeometry();
      geometry.setAttribute('position', new THREE.BufferAttribute(decode(payload.vertices, Float32Array), 3));
      geometry.setIndex(new THREE.BufferAttribute(decode(payload.faces, Uint32Array), 1));
      geometry.computeVertexNormals();
      meshes.set(Number(id), geometry);
    }
    for (const geom of model.geoms || []) {
      const surface = geom.material || {};
      const mesh = new THREE.Mesh(
        meshes.get(Number(geom.mesh)),
        new THREE.MeshStandardMaterial({
          color: color(geom.rgba),
          roughness: 0.66,
          metalness: 0.08,
        }),
      );
      mesh.castShadow = true;
      mesh.receiveShadow = true;
      world.add(mesh);
      meshEntries.set(geom.id, mesh);
    }
    if (model.initial_state) {
      modelStart = {
        x: Number(model.initial_state.start?.[0] || 0),
        y: Number(model.initial_state.start?.[1] || 0),
      };
      updateState(model.initial_state, { live: false });
    }
  }
  function updateState(state, { live = true } = {}) {
    active = live;
    world.position.set(0, 0, 0);
    if (!live) {
      modelStart = {
        x: Number(state.start?.[0] || 0),
        y: Number(state.start?.[1] || 0),
      };
      world.position.set(placedStart.x - modelStart.x, 0, -(placedStart.y - modelStart.y));
    }
    for (const geom of state.geoms || []) {
      const mesh = meshEntries.get(geom.id);
      if (!mesh) continue;
      mesh.position.set(geom.position[0], geom.position[1], geom.position[2]);
      const matrix = new THREE.Matrix4().set(
        geom.rotation[0][0], geom.rotation[0][1], geom.rotation[0][2], 0,
        geom.rotation[1][0], geom.rotation[1][1], geom.rotation[1][2], 0,
        geom.rotation[2][0], geom.rotation[2][1], geom.rotation[2][2], 0,
        0, 0, 0, 1,
      );
      mesh.quaternion.setFromRotationMatrix(matrix);
    }
    goal.visible = true;
    startMarker.visible = !active;
  }
  function resetCamera() {
    controls.reset();
    reportView();
  }
  function animate() {
    controls.update();
    renderer.render(scene, camera);
    animation = requestAnimationFrame(animate);
  }
  animate();
  updateMarkers();
  return {
    element: renderer.domElement,
    setScene,
    setPlacement,
    loadModel,
    updateState,
    resetCamera,
    getPlacement: () => ({ start: placedStart, goal: placedGoal }),
    dispose: () => { cancelAnimationFrame(animation); controls.dispose(); renderer.dispose(); },
  };
}
