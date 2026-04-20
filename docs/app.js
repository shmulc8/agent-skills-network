import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

const PALETTE_HEX = [
  '#ff8aab', '#ffae70', '#ffe07a', '#8bf0a6',
  '#7ee1f2', '#85a6ff', '#c59cff', '#ff8fd0',
  '#ff7a7a', '#aeea5a',
];
const PALETTE = PALETTE_HEX.map(h => new THREE.Color(h));

const loadingEl = document.getElementById('loading');
const container = document.getElementById('stage');
const tipEl = document.getElementById('tip');
const tipTitle = tipEl.querySelector('.t-title');
const tipTeam  = tipEl.querySelector('.t-team');
const tipPrev  = tipEl.querySelector('.t-prev');
const info = document.getElementById('info');
const iTitle = document.getElementById('i-title');
const iMeta  = document.getElementById('i-meta');
const iBody  = document.getElementById('i-body');
const clustersEl = document.getElementById('clusters');
const legendTitleEl = document.getElementById('legendTitle');
const brandSubEl = document.getElementById('brandSub');
const hint = document.getElementById('hint');

let points, clusters, teams;
let teamById = new Map();

init();

async function init() {
  const res = await fetch('./data.json');
  const DATA = await res.json();
  points = DATA.points;
  clusters = DATA.clusters;
  teams = DATA.teams || [];
  teamById = new Map(teams.map(t => [t.id, t]));
  if (brandSubEl) brandSubEl.textContent = `${points.length} skills · MiniLM · UMAP 3D`;
  loadingEl.classList.add('hidden');
  setTimeout(() => loadingEl.remove(), 600);
  build();
}

function build() {
  const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  renderer.setSize(window.innerWidth, window.innerHeight);
  renderer.setClearColor(0x05060a, 1);
  container.appendChild(renderer.domElement);

  const scene = new THREE.Scene();
  scene.fog = new THREE.FogExp2(0x05060a, 0.18);

  const camera = new THREE.PerspectiveCamera(55, window.innerWidth / window.innerHeight, 0.01, 100);
  camera.position.set(1.1, 0.55, 1.1);

  const controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  controls.dampingFactor = 0.08;
  controls.rotateSpeed = 0.6;
  controls.zoomSpeed = 0.8;
  controls.panSpeed = 0.5;
  controls.minDistance = 0.15;
  controls.maxDistance = 8;

  // Starfield
  {
    const starGeo = new THREE.BufferGeometry();
    const S = 800;
    const pos = new Float32Array(S * 3);
    for (let i = 0; i < S; i++) {
      const r = 30 + Math.random() * 15;
      const theta = Math.random() * Math.PI * 2;
      const phi = Math.acos(2 * Math.random() - 1);
      pos[i*3]   = r * Math.sin(phi) * Math.cos(theta);
      pos[i*3+1] = r * Math.sin(phi) * Math.sin(theta);
      pos[i*3+2] = r * Math.cos(phi);
    }
    starGeo.setAttribute('position', new THREE.BufferAttribute(pos, 3));
    const starMat = new THREE.PointsMaterial({
      color: 0xaab0d0, size: 0.06, sizeAttenuation: true,
      transparent: true, opacity: 0.45, depthWrite: false,
    });
    scene.add(new THREE.Points(starGeo, starMat));
  }

  const SPRITE = makeSpriteTexture();
  const N = points.length;

  const geometry = new THREE.BufferGeometry();
  const positions = new Float32Array(N * 3);
  const colors    = new Float32Array(N * 3);
  const sizes     = new Float32Array(N);
  const alphas    = new Float32Array(N);

  // Team colors come from data.json (stable per-team palette) converted to THREE.Color
  const teamColor = new Map();
  for (const t of teams) teamColor.set(t.id, new THREE.Color(t.color));

  let colorMode = 'cluster'; // 'cluster' | 'team'

  function colorForPoint(p) {
    if (colorMode === 'team') {
      return teamColor.get(p.team) || PALETTE[0];
    }
    return PALETTE[p.cluster % PALETTE.length];
  }
  function colorHexForPoint(p) {
    if (colorMode === 'team') {
      return (teamById.get(p.team) || {}).color || PALETTE_HEX[0];
    }
    return PALETTE_HEX[p.cluster % PALETTE_HEX.length];
  }

  const SCALE = 1.6;
  for (let i = 0; i < N; i++) {
    const p = points[i];
    positions[i*3]   = p.x * SCALE;
    positions[i*3+1] = p.y * SCALE;
    positions[i*3+2] = p.z * SCALE;
    const col = colorForPoint(p);
    colors[i*3]   = col.r;
    colors[i*3+1] = col.g;
    colors[i*3+2] = col.b;
    sizes[i]  = 0.08;
    alphas[i] = 1.0;
  }
  geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
  geometry.setAttribute('color',    new THREE.BufferAttribute(colors, 3));
  geometry.setAttribute('aSize',    new THREE.BufferAttribute(sizes, 1));
  geometry.setAttribute('aAlpha',   new THREE.BufferAttribute(alphas, 1));

  // Soft glowing core: bright center fading smoothly via the sprite gradient,
  // additively blended so the cloud reads as a luminous nebula — not flat dots.
  // Per-fragment alpha is tuned low so 907 points don't pile up into white.
  const pointMaterial = new THREE.ShaderMaterial({
    uniforms: {
      uMap:        { value: SPRITE },
      uPixelRatio: { value: renderer.getPixelRatio() },
    },
    vertexShader: `
      attribute float aSize;
      attribute float aAlpha;
      varying vec3 vColor;
      varying float vAlpha;
      uniform float uPixelRatio;
      void main() {
        vColor = color;
        vAlpha = aAlpha;
        vec4 mv = modelViewMatrix * vec4(position, 1.0);
        gl_PointSize = aSize * 380.0 * uPixelRatio / -mv.z;
        gl_Position = projectionMatrix * mv;
      }
    `,
    fragmentShader: `
      uniform sampler2D uMap;
      varying vec3 vColor;
      varying float vAlpha;
      void main() {
        vec4 tex = texture2D(uMap, gl_PointCoord);
        if (tex.a < 0.01) discard;
        vec3 c = vColor * (0.55 + 0.45 * tex.a);
        gl_FragColor = vec4(c, tex.a * vAlpha * 0.38);
      }
    `,
    transparent: true,
    depthWrite: false,
    blending: THREE.AdditiveBlending,
    vertexColors: true,
  });

  const cloud = new THREE.Points(geometry, pointMaterial);
  scene.add(cloud);

  // Very faint wide halo around each point — gives distance cues / atmosphere
  // without adding much luminance per point.
  const glowGeometry = new THREE.BufferGeometry();
  glowGeometry.setAttribute('position', geometry.attributes.position);
  glowGeometry.setAttribute('color',    geometry.attributes.color);
  const glowSizeArr = new Float32Array(N);
  for (let i = 0; i < N; i++) glowSizeArr[i] = 0.025;
  glowGeometry.setAttribute('aSize', new THREE.BufferAttribute(glowSizeArr, 1));

  const glowMaterial = new THREE.ShaderMaterial({
    uniforms: {
      uMap:        { value: SPRITE },
      uPixelRatio: { value: renderer.getPixelRatio() },
    },
    vertexShader: `
      attribute float aSize;
      varying vec3 vColor;
      uniform float uPixelRatio;
      void main() {
        vColor = color;
        vec4 mv = modelViewMatrix * vec4(position, 1.0);
        gl_PointSize = aSize * 1400.0 * uPixelRatio / -mv.z;
        gl_Position = projectionMatrix * mv;
      }
    `,
    fragmentShader: `
      uniform sampler2D uMap;
      varying vec3 vColor;
      void main() {
        float a = texture2D(uMap, gl_PointCoord).a;
        if (a < 0.01) discard;
        gl_FragColor = vec4(vColor, a * 0.04);
      }
    `,
    transparent: true,
    depthWrite: false,
    blending: THREE.AdditiveBlending,
    vertexColors: true,
  });
  const glowCloud = new THREE.Points(glowGeometry, glowMaterial);
  scene.add(glowCloud);


  const ringGeo = new THREE.RingGeometry(0.018, 0.022, 32);
  const ringMat = new THREE.MeshBasicMaterial({
    color: 0xffffff, transparent: true, opacity: 0.7, side: THREE.DoubleSide,
  });
  const selRing = new THREE.Mesh(ringGeo, ringMat);
  selRing.visible = false;
  scene.add(selRing);

  const neighbors = points.map((p, i) => {
    const arr = [];
    for (let j = 0; j < N; j++) {
      if (i === j) continue;
      const dx = points[j].x - p.x, dy = points[j].y - p.y, dz = points[j].z - p.z;
      arr.push([j, dx*dx + dy*dy + dz*dz]);
    }
    arr.sort((a, b) => a[1] - b[1]);
    return arr.slice(0, 6).map(d => d[0]);
  });

  const lineGeo = new THREE.BufferGeometry();
  const lineMat = new THREE.LineBasicMaterial({ color: 0xffffff, transparent: true, opacity: 0.45 });
  const lineObj = new THREE.LineSegments(lineGeo, lineMat);
  lineObj.visible = false;
  scene.add(lineObj);

  // Persistent neighbor web (thin & faint)
  const staticLineGeo = new THREE.BufferGeometry();
  const staticLineMat = new THREE.LineBasicMaterial({
    color: 0xc4cae6, transparent: true, opacity: 0.15, depthWrite: false,
  });
  const staticLineObj = new THREE.LineSegments(staticLineGeo, staticLineMat);
  scene.add(staticLineObj);
  // Dedup pairs (A,B) == (B,A) and use only top-3 per node for clarity
  const webEdges = [];
  {
    const seen = new Set();
    const K = 5;
    for (let i = 0; i < N; i++) {
      for (let k = 0; k < Math.min(K, neighbors[i].length); k++) {
        const j = neighbors[i][k];
        const key = i < j ? `${i}-${j}` : `${j}-${i}`;
        if (seen.has(key)) continue;
        seen.add(key);
        webEdges.push([i, j]);
      }
    }
  }

  const raycaster = new THREE.Raycaster();
  raycaster.params.Points.threshold = 0.07;
  const mouse = new THREE.Vector2();

  const activeClusters = new Set(clusters.map(c => c.id));
  const activeTeams = new Set(teams.map(t => t.id));
  let searchQuery = '';
  let hoverId = -1;
  let selectedId = -1;

  function matchesSearch(p) {
    if (!searchQuery) return true;
    const q = searchQuery.toLowerCase();
    return p.title.toLowerCase().includes(q)
        || (p.description || '').toLowerCase().includes(q)
        || (p.team || '').toLowerCase().includes(q);
  }
  function isVisible(p) {
    const groupOk = colorMode === 'team' ? activeTeams.has(p.team) : activeClusters.has(p.cluster);
    return groupOk && matchesSearch(p);
  }

  function refreshColors() {
    const colorAttr = geometry.attributes.color;
    for (let i = 0; i < N; i++) {
      const col = colorForPoint(points[i]);
      colorAttr.array[i*3]   = col.r;
      colorAttr.array[i*3+1] = col.g;
      colorAttr.array[i*3+2] = col.b;
    }
    colorAttr.needsUpdate = true;
  }

  function updateAlphas() {
    const alphaAttr = geometry.attributes.aAlpha;
    const sizeAttr  = geometry.attributes.aSize;
    const glowSizeAttr = glowGeometry.attributes.aSize;
    for (let i = 0; i < N; i++) {
      const vis = isVisible(points[i]);
      const emph = i === hoverId || i === selectedId;
      alphaAttr.array[i] = vis ? 1.0 : 0.05;
      sizeAttr.array[i]  = vis ? (emph ? 0.14 : 0.09) : 0.03;
      glowSizeAttr.array[i] = vis ? 0.08 : 0.0;
    }
    alphaAttr.needsUpdate = true;
    sizeAttr.needsUpdate = true;
    glowSizeAttr.needsUpdate = true;
    rebuildStaticLines();
  }

  function rebuildStaticLines() {
    const pos = [];
    for (const [i, j] of webEdges) {
      if (!isVisible(points[i]) || !isVisible(points[j])) continue;
      const a = points[i], b = points[j];
      pos.push(a.x * SCALE, a.y * SCALE, a.z * SCALE);
      pos.push(b.x * SCALE, b.y * SCALE, b.z * SCALE);
    }
    staticLineGeo.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3));
    staticLineGeo.computeBoundingSphere();
  }

  const raw = renderer.domElement;

  raw.addEventListener('pointermove', (e) => {
    const rect = raw.getBoundingClientRect();
    mouse.x = ((e.clientX - rect.left) / rect.width) * 2 - 1;
    mouse.y = -((e.clientY - rect.top) / rect.height) * 2 + 1;

    raycaster.setFromCamera(mouse, camera);
    const hits = raycaster.intersectObject(cloud);
    let chosen = -1;
    for (const h of hits) {
      const idx = h.index;
      if (isVisible(points[idx])) { chosen = idx; break; }
    }

    if (chosen !== hoverId) {
      hoverId = chosen;
      updateAlphas();
      if (chosen >= 0) {
        showTip(points[chosen], e.clientX, e.clientY);
        drawNeighborLines(chosen);
        lineObj.visible = true;
      } else {
        hideTip();
        lineObj.visible = false;
      }
    } else if (chosen >= 0) {
      positionTip(e.clientX, e.clientY);
    }
  });

  let downX = 0, downY = 0;
  raw.addEventListener('pointerdown', (e) => { downX = e.clientX; downY = e.clientY; });
  raw.addEventListener('pointerup', (e) => {
    const moved = Math.abs(e.clientX - downX) + Math.abs(e.clientY - downY);
    if (moved > 4) return;
    if (hoverId >= 0) {
      selectedId = hoverId;
      openInfo(points[selectedId]);
      positionSelRing();
      hint.classList.add('hidden');
      updateAlphas();
    }
  });

  function drawNeighborLines(i) {
    const pos = [];
    const src = points[i];
    for (const j of neighbors[i]) {
      if (!isVisible(points[j])) continue;
      pos.push(src.x * SCALE, src.y * SCALE, src.z * SCALE);
      pos.push(points[j].x * SCALE, points[j].y * SCALE, points[j].z * SCALE);
    }
    lineGeo.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3));
    lineMat.color.copy(colorForPoint(src));
    lineMat.opacity = 0.35;
  }

  function positionSelRing() {
    if (selectedId < 0) { selRing.visible = false; return; }
    const p = points[selectedId];
    selRing.position.set(p.x * SCALE, p.y * SCALE, p.z * SCALE);
    selRing.lookAt(camera.position);
    selRing.visible = true;
  }

  function openInfo(p) {
    iTitle.textContent = p.title;
    iMeta.innerHTML = '';
    // Team tag
    const teamTag = document.createElement('span');
    teamTag.className = 'tag';
    const teamCol = (teamById.get(p.team) || {}).color || '#c4cae6';
    teamTag.style.color = teamCol;
    teamTag.style.borderColor = teamCol + '55';
    teamTag.textContent = p.team;
    iMeta.appendChild(teamTag);
    // Cluster tag
    const clusterTag = document.createElement('span');
    clusterTag.className = 'tag';
    const clusterCol = PALETTE_HEX[p.cluster % PALETTE_HEX.length];
    clusterTag.style.color = clusterCol;
    clusterTag.style.borderColor = clusterCol + '44';
    clusterTag.textContent = clusters[p.cluster].label;
    iMeta.appendChild(clusterTag);
    // Link tag
    if (p.url) {
      const linkTag = document.createElement('a');
      linkTag.className = 'tag link';
      linkTag.href = p.url;
      linkTag.target = '_blank';
      linkTag.rel = 'noopener noreferrer';
      linkTag.textContent = 'open ↗';
      iMeta.appendChild(linkTag);
    }
    iBody.textContent = p.description || '';
    info.classList.add('open');
  }
  const legendEl = document.getElementById('legend');
  const backdropEl = document.getElementById('backdrop');
  const legendToggle = document.getElementById('legendToggle');
  function toggleLegend(open) {
    const willOpen = open ?? !legendEl.classList.contains('open');
    legendEl.classList.toggle('open', willOpen);
    backdropEl.classList.toggle('open', willOpen);
  }
  legendToggle.addEventListener('click', () => toggleLegend());
  backdropEl.addEventListener('click', () => toggleLegend(false));

  document.getElementById('close').addEventListener('click', () => {
    info.classList.remove('open');
    selectedId = -1;
    selRing.visible = false;
    updateAlphas();
  });

  // Help modal
  const helpModal = document.getElementById('helpModal');
  const helpBtn = document.getElementById('helpBtn');
  const helpClose = document.getElementById('helpClose');
  function toggleHelp(open) {
    const willOpen = open ?? !helpModal.classList.contains('open');
    helpModal.classList.toggle('open', willOpen);
  }
  helpBtn.addEventListener('click', () => toggleHelp(true));
  helpClose.addEventListener('click', () => toggleHelp(false));
  helpModal.addEventListener('click', (e) => { if (e.target === helpModal) toggleHelp(false); });

  function renderLegend() {
    clustersEl.innerHTML = '';
    if (colorMode === 'team') {
      legendTitleEl.textContent = `Teams (${teams.length})`;
      teams.forEach(t => {
        const row = document.createElement('div');
        row.className = 'cluster';
        row.dataset.team = t.id;
        row.innerHTML = `
          <div class="swatch" style="background:${t.color}"></div>
          <div class="label">${escapeHtml(t.id)}<span class="cnt">${t.count}</span></div>
        `;
        row.addEventListener('click', (e) => {
          const id = t.id;
          if (e.shiftKey || e.metaKey) {
            if (activeTeams.has(id)) activeTeams.delete(id);
            else activeTeams.add(id);
          } else {
            if (activeTeams.size === 1 && activeTeams.has(id)) {
              teams.forEach(tt => activeTeams.add(tt.id));
            } else {
              activeTeams.clear();
              activeTeams.add(id);
            }
          }
          refreshLegendDim();
          updateAlphas();
          if (window.innerWidth <= 720) toggleLegend(false);
        });
        clustersEl.appendChild(row);
      });
    } else {
      legendTitleEl.textContent = 'Topic clusters';
      clusters.forEach(c => {
        const row = document.createElement('div');
        row.className = 'cluster';
        row.dataset.id = c.id;
        const color = PALETTE_HEX[c.id % PALETTE_HEX.length];
        row.innerHTML = `
          <div class="swatch" style="background:${color}"></div>
          <div class="label">${escapeHtml(c.label)}<span class="cnt">${c.count}</span></div>
        `;
        row.addEventListener('click', (e) => {
          const id = c.id;
          if (e.shiftKey || e.metaKey) {
            if (activeClusters.has(id)) activeClusters.delete(id);
            else activeClusters.add(id);
          } else {
            if (activeClusters.size === 1 && activeClusters.has(id)) {
              clusters.forEach(cc => activeClusters.add(cc.id));
            } else {
              activeClusters.clear();
              activeClusters.add(id);
            }
          }
          refreshLegendDim();
          updateAlphas();
          if (window.innerWidth <= 720) toggleLegend(false);
        });
        clustersEl.appendChild(row);
      });
    }
    refreshLegendDim();
  }
  function refreshLegendDim() {
    [...clustersEl.children].forEach(row => {
      if (colorMode === 'team') {
        row.classList.toggle('dim', !activeTeams.has(row.dataset.team));
      } else {
        const id = parseInt(row.dataset.id, 10);
        row.classList.toggle('dim', !activeClusters.has(id));
      }
    });
  }
  renderLegend();

  document.querySelectorAll('input[name="colorMode"]').forEach(el => {
    el.addEventListener('change', (e) => {
      colorMode = e.target.value;
      refreshColors();
      renderLegend();
      updateAlphas();
    });
  });

  const searchEl = document.getElementById('search');
  const resultsEl = document.getElementById('results');
  let resultsFocus = -1;
  let resultsMatches = [];

  function renderResults() {
    resultsEl.innerHTML = '';
    if (!searchQuery) { resultsEl.classList.remove('open'); return; }
    const q = searchQuery.toLowerCase();
    resultsMatches = [];
    for (let i = 0; i < N; i++) {
      const p = points[i];
      const groupOk = colorMode === 'team' ? activeTeams.has(p.team) : activeClusters.has(p.cluster);
      if (!groupOk) continue;
      const inTitle = p.title.toLowerCase().includes(q);
      const inBody  = !inTitle && ((p.description || '').toLowerCase().includes(q) || (p.team || '').toLowerCase().includes(q));
      if (!inTitle && !inBody) continue;
      resultsMatches.push({ id: i, inTitle });
    }
    // title matches first, then body matches
    resultsMatches.sort((a, b) => (a.inTitle === b.inTitle) ? 0 : (a.inTitle ? -1 : 1));

    const header = document.createElement('div');
    header.className = 'r-count';
    header.textContent = `${resultsMatches.length} ${resultsMatches.length === 1 ? 'match' : 'matches'}`;
    resultsEl.appendChild(header);

    if (resultsMatches.length === 0) {
      const e = document.createElement('div');
      e.className = 'r-empty';
      e.textContent = 'No skills match your search.';
      resultsEl.appendChild(e);
    } else {
      resultsMatches.slice(0, 40).forEach((m, idx) => {
        const p = points[m.id];
        const row = document.createElement('div');
        row.className = 'r-row';
        row.dataset.idx = idx;
        const color = colorHexForPoint(p);
        row.innerHTML = `
          <div class="r-title"></div>
          <div class="r-meta">
            <span class="r-dot" style="background:${color}"></span>
            <span>${escapeHtml(p.team)}</span>
            <span>·</span>
            <span>${escapeHtml(clusters[p.cluster].label)}</span>
          </div>
        `;
        row.querySelector('.r-title').textContent = p.title;
        row.addEventListener('mousedown', (e) => e.preventDefault()); // keep input focused
        row.addEventListener('click', () => selectFromSearch(m.id));
        resultsEl.appendChild(row);
      });
    }
    resultsEl.classList.add('open');
    resultsFocus = -1;
  }

  function selectFromSearch(i) {
    selectedId = i;
    hoverId = i;
    openInfo(points[i]);
    flyToPoint(i);
    positionSelRing();
    updateAlphas();
    hint.classList.add('hidden');
    resultsEl.classList.remove('open');
  }

  // Camera fly-to
  const flyState = { active: false, t: 0, duration: 650,
    fromPos: new THREE.Vector3(), toPos: new THREE.Vector3(),
    fromTgt: new THREE.Vector3(), toTgt: new THREE.Vector3() };
  function flyToPoint(i) {
    const p = points[i];
    const target = new THREE.Vector3(p.x * SCALE, p.y * SCALE, p.z * SCALE);
    const dir = new THREE.Vector3().subVectors(camera.position, controls.target).normalize();
    const distance = 0.75; // desired standoff
    const newCam = target.clone().addScaledVector(dir, distance);
    flyState.active = true;
    flyState.t = 0;
    flyState.fromPos.copy(camera.position);
    flyState.toPos.copy(newCam);
    flyState.fromTgt.copy(controls.target);
    flyState.toTgt.copy(target);
    controls.autoRotate = false;
  }

  searchEl.addEventListener('input', (e) => {
    searchQuery = e.target.value.trim();
    updateAlphas();
    renderResults();
  });
  searchEl.addEventListener('focus', () => {
    if (searchQuery) renderResults();
  });
  searchEl.addEventListener('blur', () => {
    setTimeout(() => resultsEl.classList.remove('open'), 120);
  });
  searchEl.addEventListener('keydown', (e) => {
    if (!resultsEl.classList.contains('open')) return;
    const rows = [...resultsEl.querySelectorAll('.r-row')];
    if (!rows.length) return;
    if (e.key === 'ArrowDown') {
      e.preventDefault();
      resultsFocus = (resultsFocus + 1) % rows.length;
    } else if (e.key === 'ArrowUp') {
      e.preventDefault();
      resultsFocus = (resultsFocus - 1 + rows.length) % rows.length;
    } else if (e.key === 'Enter') {
      e.preventDefault();
      const idx = resultsFocus >= 0 ? resultsFocus : 0;
      selectFromSearch(resultsMatches[idx].id);
      return;
    } else if (e.key === 'Escape') {
      resultsEl.classList.remove('open');
      return;
    } else {
      return;
    }
    rows.forEach((r, i) => r.classList.toggle('focus', i === resultsFocus));
    rows[resultsFocus].scrollIntoView({ block: 'nearest' });
  });

  document.getElementById('resetView').addEventListener('click', () => {
    activeClusters.clear();
    clusters.forEach(c => activeClusters.add(c.id));
    activeTeams.clear();
    teams.forEach(t => activeTeams.add(t.id));
    refreshLegendDim();
    searchEl.value = ''; searchQuery = '';
    camera.position.set(1.1, 0.55, 1.1);
    controls.target.set(0, 0, 0);
    controls.update();
    updateAlphas();
  });

  window.addEventListener('keydown', (e) => {
    if (e.key === 'Escape') {
      if (helpModal.classList.contains('open')) { toggleHelp(false); return; }
      info.classList.remove('open');
      selectedId = -1;
      selRing.visible = false;
      updateAlphas();
    } else if (e.key === '?' || (e.key === '/' && e.shiftKey)) {
      if (document.activeElement.tagName !== 'INPUT') {
        e.preventDefault();
        toggleHelp();
      }
    }
  });

  window.addEventListener('resize', () => {
    camera.aspect = window.innerWidth / window.innerHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(window.innerWidth, window.innerHeight);
    pointMaterial.uniforms.uPixelRatio.value = renderer.getPixelRatio();
    glowMaterial.uniforms.uPixelRatio.value  = renderer.getPixelRatio();
  });

  controls.autoRotate = true;
  controls.autoRotateSpeed = 0.25;
  let idleTimer = null;
  function markInteraction() {
    controls.autoRotate = false;
    clearTimeout(idleTimer);
    idleTimer = setTimeout(() => { controls.autoRotate = true; }, 6000);
  }
  raw.addEventListener('pointerdown', markInteraction);
  raw.addEventListener('wheel', markInteraction);

  function showTip(p, x, y) {
    tipTitle.textContent = p.title;
    const teamCol = (teamById.get(p.team) || {}).color || '#c4cae6';
    tipTeam.innerHTML = `<span class="dot" style="background:${teamCol}"></span><span>${escapeHtml(p.team)}</span>`;
    tipPrev.textContent  = p.preview;
    tipEl.classList.add('show');
    positionTip(x, y);
  }
  function positionTip(x, y) {
    const tw = tipEl.offsetWidth, th = tipEl.offsetHeight;
    let tx = x + 14, ty = y + 14;
    if (tx + tw > window.innerWidth - 10) tx = x - tw - 14;
    if (ty + th > window.innerHeight - 10) ty = y - th - 14;
    tipEl.style.left = tx + 'px';
    tipEl.style.top  = ty + 'px';
  }
  function hideTip() { tipEl.classList.remove('show'); }

  let lastT = performance.now();
  function animate() {
    requestAnimationFrame(animate);
    const now = performance.now();
    const dt = now - lastT;
    lastT = now;
    if (flyState.active) {
      flyState.t += dt;
      let u = Math.min(1, flyState.t / flyState.duration);
      // ease out cubic
      u = 1 - Math.pow(1 - u, 3);
      camera.position.lerpVectors(flyState.fromPos, flyState.toPos, u);
      controls.target.lerpVectors(flyState.fromTgt, flyState.toTgt, u);
      if (u >= 1) flyState.active = false;
    }
    controls.update();
    if (selectedId >= 0) selRing.lookAt(camera.position);
    renderer.render(scene, camera);
  }
  animate();
  updateAlphas();

  setTimeout(() => hint.classList.add('hidden'), 6000);
}

function makeSpriteTexture() {
  const size = 128;
  const c = document.createElement('canvas');
  c.width = c.height = size;
  const ctx = c.getContext('2d');
  const g = ctx.createRadialGradient(size/2, size/2, 0, size/2, size/2, size/2);
  g.addColorStop(0.0, 'rgba(255,255,255,1)');
  g.addColorStop(0.22, 'rgba(255,255,255,0.85)');
  g.addColorStop(0.55, 'rgba(255,255,255,0.22)');
  g.addColorStop(1.0, 'rgba(255,255,255,0)');
  ctx.fillStyle = g;
  ctx.fillRect(0, 0, size, size);
  const tex = new THREE.CanvasTexture(c);
  tex.needsUpdate = true;
  return tex;
}

function escapeHtml(s) {
  return s.replace(/[&<>"']/g, ch => ({
    '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'
  }[ch]));
}
