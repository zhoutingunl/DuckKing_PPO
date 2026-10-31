import { createMicroDuck3D } from './microduck-3d.js';

const $ = (id) => document.getElementById(id);
const statuses = {
  starting: "正在准备",
  training: "正在学习导航",
  rendering: "生成阶段回放",
  evaluating: "独立验收中",
  passed: "训练结束 · 达到目标",
  unqualified: "训练结束 · 未达标",
  stopped: "已停止 · 记录已保留",
  failed: "实验失败",
};
const active = new Set(["starting", "training", "rendering", "evaluating"]);
let runId = new URLSearchParams(location.search).get("run"),
  data,
  selected = 0,
  trajectoryKey = "",
  initialRows = [],
  currentRows = [],
  polling = false;
let sceneStart = { x: -1.4, y: 0 },
  // Start with a long route so the first audience click visibly exercises the
  // 20 s generalization demo. Users can drag either marker to any position.
  sceneGoal = { x: 1.4, y: 0 },
  dragTarget = null,
  playgroundId = null,
  camera = { azimuth: 135, elevation: -45, distance: 1.6 },
  cameraDrag = null,
  scene3d = null,
  live3d = null;
const fmt = (n) => Number(n).toLocaleString("zh-CN");
const duration = (seconds) =>
  seconds == null
    ? "—"
    : seconds < 60
      ? `${Math.round(seconds)} 秒`
      : `${Math.floor(seconds / 60)} 分 ${Math.round(seconds % 60)} 秒`;
const playback = (stage) => stage?.[$("mode").value];
const url = (p) => `/artifacts/${encodeURIComponent(runId)}/${p}`;
const playgroundUrl = (id, suffix = "") =>
  `/api/runs/${encodeURIComponent(runId)}/playground/${encodeURIComponent(id)}${suffix}`;
const scene3dUrl = (id, kind) => playgroundUrl(id, `/${kind}`);
async function api(path, body) {
  const r = await fetch(
    path,
    body === undefined
      ? { cache: "no-store" }
      : {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(body),
        },
  );
  const raw = await r.text();
  let p;
  try {
    p = raw ? JSON.parse(raw) : {};
  } catch {
    throw Error(`请求失败（HTTP ${r.status}，服务返回了非 JSON 内容）`);
  }
  if (!r.ok) throw Error(p.error || `请求失败（HTTP ${r.status}）`);
  return p;
}
function setVideo(id, stage) {
  const video = $(id);
  const clip = playback(stage);
  if (!clip) {
    video.removeAttribute("src");
    delete video.dataset.src;
    video.load();
    return;
  }
  const src = url(clip.video);
  if (video.dataset.src !== src) {
    video.src = src;
    video.poster = url(clip.poster);
    video.dataset.src = src;
    video.playbackRate = Number($("playback").value);
  }
}
function outcome(row) {
  if (!row) return "";
  return `${row.is_success ? "抵达目标" : row.failure_reason === "fall" ? "摔倒" : row.failure_reason === "timeout" ? "超时未抵达" : "回合结束"} · 最终距离 ${row.distance.toFixed(3)} 米 · 仿真 ${row.time.toFixed(1)} 秒`;
}
function stageLabel(s) {
  return s.kind === "initial"
    ? "初始"
    : s.kind === "final"
      ? "最终"
      : s.kind === "stopped"
        ? "停止时"
        : `${(s.timesteps / 1000).toFixed(1)}k 步`;
}
function drawChart(stages) {
  const points = stages.filter((s) => s.evaluation);
  const max = Math.max(data.timesteps || 1, ...points.map((p) => p.timesteps));
  let out = "";
  for (let t = 0; t <= 1; t += 0.25) {
    const y = 188 - t * 165;
    out += `<path d="M45 ${y} H630" stroke="#e1e7eb"/><text x="36" y="${y + 4}" text-anchor="end" font-size="10" fill="#617786">${Math.round(t * 100)}%</text>`;
  }
  if (points.length) {
    out += `<polyline points="${points.map((p) => `${45 + (p.timesteps / max) * 580},${188 - p.evaluation.success_rate * 165}`).join(" ")}" fill="none" stroke="#175d93" stroke-width="2.5"/>`;
    points.forEach((p) => {
      out += `<circle cx="${45 + (p.timesteps / max) * 580}" cy="${188 - p.evaluation.success_rate * 165}" r="4" fill="#c45c20"><title>${fmt(p.timesteps)} 步：${Math.round(p.evaluation.success_rate * 100)}%</title></circle>`;
    });
  }
  out += `<text x="45" y="210" font-size="10" fill="#617786">0 步</text><text x="628" y="210" text-anchor="end" font-size="10" fill="#617786">${fmt(max)} 步</text>`;
  $("chart").innerHTML = out;
}
function drawTrajectory() {
  if (!currentRows.length) return;
  const rows = [...currentRows, ...initialRows];
  const extent =
    Math.max(
      0.7,
      ...rows.flatMap((r) => [
        Math.abs(r.x),
        Math.abs(r.y),
        Math.abs(r.goal_x),
        Math.abs(r.goal_y),
      ]),
    ) * 1.15;
  const point = (x, y) => [130 + (x / extent) * 95, 110 - (y / extent) * 95];
  let out = '<path d="M15 110 H245 M130 10 V210" stroke="#e0e7ec"/>';
  for (const [series, color] of [
    [initialRows, "#a0adb7"],
    [currentRows, "#c45c20"],
  ]) {
    out += `<polyline fill="none" stroke="${color}" stroke-width="2" points="${series.map((r) => point(r.x, r.y).join(",")).join(" ")}"/>`;
  }
  const first = currentRows[0],
    goal = point(first.goal_x, first.goal_y),
    start = point(first.x, first.y);
  out += `<circle cx="${goal[0]}" cy="${goal[1]}" r="${(0.1 / extent) * 95}" fill="#175d9333" stroke="#175d93"/><circle cx="${start[0]}" cy="${start[1]}" r="3" fill="#203d50"/>`;
  $("trajectory").innerHTML = out;
}
async function showStage() {
  if (!data?.stages.length) return;
  selected = Math.max(0, Math.min(selected, data.stages.length - 1));
  $("stage").value = selected;
  const s = data.stages[selected];
  $("stage-title").textContent = `${stageLabel(s)}策略`;
  $("stage-meta").textContent =
    `${fmt(s.timesteps)} 个环境步 · ${fmt(s.updates)} 次优化 epoch · ${s.status === "ready" ? "回放已就绪" : s.status === "failed" ? "导出失败" : "正在评估与渲染"}`;
  $("model-link").href = url(`${s.directory}/policy.zip`);
  setVideo("current", s);
  setVideo("initial", data.stages[0]);
  const clip = playback(s),
    initialClip = playback(data.stages[0]);
  $("video-placeholder").hidden = !!clip;
  if (!clip) {
    $("current").removeAttribute("src");
    delete $("current").dataset.src;
    $("current").load();
  }
  $("stage-outcome").textContent = s.error || outcome(clip?.outcome);
  $("initial-outcome").textContent = outcome(initialClip?.outcome);
  document
    .querySelectorAll("#stage-buttons button")
    .forEach((b, i) => b.classList.toggle("selected", i === selected));
  const key = `${runId}/${s.directory}/${$("mode").value}/${!!clip}/${!!initialClip}`;
  if (key !== trajectoryKey && clip) {
    trajectoryKey = key;
    try {
      const [a, b] = await Promise.all([
        api(url(clip.trajectory)),
        initialClip ? api(url(initialClip.trajectory)) : Promise.resolve([]),
      ]);
      if (key !== trajectoryKey) return;
      currentRows = a;
      initialRows = b;
      drawTrajectory();
    } catch (e) {
      $("message").textContent = e.message;
    }
  }
}
function render() {
  const explorationAvailable =
    data.stages.some((s) => s.exploration) || active.has(data.status);
  $("mode").options[0].disabled = !explorationAvailable;
  if (!explorationAvailable) $("mode").value = "showcase";
  $("empty").hidden = true;
  $("experiment").hidden = false;
  $("status").textContent = statuses[data.status] || data.status;
  $("progress-text").textContent =
    `${fmt(data.timesteps)} / ${fmt(data.budget)} 步`;
  $("progress").value = Math.min(100, (data.timesteps / data.budget) * 100);
  $("elapsed").textContent = duration(data.elapsed_seconds);
  $("recent-return").textContent =
    data.recent_return == null ? "—" : data.recent_return.toFixed(2);
  $("speed").textContent = data.steps_per_second
    ? `${Math.round(data.steps_per_second)} 步/秒`
    : "—";
  $("eta").textContent =
    data.status === "training" && data.steps_per_second
      ? duration(
          Math.max(0, data.budget - data.timesteps) / data.steps_per_second,
        )
      : "—";
  $("stop").disabled = !["starting", "training"].includes(data.status);
  $("start").disabled = active.has(data.status);
  $("stage").max = Math.max(0, data.stages.length - 1);
  if ($("follow").checked) {
    const ready = data.stages
      .map((s, i) => (playback(s) ? i : -1))
      .filter((i) => i >= 0);
    selected = ready.length ? ready.at(-1) : 0;
  }
  $("stage-buttons").replaceChildren(
    ...data.stages.map((s, i) => {
      const b = document.createElement("button");
      b.textContent = stageLabel(s);
      b.className = s.directory === data.best_directory ? "best" : "";
      b.onclick = () => {
        $("follow").checked = false;
        selected = i;
        showStage();
      };
      return b;
    }),
  );
  $("selection-note").textContent = data.best_directory
    ? `最佳策略来自 ${data.stages.find((s) => s.directory === data.best_directory)?.timesteps.toLocaleString()} 步，以验证集成绩选择。最终策略独立保留。`
    : "初始与各阶段按原始顺序保存；视频生成可能稍晚于训练。";
  drawChart(data.stages);
  showStage();
  $("config").textContent = JSON.stringify(
    {
      seed: data.initialization_seed,
      config: data.config,
      training: data.training,
      assets: data.assets,
      evaluation: data.evaluation_protocol,
      snapshots: data.snapshots,
    },
    null,
    2,
  );
  $("message").textContent = data.error || "";
  setupPlayground();
  const rewardMode = data.config?.reward_mode || "distance";
  $("reward-name").textContent = rewardMode === "direct" ? "正面接近" : "距离优先";
  renderRewardFormula(rewardMode);
  const accept = data.acceptance;
  const results = $("acceptance-results");
  results.replaceChildren();
  if (accept) {
    $("acceptance-note").textContent = accept.passed
      ? "最佳策略达到预先设置的导航目标。"
      : accept.qualification === "insufficient_episodes"
        ? "测试回合数不足 100，本轮仅作功能验证。"
        : "本轮未达到预先设置的成功率目标，可查看失败回合与阶段回放。";
    for (const [key, label, target] of [
      ["fixed", "固定布局", 90],
      ["random", "随机布局", 60],
    ]) {
      const r = accept[key],
        div = document.createElement("div");
      const value = document.createElement("b");
      value.textContent = `${Math.round(r.success_rate * 100)}%`;
      const text = document.createElement("span");
      text.textContent = `${label} · ${r.successes}/${r.episodes} · 目标 ${target}%`;
      const ci = document.createElement("div");
      ci.style.fontSize = "11px";
      ci.textContent = `95% 区间 ${(r.success_ci95[0] * 100).toFixed(1)}–${(r.success_ci95[1] * 100).toFixed(1)}%`;
      const quality = document.createElement("div");
      quality.style.fontSize = "11px";
      quality.textContent = `正面指标：朝向 ${(r.mean_heading_alignment_at_goal ?? 0).toFixed(2)} · 倒退 ${(r.reverse_steps_ratio * 100).toFixed(1)}%`;
      div.append(value, text, ci, quality);
      results.append(div);
    }
  } else {
    $("acceptance-note").textContent =
      data.status === "stopped"
        ? "本次训练在最终验收前被停止；当前阶段的检查点成功率不代表最终任务成功率。"
        : data.status === "failed"
          ? "训练或阶段导出失败，未生成最终验收。"
          : "训练结束后，对验证集选出的最佳策略分别运行固定布局和随机布局评估。";
  }
}
const rewardTerms = {
  "Δd": "距离进步：本步前后到目标的距离差，变小则为正奖励。",
  "h": "朝向一致性：鸭子朝向与目标方向的余弦相似度，范围约为 −1 到 1。",
  "a⁺": "正向接近动作：向前速度指令与朝向一致性的乘积。",
  "a⁻": "倒退动作：面向目标时的反向速度指令，会产生惩罚。",
  "Δa": "动作变化：本步动作相对上一步的变化量，用于抑制突变。",
  "𝟙_success": "成功指示量：连续保持在目标半径内达到要求时为 1，否则为 0。",
  "𝟙_failure": "失败指示量：摔倒或越界时为 1，否则为 0。",
  "𝟙_boundary": "越界/摔倒失败指示量。",
};
function formulaToken(token, text) {
  return `<span class="formula-token" tabindex="0" data-explanation="${text}">${token}</span>`;
}
function renderRewardFormula(mode) {
  const formula = $("reward-formula");
  const terms = mode === "direct"
    ? [
        formulaToken("30Δd", rewardTerms["Δd"]),
        "+",
        formulaToken("0.8Δh", rewardTerms.h),
        "+",
        formulaToken("0.10a⁺", rewardTerms["a⁺"]),
        "−",
        formulaToken("0.35a⁻", rewardTerms["a⁻"]),
        "− 0.015 − 0.005",
        formulaToken("Δa²", rewardTerms["Δa"]),
        "+",
        formulaToken("10𝟙_success", rewardTerms["𝟙_success"]),
        "−",
        formulaToken("5𝟙_failure", rewardTerms["𝟙_failure"]),
      ]
    : [
        formulaToken("30Δd", rewardTerms["Δd"]),
        "− 0.015 − 0.005",
        formulaToken("Δa²", rewardTerms["Δa"]),
        "+",
        formulaToken("10𝟙_success", rewardTerms["𝟙_success"]),
        "−",
        formulaToken("5𝟙_failure", rewardTerms["𝟙_failure"]),
      ];
  formula.innerHTML = terms.join(" ");
  formula.querySelectorAll(".formula-token").forEach((element) => {
    const show = () => { $("reward-explanation").textContent = element.dataset.explanation; };
    const clear = () => { $("reward-explanation").textContent = "把鼠标移到公式中的符号上查看中文含义。"; };
    element.addEventListener("mouseenter", show);
    element.addEventListener("focus", show);
    element.addEventListener("mouseleave", clear);
    element.addEventListener("blur", clear);
  });
}
function mapToWorld(x, y) {
  return { x: ((x - 260) / 200) * 3.0, y: ((150 - y) / 110) * 3.0 };
}
function worldToMap(point) {
  return { x: 260 + (point.x / 3.0) * 200, y: 150 - (point.y / 3.0) * 110 };
}
function clampScene(point) {
  return {
    x: Math.max(-3.0, Math.min(3.0, point.x)),
    y: Math.max(-3.0, Math.min(3.0, point.y)),
  };
}
function updateSceneMap() {
  const duck = worldToMap(sceneStart),
    goal = worldToMap(sceneGoal);
  $("scene-duck").setAttribute("transform", `translate(${duck.x} ${duck.y})`);
  $("scene-goal").setAttribute("cx", goal.x);
  $("scene-goal").setAttribute("cy", goal.y);
  $("scene-coords").textContent =
    `起点 (${sceneStart.x.toFixed(2)}, ${sceneStart.y.toFixed(2)}) · 目标 (${sceneGoal.x.toFixed(2)}, ${sceneGoal.y.toFixed(2)}) · 距离 ${Math.hypot(sceneGoal.x - sceneStart.x, sceneGoal.y - sceneStart.y).toFixed(2)} m`;
  const sceneDistance = Math.hypot(sceneGoal.x - sceneStart.x, sceneGoal.y - sceneStart.y);
  const valid = sceneDistance >= 0.1 && sceneDistance <= 8.5;
  $("play-run").disabled =
    !data ||
    !["passed", "unqualified", "stopped"].includes(data.status) ||
    !valid ||
    !$("play-stage").value;
  if (sceneDistance < 0.1) $("play-message").textContent = "起点和目标至少相距 0.1 米。";
  else if (sceneDistance > 0.65) $("play-message").textContent = "当前场景超出训练距离分布（0.65m），将作为 20 秒泛化测试运行。";
  else if (data && $("play-stage").value) $("play-message").textContent = "可以运行当前场景。";
}
function setupPlayground() {
  const select = $("play-stage"),
    previous = select.value;
  select.replaceChildren(
    ...(data?.stages || [])
      .filter((s) => s.status === "ready" && s.directory)
      .map((s) => {
        const option = document.createElement("option");
        option.value = s.directory;
        option.textContent =
          stageLabel(s) +
          (s.directory === data.best_directory ? " · 最佳" : "");
        return option;
      }),
  );
  if (previous && [...select.options].some((o) => o.value === previous))
    select.value = previous;
  scene3d?.setPlacement([sceneStart.x, sceneStart.y], [sceneGoal.x, sceneGoal.y]);
  updateSceneMap();
}
function updateCameraReadout() {
  $("camera-readout").textContent =
    `视角 ${Math.round(camera.azimuth)}° · ${Math.round(camera.elevation)}° · ${camera.distance.toFixed(1)}×`;
}
async function sendCamera(payload) {
  if (!playgroundId) return;
  try {
    const result = await api(
      `/api/runs/${encodeURIComponent(runId)}/playground/${encodeURIComponent(playgroundId)}/camera`,
      payload,
    );
    camera = result.camera;
    updateCameraReadout();
  } catch (error) {
    $("play-message").textContent = error.message;
  }
}
function updateRewardInfo(info) {
  const panel = $("play-reward");
  if (!info || info.reward_mode == null) {
    panel.hidden = true;
    return;
  }
  panel.hidden = false;
  $("play-reward-mode").textContent = info.reward_mode === "direct"
    ? "正面接近：鼓励鸭头朝向目标并向前走，面向目标倒退会扣分。"
    : "距离优先：只按距离进步学习，可能出现倒退进入目标。";
  for (const [id, key] of [["reward-progress", "reward_progress"], ["reward-heading", "reward_heading"], ["reward-approach", "reward_approach"], ["reward-reverse", "reward_reverse"]]) {
    $(id).textContent = info[key] == null ? "—" : Number(info[key]).toFixed(3);
  }
}
function svgPoint(event) {
  const svg = $("scene-map"),
    rect = svg.getBoundingClientRect();
  return {
    x: ((event.clientX - rect.left) * 520) / rect.width,
    y: ((event.clientY - rect.top) * 300) / rect.height,
  };
}
function moveScene(event) {
  if (!dragTarget) return;
  const point = clampScene(mapToWorld(...Object.values(svgPoint(event))));
  if (dragTarget === "duck") sceneStart = point;
  else sceneGoal = point;
  updateSceneMap();
}
$("scene-map").addEventListener("pointerdown", (event) => {
  const point = svgPoint(event),
    duck = worldToMap(sceneStart),
    goal = worldToMap(sceneGoal);
  dragTarget =
    Math.hypot(point.x - duck.x, point.y - duck.y) < 28
      ? "duck"
      : Math.hypot(point.x - goal.x, point.y - goal.y) < 28
        ? "goal"
        : null;
  if (dragTarget) event.currentTarget.setPointerCapture(event.pointerId);
});
$("scene-map").addEventListener("pointermove", moveScene);
$("scene-map").addEventListener("pointerup", () => {
  dragTarget = null;
});
$("scene-map").addEventListener("pointercancel", () => {
  dragTarget = null;
});
$("play-yaw").oninput = () => {
  $("play-yaw-value").textContent = `${$("play-yaw").value}°`;
};
$("play-stage").onchange = updateSceneMap;
$("play-run").onclick = async () => {
  if (!runId || !$("play-stage").value) return;
  $("play-run").disabled = true;
  $("play-message").textContent = "正在用这个场景运行 MuJoCo…";
  $("play-result").hidden = true;
  try {
    const result = await api(
      `/api/runs/${encodeURIComponent(runId)}/playground`,
      {
        stage: $("play-stage").value,
        start: [sceneStart.x, sceneStart.y],
        goal: [sceneGoal.x, sceneGoal.y],
        yaw: (Number($("play-yaw").value) * Math.PI) / 180,
      },
    );
    playgroundId = result.id;
    live3d?.setPlacement([sceneStart.x, sceneStart.y], [sceneGoal.x, sceneGoal.y]);
    camera = { azimuth: 135, elevation: -45, distance: 1.6 };
    updateCameraReadout();
    let liveModelLoaded = false;
    const wait = async () => {
      const record = await api(
        `/api/runs/${encodeURIComponent(runId)}/playground/${encodeURIComponent(playgroundId)}`,
      );
      if (record.visual_ready && !liveModelLoaded) {
        try {
          const modelPayload = await api(scene3dUrl(playgroundId, "visual-model"));
          live3d?.loadModel(modelPayload);
          liveModelLoaded = true;
        } catch (error) {
          $("play-message").textContent = `3D 模型加载中：${error.message}`;
        }
      }
      if (["starting", "running"].includes(record.status)) {
        $("play-frame").src =
          `${playgroundUrl(playgroundId, "/frame.jpg")}?seq=${record.frame_seq}`;
        const info = record.info || {};
        try {
          const visual = await api(scene3dUrl(playgroundId, "visual-state"));
          if (liveModelLoaded) live3d?.updateState(visual);
        } catch (error) {
          // The JPEG evidence remains available while the first mesh state loads.
          console.debug("3D state pending", error);
        }
        if (record.camera) camera = record.camera;
        updateCameraReadout();
        updateRewardInfo(info);
        $("play-result-title").textContent =
          `实时仿真中 · ${info.time == null ? "0.0" : info.time.toFixed(1)} 秒`;
        $("play-result-meta").textContent =
          `当前距离 ${(info.distance || 0).toFixed(3)} 米 · 画面每 0.1 秒更新`;
        $("play-result").hidden = false;
        return setTimeout(wait, 100);
      }
      if (record.status !== "ready")
        throw Error(record.error || "场景回放失败");
      $("play-frame").src =
        `${playgroundUrl(playgroundId, "/frame.jpg")}?seq=${record.frame_seq}`;
      const outcome = record.info;
      try {
        const visual = await api(scene3dUrl(playgroundId, "visual-state"));
        live3d?.updateState(visual);
      } catch (error) {
        console.debug("3D final state unavailable", error);
      }
      if (record.camera) camera = record.camera;
      updateCameraReadout();
      updateRewardInfo(outcome);
      $("play-result-title").textContent = outcome.is_success
        ? "抵达了你放置的目标"
        : outcome.failure_reason === "fall"
          ? "小鸭摔倒了"
          : "这次没有抵达";
      $("play-result-meta").textContent =
        `策略 ${stageLabel(data.stages.find((s) => s.directory === record.stage))} · 最终距离 ${outcome.distance.toFixed(3)} 米 · 仿真 ${outcome.time.toFixed(1)} 秒 · 用时 ${record.elapsed_seconds.toFixed(1)} 秒`;
      $("play-result").hidden = false;
      $("play-message").textContent =
        "可以继续拖动位置，或换一个训练阶段再运行。";
      updateSceneMap();
    };
    await wait();
  } catch (error) {
    $("play-message").textContent = error.message;
    updateSceneMap();
  }
};
$("play-frame").addEventListener("pointerdown", (event) => {
  if (!playgroundId) return;
  cameraDrag = { x: event.clientX, y: event.clientY };
  event.currentTarget.setPointerCapture(event.pointerId);
  event.currentTarget.classList.add("dragging");
});
$("play-frame").addEventListener("pointermove", (event) => {
  if (!cameraDrag) return;
  const dx = event.clientX - cameraDrag.x;
  const dy = event.clientY - cameraDrag.y;
  cameraDrag = { x: event.clientX, y: event.clientY };
  camera.azimuth = (camera.azimuth + dx * 0.65 + 360) % 360;
  camera.elevation = Math.max(-80, Math.min(-10, camera.elevation - dy * 0.45));
  updateCameraReadout();
  clearTimeout(window.cameraTimer);
  window.cameraTimer = setTimeout(() => sendCamera(camera), 30);
});
$("play-frame").addEventListener("pointerup", (event) => {
  cameraDrag = null;
  event.currentTarget.classList.remove("dragging");
});
$("play-frame").addEventListener("pointercancel", () => { cameraDrag = null; });
$("play-frame").addEventListener("wheel", (event) => {
  if (!playgroundId) return;
  event.preventDefault();
  camera.distance = Math.max(0.5, Math.min(4, camera.distance * Math.exp(event.deltaY * 0.001)));
  updateCameraReadout();
  clearTimeout(window.cameraTimer);
  window.cameraTimer = setTimeout(() => sendCamera(camera), 30);
}, { passive: false });
$("camera-reset").onclick = () => {
  live3d?.resetCamera();
  sendCamera({ reset: true });
};
async function poll() {
  if (polling) return;
  polling = true;
  try {
    const runs = await api("/api/runs");
    const picker = $("runs");
    picker.replaceChildren(
      ...runs.map((r) => {
        const o = document.createElement("option");
        o.value = r.id;
        const rewardLabel = r.reward_mode === "direct" ? "正面接近" : "距离优先";
        o.textContent = `${r.id} · ${rewardLabel} · ${statuses[r.status] || r.status}`;
        return o;
      }),
    );
    if (!runId && runs.length) runId = runs[0].id;
    if (!runs.length) {
      $("empty").hidden = false;
      $("experiment").hidden = true;
      $("start").disabled = false;
      return;
    }
    picker.value = runId;
    data = await api(`/api/runs/${encodeURIComponent(runId)}`);
    render();
    loadEditorModel();
  } catch (e) {
    $("message").textContent = `读取记录失败：${e.message}`;
  } finally {
    polling = false;
  }
}
$("runs").onchange = () => {
  runId = $("runs").value;
  selected = 0;
  trajectoryKey = "";
  history.replaceState(null, "", `?run=${encodeURIComponent(runId)}`);
  poll();
};
$("stage").oninput = () => {
  $("follow").checked = false;
  selected = Number($("stage").value);
  showStage();
};
$("mode").onchange = () => {
  trajectoryKey = "";
  showStage();
};
$("playback").onchange = () => {
  for (const id of ["initial", "current"])
    $(id).playbackRate = Number($("playback").value);
};
$("compare").onclick = async () => {
  for (const id of ["initial", "current"]) {
    const v = $(id);
    if (v.dataset.src) {
      v.currentTime = 0;
      v.playbackRate = Number($("playback").value);
      try {
        await v.play();
      } catch (e) {
        $("message").textContent = e.message;
      }
    }
  }
};
$("start").onclick = async () => {
  $("start").disabled = true;
  $("message").textContent = "正在创建新实验…";
  try {
    const r = await api("/api/train", {
      timesteps: Number($("budget").value),
      envs: Number($("envs").value),
      reward_mode: $("reward-mode").value,
    });
    runId = r.id;
    $("follow").checked = true;
    setTimeout(poll, 1500);
  } catch (e) {
    $("message").textContent = e.message;
    $("start").disabled = false;
  }
};
$("stop").onclick = async () => {
  try {
    await api(`/api/runs/${encodeURIComponent(runId)}/stop`, {});
    $("message").textContent = "已请求停止，正在保留最后策略和回放。";
    $("stop").disabled = true;
  } catch (e) {
    $("message").textContent = e.message;
  }
};
poll();
setInterval(poll, 3000);

try {
  scene3d = createMicroDuck3D($("microduck-3d-editor"), {
    onPlace: ({ start, goal }) => {
      sceneStart = start;
      sceneGoal = goal;
      updateSceneMap();
    },
  });
  live3d = createMicroDuck3D($("microduck-3d-live"), {
    onView: (next) => {
      camera = next;
      updateCameraReadout();
      clearTimeout(window.cameraTimer);
      window.cameraTimer = setTimeout(() => sendCamera(camera), 80);
    },
  });
  scene3d.setScene([sceneStart.x, sceneStart.y], [sceneGoal.x, sceneGoal.y]);
  live3d.setScene([sceneStart.x, sceneStart.y], [sceneGoal.x, sceneGoal.y]);
} catch (error) {
  console.error("MicroDuck 3D viewer unavailable", error);
  $("play-message").textContent = "3D 视图暂不可用，仍可使用二维场景编辑器。";
}
let editorModelRun = "";
async function loadEditorModel() {
  if (!scene3d || !runId || editorModelRun === runId) return;
  try {
    const model = await api(`/api/runs/${encodeURIComponent(runId)}/visual-model`);
    if (runId !== data?.id && runId !== new URLSearchParams(location.search).get("run")) return;
    scene3d.loadModel(model);
    scene3d.setScene([sceneStart.x, sceneStart.y], [sceneGoal.x, sceneGoal.y]);
    editorModelRun = runId;
  } catch (error) {
    $("play-message").textContent = `3D 官方模型加载失败：${error.message}`;
  }
}
