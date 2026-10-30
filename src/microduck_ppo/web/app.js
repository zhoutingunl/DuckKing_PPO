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
const fmt = (n) => Number(n).toLocaleString("zh-CN");
const duration = (seconds) =>
  seconds == null
    ? "—"
    : seconds < 60
      ? `${Math.round(seconds)} 秒`
      : `${Math.floor(seconds / 60)} 分 ${Math.round(seconds % 60)} 秒`;
const playback = (stage) => stage?.[$("mode").value];
const url = (p) => `/artifacts/${encodeURIComponent(runId)}/${p}`;
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
  const p = await r.json();
  if (!r.ok) throw Error(p.error || "请求失败");
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
      div.append(value, text, ci);
      results.append(div);
    }
  } else
    $("acceptance-note").textContent =
      "训练结束后，对验证集选出的最佳策略分别运行固定布局和随机布局评估。";
}
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
        o.textContent = `${r.id} · ${statuses[r.status] || r.status}`;
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
