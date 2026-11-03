// Pick & Place UI: polls /api/state, sends commands. E-stop and Go Home are always live.
"use strict";

const $ = (id) => document.getElementById(id);
const STATUS_TEXT = {
  starting: "Starting up", idle: "Ready", busy: "Working", stopped: "STOPPED", error: "Needs attention", quitting: "Shutting down",
};
const CONF_OPTIONS = [0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.8];
const CLEARANCE_OPTIONS = [1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20];
const EMOJI = [
  [/banana/, "🍌"], [/apple/, "🍎"], [/orange/, "🍊"], [/lemon/, "🍋"], [/bottle/, "🧴"],
  [/coffee|mug/, "☕"], [/cup|glass/, "🥤"], [/bowl/, "🥣"], [/ball/, "⚽"], [/box|package/, "📦"],
  [/marker|pen|pencil/, "🖊️"], [/tool|wrench|screwdriver/, "🔧"], [/block|cube|object/, "🧊"],
  [/plate|dish/, "🍽️"], [/can\b/, "🥫"], [/toy|duck/, "🦆"], [/basket|bin/, "🧺"],
];

let state = null;
let presets = [];
let drawerOpen = false;
let menuRole = null;
let failures = 0;
let toastTimer = null;
let resultKey = null;          // result already shown, so each pick's result pops up once
let resultTimer = null;
let selectEditUntil = 0;       // don't overwrite a dropdown the user is changing
let quitting = false;          // this page asked to quit, or saw the robot shutting down
let shutDown = false;          // server gone after quitting

const emojiFor = (label) => (EMOJI.find(([re]) => re.test(label.toLowerCase())) || [null, "🔹"])[1];
const esc = (t) => String(t).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

// ---------------------------------------------------------------- api

async function api(path, body) {
  try {
    const res = await fetch(path, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body || {}),
    });
    if (!res.ok) {
      let msg = res.statusText;
      try { msg = (await res.json()).detail || msg; } catch (_) { /* not JSON */ }
      toast(msg);
      return false;
    }
    poll();
    return true;
  } catch (_) {
    toast("Can't reach the robot server");
    return false;
  }
}

function toast(text) {
  const t = $("toast");
  t.textContent = text;
  t.classList.remove("hidden");
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => t.classList.add("hidden"), 3500);
}

// ---------------------------------------------------------------- safety

function emergencyStop() {
  // Fire immediately; retry once if the first request fails.
  fetch("/api/estop", { method: "POST" }).catch(() =>
    setTimeout(() => fetch("/api/estop", { method: "POST" }).catch(() => toast("E-stop could not reach the robot!")), 150));
  document.body.dataset.status = "stopped";
  $("stopOverlay").classList.remove("hidden");
}
$("estopBtn").addEventListener("pointerdown", (e) => { e.preventDefault(); emergencyStop(); });
$("estopBtn").addEventListener("keydown", (e) => { if (e.key === "Enter") emergencyStop(); });
$("homeBtn").addEventListener("click", () => api("/api/home"));
$("byeStop").addEventListener("pointerdown", (e) => { e.preventDefault(); emergencyStop(); });

// ---------------------------------------------------------------- quit

function setQuitDialog(open) {
  $("quitDialog").classList.toggle("hidden", !open);
  if (open) {
    $("quitHeld").classList.toggle("hidden", !(state && state.holding));
    $("quitCancel").focus();
  }
}
$("quitBtn").addEventListener("click", () => setQuitDialog(true));
$("quitCancel").addEventListener("click", () => setQuitDialog(false));
$("quitDialog").addEventListener("click", (e) => { if (e.target === $("quitDialog")) setQuitDialog(false); });
$("quitConfirm").addEventListener("click", async () => {
  setQuitDialog(false);
  if (await api("/api/quit")) {
    quitting = true;
    renderBye();
  }
});

function renderBye() {
  const bye = $("byeScreen");
  bye.classList.toggle("hidden", !quitting);
  bye.classList.toggle("done", shutDown);
  $("byeSpinner").classList.toggle("hidden", shutDown);
  $("byeStop").classList.toggle("hidden", shutDown);
  $("byeTitle").textContent = shutDown ? "Robot shut down" : "Parking the arm…";
  $("byeText").textContent = shutDown
    ? "The program has stopped. You can close this page."
    : (state && state.message) || "Folding to the rest position and closing the gripper.";
}

document.addEventListener("keydown", (e) => {
  if (["INPUT", "SELECT"].includes(e.target.tagName)) {
    if (e.key === "Escape") closeMenu();
    return;
  }
  if (e.code === "Space") { e.preventDefault(); emergencyStop(); }
  if (e.key === "Escape") { closeMenu(); setQuitDialog(false); if (drawerOpen) setDrawer(false); }
});

// ---------------------------------------------------------------- task pickers

document.querySelectorAll(".picker").forEach((p) =>
  p.querySelector(".picker-btn").addEventListener("click", (e) => {
    e.stopPropagation();
    menuRole === p.dataset.role ? closeMenu() : openMenu(p.dataset.role, e.currentTarget);
  }));

function openMenu(role, anchor) {
  menuRole = role;
  const current = state[role];
  const other = role === "pick" ? state.place : state.pick;
  const labels = presets.includes(current) ? presets : [current, ...presets];
  $("menuGrid").replaceChildren(...labels.map((label) => {
    const b = document.createElement("button");
    b.type = "button";
    b.className = "menu-item" + (label === current ? " sel" : "");
    b.innerHTML = `<span class="emoji">${emojiFor(label)}</span><span>${esc(label)}</span>`;
    b.disabled = label.toLowerCase() === other.toLowerCase();
    b.title = b.disabled ? `Already chosen as the ${role === "pick" ? "destination" : "object"}` : "";
    b.addEventListener("click", () => chooseTask(role, label));
    return b;
  }));
  $("customInput").value = "";
  $("customInput").placeholder = role === "pick" ? "Or type any object, e.g. red apple" : "Or type any destination, e.g. blue bowl";

  const menu = $("menu");
  menu.classList.remove("hidden");
  const r = anchor.getBoundingClientRect();
  const w = menu.offsetWidth, h = menu.offsetHeight;
  const left = Math.min(Math.max(12, r.left + r.width / 2 - w / 2), window.innerWidth - w - 12);
  const below = r.bottom + 8;
  const top = below + h > window.innerHeight - 12 ? Math.max(12, r.top - h - 8) : below;
  menu.style.left = left + "px";
  menu.style.top = top + "px";
}

function closeMenu() {
  menuRole = null;
  $("menu").classList.add("hidden");
}
document.addEventListener("click", (e) => { if (menuRole && !$("menu").contains(e.target)) closeMenu(); });

$("customForm").addEventListener("submit", (e) => {
  e.preventDefault();
  const text = $("customInput").value.trim();
  if (text && menuRole) chooseTask(menuRole, text);
});

async function chooseTask(role, label) {
  const pick = role === "pick" ? label : state.pick;
  const place = role === "place" ? label : state.place;
  closeMenu();
  if (await api("/api/task", { pick, place })) {
    state.pick = pick;
    state.place = place;
    renderTask(state);
  }
}

$("startBtn").addEventListener("click", () => api("/api/start"));

// ---------------------------------------------------------------- gripper + settings

document.querySelectorAll("[data-gripper]").forEach((b) =>
  b.addEventListener("click", () => api("/api/gripper", { open: b.dataset.gripper === "open" })));

function fillSelect(sel, values, current, label) {
  const vals = values.includes(current) ? values : [...values, current].sort((a, b) => a - b);
  const key = vals.join(",");
  if (sel.dataset.key !== key) {
    sel.innerHTML = vals.map((v) => `<option value="${v}">${label(v)}</option>`).join("");
    sel.dataset.key = key;
  }
  sel.value = String(current);
}

$("confSelect").addEventListener("change", (e) => {
  selectEditUntil = Date.now() + 1500;
  api("/api/settings", { conf: +e.target.value });
});
$("clrSelect").addEventListener("change", (e) => {
  selectEditUntil = Date.now() + 1500;
  state.clearance_cm = +e.target.value;
  renderReleaseHelp(state);
  api("/api/settings", { clearance_cm: +e.target.value });
});

// The blue zone is empty at the current release height: the arm can't reach a drop point
// anywhere in view (only once the zone has been computed for about this height).
function dropBlocked(s) {
  const z = s.drop_zone;
  if (!z || z.cells > 0) return false;
  const wanted = (s.place_top_cm || 0) + s.clearance_cm;
  return Math.abs(z.height_cm - wanted) < 3;
}

function renderReleaseHelp(s) {
  const place = esc(s.place);
  const clr = s.clearance_cm;
  const base = `Measured from the <b>top of the ${place}</b> (its rim or upper surface) &mdash; not from the table.`;
  $("clrHelp").innerHTML = s.place_top_cm != null
    ? `${base} The ${place}'s top is <b>${s.place_top_cm} cm</b> above the table, so the gripper lets go at
       <b>${(s.place_top_cm + clr).toFixed(1)} cm</b> above the table.`
    : `${base} The ${place}'s height isn't measured yet; until it is, the drop counts from the table
       (gripper lets go at <b>${clr} cm</b>).`;
  if (dropBlocked(s)) {
    $("clrHelp").innerHTML += `<span class="warn">⚠ The arm can't reach a drop point
      ${s.drop_zone.height_cm} cm high anywhere in view, so the blue area is gone. Choose a lower release height.</span>`;
  }
}

// ---------------------------------------------------------------- staff drawer

function setDrawer(open) {
  drawerOpen = open;
  $("drawer").classList.toggle("open", open);
  $("drawer").setAttribute("aria-hidden", String(!open));
  $("drawerScrim").classList.toggle("hidden", !open);
  if (open) poll();
}
$("staffBtn").addEventListener("click", () => setDrawer(!drawerOpen));
$("drawerClose").addEventListener("click", () => setDrawer(false));
$("drawerScrim").addEventListener("click", () => setDrawer(false));
document.querySelectorAll("[data-cmd]").forEach((b) =>
  b.addEventListener("click", () => api("/api/" + b.dataset.cmd)));
document.querySelectorAll("[data-retry]").forEach((b) =>
  b.addEventListener("click", () => api("/api/retry", { skip_zero_check: b.dataset.retry === "1" })));

// ---------------------------------------------------------------- render

function stepText(s) {
  if (s.status === "stopped") return null;
  if (s.action === "pick") {
    return [
      "Waking up the arm…",
      `Looking for the ${s.pick}…`,
      `Reaching for the ${s.pick}…`,
      "Lifting it up…",
      `Dropping it on the ${s.place}…`,
      "Heading back…",
    ][s.step] || s.message;
  }
  if (s.action === "home") return s.message || "Going home…";
  if (s.action === "ready") return "Getting ready…";
  if (s.action === "gripper") return "Moving the gripper…";
  if (s.status === "quitting") return s.message;
  if (s.status === "starting") return s.message;
  return null;
}

function renderTask(s) {
  const locked = s.status !== "idle" || s.action !== null;
  for (const role of ["pick", "place"]) {
    const p = document.querySelector(`.picker[data-role="${role}"]`);
    const label = s[role];
    p.querySelector(".emoji").textContent = emojiFor(label);
    p.querySelector(".name").textContent = label;
    p.querySelector(".picker-btn").disabled = locked;
    const det = s.detections.find((d) => d.role === role);
    const seen = p.querySelector(".seen");
    if (!s.fps) {
      seen.className = "seen";
      seen.textContent = "";
    } else if (role === "place" && dropBlocked(s)) {
      seen.className = "seen no";
      seen.textContent = "⚠ Too high to drop - lower the release height";
    } else if (det) {
      seen.className = "seen ok";
      seen.textContent = "✓ I see it" + (role === "pick" && det.width_cm ? ` · ${det.width_cm} cm wide` : "");
    } else {
      seen.className = "seen no";
      seen.textContent = "Not in view yet";
    }
  }
  if (locked && menuRole) closeMenu();
}

function renderResult(s) {
  const key = s.result ? `${s.stats.attempts}|${s.result.text}` : null;
  const ov = $("resultOverlay");
  if (s.action === "pick" || s.status === "stopped") {
    ov.classList.add("hidden");
  } else if (key && key !== resultKey && !s.action) {
    resultKey = key;
    ov.className = "result-overlay " + (s.result.ok ? "ok" : "fail");
    $("resultIcon").textContent = s.result.ok ? "✓" : "!";
    $("resultText").textContent = s.result.text;
    clearTimeout(resultTimer);
    resultTimer = setTimeout(() => ov.classList.add("hidden"), 6000);
  }
}
$("resultOverlay").addEventListener("click", () => $("resultOverlay").classList.add("hidden"));

function render(s) {
  if (resultKey === null && s.result) resultKey = `${s.stats.attempts}|${s.result.text}`;  // not on page load
  state = s;
  document.body.dataset.status = s.status;
  if (shutDown) { location.reload(); return; }   // server is back after a quit: start fresh
  // Show the shutdown screen while parking; an E-stop during it cancels the quit.
  quitting = s.status === "quitting";
  renderBye();
  $("quitBtn").disabled = quitting;
  $("statusPill").querySelector("span").textContent = STATUS_TEXT[s.status] || s.status;

  // stage
  const banner = stepText(s);
  $("stepBanner").textContent = banner || "";
  $("stepBanner").classList.toggle("hidden", !banner);
  $("stopOverlay").classList.toggle("hidden", s.status !== "stopped");
  $("videoPlaceholder").classList.toggle("hidden", s.fps > 0);
  $("videoPlaceholder").querySelector("span").textContent =
    s.status === "error" && !s.fps ? "Camera unavailable" : "Starting the camera…";
  $("fps").textContent = s.fps ? `${Math.round(s.fps)} fps` : "--";
  $("modeBadge").classList.toggle("hidden", !s.dry_run);
  $("successCount").textContent = s.stats.success;
  $("steps").querySelectorAll("li").forEach((li) => {
    const n = +li.dataset.step;
    const running = s.action === "pick";
    li.classList.toggle("active", running && n === s.step);
    li.classList.toggle("done", running && n < s.step);
  });
  renderResult(s);

  // task + start
  renderTask(s);
  const blocked = dropBlocked(s);
  const canStart = s.status === "idle" && s.action === null && s.robot && s.can_grasp && !blocked;
  $("startBtn").disabled = !canStart;
  $("startLabel").textContent =
    s.action === "pick" ? "WORKING…" :
    s.action ? "PLEASE WAIT…" :
    s.status === "stopped" ? "STOPPED" :
    s.status === "starting" ? "STARTING…" :
    s.status === "error" ? "UNAVAILABLE" :
    !s.robot ? "ARM OFFLINE" :
    !s.can_grasp ? "NOT CALIBRATED" :
    blocked ? "RELEASE TOO HIGH" : "START";
  $("homeBtn").disabled = !s.robot || s.action === "home";

  // gripper
  const gripOk = s.robot && s.status === "idle" && s.action === null;
  $("openBtn").disabled = !gripOk;
  $("closeBtn").disabled = !gripOk;
  const max = s.jaw_max_cm || 9;
  $("jawFill").style.width = s.jaw_cm == null ? "0" : Math.min(100, (s.jaw_cm / max) * 100) + "%";
  $("jawText").textContent = s.jaw_cm == null ? "–" : `${s.jaw_cm.toFixed(1)} / ${max} cm`;
  $("holdTag").classList.toggle("hidden", !s.holding);

  // settings
  if (Date.now() > selectEditUntil && document.activeElement.tagName !== "SELECT") {
    fillSelect($("confSelect"), CONF_OPTIONS, s.conf,
      (v) => v.toFixed(2) + (v <= 0.25 ? "  · finds more" : v >= 0.6 ? "  · very strict" : v === 0.5 ? "  · default" : ""));
    fillSelect($("clrSelect"), CLEARANCE_OPTIONS, s.clearance_cm, (v) => `${v} cm above the destination's top`);
  }
  renderReleaseHelp(s);

  // errors
  $("errorCard").classList.toggle("hidden", s.status !== "error");
  $("retryRow").classList.toggle("hidden", s.status !== "error");
  $("errorText").textContent = s.error || "";
  $("zeroReport").textContent = (s.zero_report || []).join("\n");
  $("zeroReport").classList.toggle("hidden", !(s.zero_report || []).length);

  // staff drawer
  if (drawerOpen) {
    const facts = {
      "Status": `${s.status}${s.action ? " (" + s.action + ")" : ""}`,
      "Arm pose": s.pose,
      "Robot": s.robot ? "connected" : "not connected",
      "Holding object": s.holding ? "yes" : "no",
      "Destination top": s.place_top_cm == null ? "not measured" : s.place_top_cm + " cm",
      "Camera": s.fps + " fps",
      "Detections": s.detections.map((d) => `${d.name} ${d.conf}`).join(", ") || "none",
      "Picks": `${s.stats.success} / ${s.stats.attempts} succeeded`,
      "Mode": s.dry_run ? "dry run (no pick motion)" : "live",
    };
    $("facts").innerHTML = Object.entries(facts).map(([k, v]) => `<dt>${k}</dt><dd>${esc(v)}</dd>`).join("");
    if (s.log) {
      const log = $("log");
      const atBottom = log.scrollHeight - log.scrollTop - log.clientHeight < 30;
      log.textContent = s.log.join("\n");
      if (atBottom) log.scrollTop = log.scrollHeight;
    }
  }
}

// ---------------------------------------------------------------- loops

async function poll() {
  try {
    const res = await fetch("/api/state" + (drawerOpen ? "?log=true" : ""), { cache: "no-store" });
    if (!res.ok) throw new Error(res.statusText);
    render(await res.json());
    failures = 0;
    $("connBanner").classList.add("hidden");
  } catch (_) {
    if (quitting) {                              // the server stopped after parking: done
      shutDown = true;
      renderBye();
      return;
    }
    if (++failures >= 3) $("connBanner").classList.remove("hidden");
  }
}

const video = $("video");
video.addEventListener("error", () => setTimeout(() => { video.src = "/video?t=" + Date.now(); }, 2000));

// Header logos and title from config (ui.logos / ui.title); files live in ui/static/logos/.
function renderBranding(b) {
  if (!b) return;
  const logos = (b.logos || []).map((l) =>
    `<img src="/static/logos/${encodeURIComponent(l.file)}" alt="${esc(l.alt || "")}" class="logo"` +
    ` style="--h:${Number(l.height) || 30}px">`);
  $("brands").innerHTML = logos.join('<span class="x" aria-hidden="true">&times;</span>');
  $("brands").setAttribute("aria-label", (b.logos || []).map((l) => l.alt).join(" x "));
  if (b.title) {
    document.title = b.title;
    $("appTitle").innerHTML = esc(b.title).replace(/&amp;/g, "<span>&amp;</span>");
  }
}

(async function boot() {
  try {
    const cfg = await (await fetch("/api/config")).json();
    presets = cfg.presets || [];
    renderBranding(cfg.branding);
  } catch (_) { presets = []; }
  await poll();
  setInterval(poll, 300);
})();
