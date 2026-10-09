"use strict";

// This preview uses local sample data. It never sends prompts or runs tools.
const scenarios = {
  explore: {
    prompt: "Help me understand this project.",
    intro: "I'll map the files and follow the entry points.",
    code: 'files = await call_tool("list_files", {"path": "src"})\nentry = await call_tool("read_file", {"path": "src/main.py"})\nconfig = await call_tool("read_file", {"path": "pyproject.toml"})',
    title: "Your project, a little clearer.",
    rows: [["src/main.py", "The starting point"], ["src/tools/", "The building blocks"], ["tests/", "The safety net"]],
    phases: ["Mapping sample files", "Reading the entry point", "Connecting the pieces"]
  },
  review: {
    prompt: "Review the changes to my parser.",
    intro: "I'll read the implementation and its test coverage.",
    code: 'parser = await call_tool("read_file", {"path": "src/parser.py"})\ntests = await call_tool("read_file", {"path": "tests/test_parser.py"})\nconfig = await call_tool("read_file", {"path": "pyproject.toml"})',
    title: "One edge case worth a closer look.",
    rows: [["parser.py:24", "Empty input needs a guard"], ["test_parser.py", "Add an empty-input case"], ["Next step", "Review, then update"]],
    phases: ["Reading the sample parser", "Checking sample tests", "Summarizing the review"]
  },
  plan: {
    prompt: "Plan a CSV export for this project.",
    intro: "I'll find the right place and break the work into steps.",
    code: 'files = await call_tool("list_files", {"path": "src"})\nmodels = await call_tool("read_file", {"path": "src/models.py"})\ncli = await call_tool("read_file", {"path": "src/cli.py"})',
    title: "A plan you can take one step at a time.",
    rows: [["01 / Serialize", "Map records to CSV rows"], ["02 / Connect", "Add a CLI export option"], ["03 / Verify", "Test empty and mixed data"]],
    phases: ["Exploring the sample project", "Reading the data model", "Preparing a small plan"]
  }
};

const get = (id) => document.getElementById(id);
const scenarioSelect = get("demo-scenario");
const transcript = get("transcript");
const runButton = get("demo-run");
const result = get("demo-result");
const demoState = get("demo-state");
const timers = new Set();
let running = false;
let setupMode = "install";
let toastTimer;

function clearRunTimers() {
  timers.forEach(clearTimeout);
  timers.clear();
}

function later(fn, delay) {
  const timer = setTimeout(() => {
    timers.delete(timer);
    fn();
  }, delay);
  timers.add(timer);
}

function showResult(scenario) {
  result.replaceChildren();
  const title = document.createElement("p");
  title.textContent = scenario.title;
  result.append(title);
  for (const [label, description] of scenario.rows) {
    const row = document.createElement("div");
    const code = document.createElement("code");
    const detail = document.createElement("span");
    code.textContent = label;
    detail.textContent = description;
    row.append(code, detail);
    result.append(row);
  }
  result.hidden = false;
}

function setRunControl(value) {
  running = value;
  runButton.dataset.running = String(value);
  runButton.innerHTML = value
    ? 'Stop <span aria-hidden="true">■</span>'
    : 'Run <svg class="icon small" aria-hidden="true"><use href="#arrow-right"/></svg>';
  runButton.setAttribute("aria-label", value ? "Stop demo" : "Run selected demo");
  scenarioSelect.disabled = value;
}

function setState(text) {
  demoState.textContent = text;
  get("map-state").textContent = text;
}

function renderScenario() {
  clearRunTimers();
  setRunControl(false);
  const scenario = scenarios[scenarioSelect.value];
  get("demo-prompt").textContent = scenario.prompt;
  get("map-prompt").textContent = "“" + scenario.prompt + "”";
  get("demo-intro").textContent = scenario.intro;
  get("trace-code").textContent = scenario.code;
  get("trace-indicator").textContent = "✓";
  get("trace-meta").textContent = "3 tool calls";
  get("activity-phase").textContent = "Complete";
  get("activity-calls").textContent = "3";
  showResult(scenario);
  setState("Ready when you are ●");
  transcript.scrollTop = 0;
}

function stopDemo() {
  if (!running) return;
  clearRunTimers();
  setRunControl(false);
  get("trace-indicator").textContent = "–";
  get("trace-meta").textContent = "stopped";
  get("activity-phase").textContent = "Stopped";
  get("map-router").classList.remove("is-active");
  get("map-code").classList.remove("is-active");
  setState("Stopped · run to replay");
}

function runDemo() {
  if (running) {
    stopDemo();
    return;
  }
  const scenario = scenarios[scenarioSelect.value];
  clearRunTimers();
  get("demo-prompt").textContent = scenario.prompt;
  get("map-prompt").textContent = "“" + scenario.prompt + "”";
  get("demo-intro").textContent = scenario.intro;
  get("trace-code").textContent = scenario.code;
  get("demo-trace").open = false;
  result.hidden = true;
  get("trace-indicator").textContent = "…";
  get("trace-meta").textContent = "starting";
  get("activity-calls").textContent = "0";
  get("activity-phase").textContent = "Starting";
  transcript.scrollTop = 0;
  setRunControl(true);
  get("map-router").classList.add("is-active");
  get("map-code").classList.remove("is-active");
  setState("Running the sample…");

  const shortMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  const interval = shortMotion ? 120 : 650;
  scenario.phases.forEach((phase, index) => {
    later(() => {
      get("activity-phase").textContent = phase;
      get("map-router").classList.remove("is-active");
      get("map-code").classList.add("is-active");
      get("activity-calls").textContent = String(index + 1);
      get("trace-meta").textContent = (index + 1) + " / 3 calls";
      setState(phase + "…");
    }, interval * (index + 1));
  });
  later(() => {
    get("trace-indicator").textContent = "✓";
    get("trace-meta").textContent = "3 tool calls";
    get("activity-phase").textContent = "Complete";
    setRunControl(false);
    get("map-code").classList.remove("is-active");
    showResult(scenario);
    setState("Demo complete ✓");
    transcript.scrollTo({ top: transcript.scrollHeight, behavior: shortMotion ? "instant" : "smooth" });
  }, interval * 4);
}

get("demo-form").addEventListener("submit", (event) => {
  event.preventDefault();
  runDemo();
});
scenarioSelect.addEventListener("change", renderScenario);
get("hero-demo").addEventListener("click", () => {
  get("demo").scrollIntoView({ behavior: window.matchMedia("(prefers-reduced-motion: reduce)").matches ? "instant" : "smooth", block: "center" });
  if (running) stopDemo();
  runDemo();
  runButton.focus({ preventScroll: true });
});
get("activity-toggle").addEventListener("click", () => {
  const rail = get("activity-rail");
  rail.hidden = !rail.hidden;
  get("activity-toggle").setAttribute("aria-expanded", String(!rail.hidden));
});

function showToast(message) {
  clearTimeout(toastTimer);
  get("toast").textContent = message;
  get("toast").hidden = false;
  toastTimer = setTimeout(() => { get("toast").hidden = true; }, 3500);
}

async function copyText(text) {
  try {
    if (navigator.clipboard && window.isSecureContext) {
      await navigator.clipboard.writeText(text);
    } else {
      const previousFocus = document.activeElement;
      const area = document.createElement("textarea");
      area.value = text;
      area.setAttribute("readonly", "");
      area.style.position = "fixed";
      area.style.left = "-9999px";
      document.body.append(area);
      let copied = false;
      try {
        area.select();
        copied = document.execCommand("copy");
      } finally {
        area.remove();
        previousFocus?.focus({ preventScroll: true });
      }
      if (!copied) throw new Error("Clipboard unavailable");
    }
    showToast("Copied. Your terminal is next.");
  } catch {
    showToast("Couldn't access the clipboard. Select and copy the command instead.");
  }
}

document.querySelectorAll("[data-copy]").forEach((button) => {
  button.addEventListener("click", () => copyText(button.dataset.copy));
});

const setupTabs = [...document.querySelectorAll("[data-setup]")];
function switchSetup(mode, focus = false) {
  setupMode = mode;
  const offline = mode === "offline";
  setupTabs.forEach((tab) => {
    const active = tab.dataset.setup === mode;
    tab.setAttribute("aria-selected", String(active));
    tab.tabIndex = active ? 0 : -1;
    if (active && focus) tab.focus();
  });
  get("setup-code").setAttribute("aria-labelledby", offline ? "offline-tab" : "install-tab");
  get("setup-middle").textContent = offline ? "uv sync" : "uv sync\ncp .env.example .env";
  get("setup-launch").textContent = offline ? "uv run agent-tui --demo" : "uv run agent-tui";
  get("setup-instruction").replaceChildren();
  if (offline) {
    get("setup-instruction").textContent = "After installation, the demo runs without credentials or network requests.";
  } else {
    const env = document.createElement("code");
    env.textContent = ".env";
    get("setup-instruction").append("Add your OPENROUTER_API_KEY to ", env, ".");
  }
  get("key-requirement").textContent = offline ? "No API key needed" : "OpenRouter API key";
  get("setup-note").textContent = offline ? "●  Explore at your own pace." : "●  Ready for your first prompt.";
}

setupTabs.forEach((tab, index) => {
  tab.addEventListener("click", () => switchSetup(tab.dataset.setup));
  tab.addEventListener("keydown", (event) => {
    let target;
    if (event.key === "ArrowRight") target = (index + 1) % setupTabs.length;
    if (event.key === "ArrowLeft") target = (index + setupTabs.length - 1) % setupTabs.length;
    if (event.key === "Home") target = 0;
    if (event.key === "End") target = setupTabs.length - 1;
    if (target !== undefined) {
      event.preventDefault();
      switchSetup(setupTabs[target].dataset.setup, true);
    }
  });
});

get("copy-setup").addEventListener("click", () => {
  const base = "git clone https://github.com/Protocol-Lattice/PyLattice.git\ncd PyLattice\nuv sync\n";
  const commands = setupMode === "offline"
    ? base + "uv run agent-tui --demo"
    : base + "cp .env.example .env\n# Add your OPENROUTER_API_KEY to .env, then launch:\n# uv run agent-tui";
  copyText(commands);
});

const menuButton = document.querySelector(".menu-toggle");
const mobileNav = get("mobile-nav");
function closeMenu(restoreFocus = false) {
  const wasOpen = !mobileNav.hidden;
  mobileNav.hidden = true;
  menuButton.setAttribute("aria-expanded", "false");
  menuButton.setAttribute("aria-label", "Open navigation");
  if (wasOpen && restoreFocus) menuButton.focus();
}
menuButton.addEventListener("click", () => {
  const opening = mobileNav.hidden;
  mobileNav.hidden = !opening;
  menuButton.setAttribute("aria-expanded", String(opening));
  menuButton.setAttribute("aria-label", opening ? "Close navigation" : "Open navigation");
});
mobileNav.querySelectorAll("a").forEach((link) => {
  link.addEventListener("click", () => {
    closeMenu();
    if (link.hash) {
      const destination = document.querySelector(link.hash);
      if (destination) {
        destination.tabIndex = -1;
        destination.focus({ preventScroll: true });
      }
    }
  });
});
document.addEventListener("keydown", (event) => {
  if (event.key === "Escape") {
    stopDemo();
    closeMenu(true);
  }
});
window.matchMedia("(min-width: 601px)").addEventListener("change", (event) => {
  if (event.matches) closeMenu();
});
