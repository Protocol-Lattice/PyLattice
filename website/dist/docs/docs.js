"use strict";

// Reading and navigation work without JavaScript; enhance examples with copying.
const copyStatus = document.getElementById("copy-status");
let statusTimer;

document.querySelectorAll(".docs-code[data-copy-label]").forEach((block) => {
  const code = block.querySelector("pre code");
  const button = document.createElement("button");
  button.type = "button";
  button.className = "docs-copy";
  button.textContent = "Copy";
  button.setAttribute("aria-label", `Copy ${block.dataset.copyLabel}`);
  block.querySelector(".docs-code-header").append(button);

  button.addEventListener("click", async () => {
    clearTimeout(statusTimer);
    try {
      await navigator.clipboard.writeText(code.textContent);
      copyStatus.textContent = "Copied to clipboard.";
    } catch {
      const selection = window.getSelection();
      const range = document.createRange();
      range.selectNodeContents(code);
      selection.removeAllRanges();
      selection.addRange(range);
      copyStatus.textContent = "Clipboard unavailable. Code selected; use your copy shortcut.";
    }
    statusTimer = setTimeout(() => { copyStatus.textContent = ""; }, 4500);
  });
});
