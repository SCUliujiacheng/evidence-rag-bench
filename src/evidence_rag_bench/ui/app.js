const form = document.querySelector("#ask-form");
const profile = document.querySelector("#profile");
const result = document.querySelector("#result");
const status = document.querySelector("#status");
const answer = document.querySelector("#answer");
const reason = document.querySelector("#reason");
const latency = document.querySelector("#latency");
const evidence = document.querySelector("#evidence");
const reasonLabels = {
  insufficient_evidence: "Not enough evidence was retrieved",
  citation_validation_failed: "The citation check failed",
};

function element(tag, text) {
  const node = document.createElement(tag);
  node.textContent = text;
  return node;
}

function readableSource(text) {
  return text
    .replace(/!\[([^\]]*)\]\([^)]*\)/g, "$1")
    .replace(/\[([^\]]+)\]\([^)]*\)/g, "$1")
    .replace(/<[^>]+>/g, " ")
    .replace(/\b(?:details|summary|b|strong|em)\s*>/gi, "")
    .replace(/^\s{0,3}#{1,6}\s+/gm, "")
    .replace(/^\s*[-*]\s+/gm, "• ")
    .replace(/^\s*\|?\s*:?-{3,}:?\s*(?:\|\s*:?-{3,}:?\s*)+\|?\s*$/gm, "")
    .replace(/\s+\|\s+/g, " · ")
    .replace(/```[a-z0-9_-]*/gi, "")
    .replace(/[*_`~]/g, "")
    .replace(/[ \t]{2,}/g, " ")
    .replace(/\n{3,}/g, "\n\n")
    .trim();
}

function excerpt(text) {
  return "… " + readableSource(text) + " …";
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  const question = document.querySelector("#question").value.trim();
  const response = await fetch("/v1/ask", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question, top_k: 2, profile: profile.value }),
  });
  const body = await response.json();
  result.hidden = false;
  evidence.replaceChildren();
  if (!response.ok) {
    status.textContent = "Could not run that search";
    answer.textContent = body.detail || "The local service could not process this question.";
    reason.textContent = "";
    latency.textContent = "";
    return;
  }
  status.textContent = body.status === "answer"
    ? "RELEVANT EVIDENCE FOUND — VERIFY BELOW"
    : "NO ANSWER FROM THIS CORPUS";
  answer.textContent = body.answer ? excerpt(body.answer) : "";
  reason.textContent = body.reason
    ? `Reason: ${reasonLabels[body.reason] || body.reason}`
    : "";
  latency.textContent = `Search time: ${body.latency_ms.toFixed(1)} ms`;
  body.evidence.forEach((item) => {
    const card = document.createElement("article");
    card.className = "evidence-card";
    card.append(element("strong", item.chunk_id));
    card.append(element("p", excerpt(item.text)));
    const link = document.createElement("a");
    link.href = item.source_url;
    link.textContent = "Read the source";
    link.target = "_blank";
    link.rel = "noreferrer";
    card.append(link);
    evidence.append(card);
  });
});
