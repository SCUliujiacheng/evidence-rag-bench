const form = document.querySelector("#ask-form");
const result = document.querySelector("#result");
const status = document.querySelector("#status");
const answer = document.querySelector("#answer");
const reason = document.querySelector("#reason");
const latency = document.querySelector("#latency");
const evidence = document.querySelector("#evidence");

function element(tag, text) {
  const node = document.createElement(tag);
  node.textContent = text;
  return node;
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  const question = document.querySelector("#question").value.trim();
  const response = await fetch("/v1/ask", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question, top_k: 3 }),
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
  status.textContent = body.status === "answer" ? "ANSWER — SOURCES BELOW" : "NO ANSWER FROM THIS CORPUS";
  answer.textContent = body.answer;
  reason.textContent = body.reason ? `Why: ${body.reason}` : "";
  latency.textContent = `Search time: ${body.latency_ms.toFixed(1)} ms`;
  body.evidence.forEach((item) => {
    const card = document.createElement("article");
    card.className = "evidence-card";
    card.append(element("strong", item.chunk_id));
    card.append(element("p", item.text));
    const link = document.createElement("a");
    link.href = item.source_url;
    link.textContent = "Read the source";
    link.target = "_blank";
    link.rel = "noreferrer";
    card.append(link);
    evidence.append(card);
  });
});
