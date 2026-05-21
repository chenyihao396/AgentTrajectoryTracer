const fs = require("fs");
const path = require("path");
const vscode = require("vscode");

function activate(context) {
  context.subscriptions.push(
    vscode.commands.registerCommand("agentTrajectoryTracer.openTrajectory", async (uri) => {
      const filePath = await resolveTrajectoryPath(uri);
      if (!filePath) {
        return;
      }
      openTrajectoryPanel(context, filePath);
    }),
    vscode.commands.registerCommand("agentTrajectoryTracer.openLatest", async () => {
      const folder = vscode.workspace.workspaceFolders?.[0]?.uri.fsPath;
      if (!folder) {
        const filePath = await promptForTrajectoryPath(
          "No workspace folder is open. Select an output/latest/trajectory.json file."
        );
        if (filePath) {
          openTrajectoryPanel(context, filePath);
        }
        return;
      }
      const filePath = path.join(folder, "output", "latest", "trajectory.json");
      if (!fs.existsSync(filePath)) {
        const selected = await promptForTrajectoryPath(`Cannot find ${filePath}. Select trajectory.json manually.`);
        if (selected) {
          openTrajectoryPanel(context, selected);
        }
        return;
      }
      openTrajectoryPanel(context, filePath);
    })
  );
}

async function resolveTrajectoryPath(uri) {
  if (uri?.fsPath) {
    return uri.fsPath;
  }

  const active = vscode.window.activeTextEditor?.document?.uri?.fsPath;
  if (active && path.basename(active) === "trajectory.json") {
    return active;
  }

  return promptForTrajectoryPath("Select trajectory.json");
}

async function promptForTrajectoryPath(title) {
  if (title) {
    vscode.window.showInformationMessage(title);
  }
  const selected = await vscode.window.showOpenDialog({
    canSelectMany: false,
    filters: { JSON: ["json"] },
    title: title || "Select trajectory.json",
  });
  return selected?.[0]?.fsPath;
}

function openTrajectoryPanel(context, filePath) {
  let data;
  try {
    data = JSON.parse(fs.readFileSync(filePath, "utf8"));
  } catch (error) {
    vscode.window.showErrorMessage(`Failed to read trajectory: ${error.message}`);
    return;
  }

  if (!data || !data.trace || !Array.isArray(data.observations)) {
    vscode.window.showErrorMessage("This file does not look like an AgentTrajectoryTracer trajectory.json.");
    return;
  }

  const panel = vscode.window.createWebviewPanel(
    "agentTrajectoryTracer",
    `Trajectory: ${data.trace.name || path.basename(path.dirname(filePath))}`,
    vscode.ViewColumn.Beside,
    {
      enableScripts: true,
      retainContextWhenHidden: true,
    }
  );

  panel.webview.html = renderHtml(panel.webview, data, filePath);
}

function renderHtml(webview, data, filePath) {
  const nonce = makeNonce();
  const state = buildViewModel(data, filePath);
  const payload = JSON.stringify(state).replace(/</g, "\\u003c");

  return `<!doctype html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta http-equiv="Content-Security-Policy" content="default-src 'none'; style-src 'unsafe-inline'; script-src 'nonce-${nonce}';" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Agent Trajectory</title>
  <style>
    :root {
      --bg: #f6f7f9;
      --panel: #ffffff;
      --text: #17202a;
      --muted: #687385;
      --line: #d7dce3;
      --accent: #2563eb;
      --agent: #0f766e;
      --generation: #7c3aed;
      --tool: #b45309;
      --event: #475569;
      --error: #dc2626;
      --code: #111827;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      background: var(--bg);
      color: var(--text);
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      font-size: 13px;
    }
    .app {
      display: grid;
      grid-template-columns: minmax(520px, 1fr) minmax(360px, 42vw);
      height: 100vh;
      min-width: 900px;
    }
    .left, .right {
      min-width: 0;
      overflow: auto;
    }
    .left {
      padding: 16px 18px 28px;
      border-right: 1px solid var(--line);
    }
    .right {
      background: var(--panel);
      padding: 16px;
    }
    .topbar {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
      margin-bottom: 14px;
    }
    h1 {
      font-size: 17px;
      line-height: 1.25;
      margin: 0;
      font-weight: 650;
    }
    .meta {
      color: var(--muted);
      font-size: 12px;
      margin-top: 3px;
      word-break: break-all;
    }
    .pills {
      display: flex;
      flex-wrap: wrap;
      gap: 6px;
      justify-content: flex-end;
    }
    .pill {
      border: 1px solid var(--line);
      background: #fff;
      border-radius: 999px;
      padding: 4px 8px;
      color: var(--muted);
      white-space: nowrap;
    }
    .graph {
      position: relative;
      padding: 8px 0 20px;
    }
    .graph-canvas {
      position: relative;
    }
    .edge-svg {
      position: absolute;
      inset: 0;
      width: 100%;
      height: 100%;
      overflow: visible;
      pointer-events: none;
      z-index: 1;
    }
    .arrow-svg {
      position: absolute;
      inset: 0;
      width: 100%;
      height: 100%;
      overflow: visible;
      pointer-events: none;
      z-index: 3;
    }
    .edge-path {
      fill: none;
      stroke: #94a3b8;
      stroke-width: 1.5;
      stroke-linecap: round;
      stroke-linejoin: round;
    }
    .edge-path.tool-output {
      stroke-dasharray: 5 4;
      stroke: #a8b1bf;
    }
    .edge-path.main-edge {
      stroke: #7c8694;
      stroke-width: 1.8;
    }
    .graph-node {
      position: absolute;
      width: 340px;
      z-index: 2;
    }
    .graph-node.child {
      width: 340px;
    }
    .node {
      position: relative;
      border: 1px solid var(--line);
      background: var(--panel);
      border-radius: 8px;
      padding: 10px 11px;
      height: 108px;
      cursor: pointer;
      transition: border-color 120ms ease, box-shadow 120ms ease;
    }
    .node:hover, .node.selected {
      border-color: var(--accent);
      box-shadow: 0 6px 16px rgba(15, 23, 42, 0.12);
    }
    .node.selected {
      outline: 2px solid rgba(37, 99, 235, 0.16);
    }
    .child-node {
      max-width: 620px;
    }
    .node-title {
      display: flex;
      align-items: center;
      gap: 7px;
      font-weight: 650;
      min-width: 0;
    }
    .type {
      color: #fff;
      border-radius: 5px;
      padding: 2px 6px;
      font-size: 10px;
      line-height: 16px;
      letter-spacing: 0;
      flex: none;
    }
    .type.GENERATION { background: var(--generation); }
    .type.AGENT { background: var(--agent); }
    .type.TOOL { background: var(--tool); }
    .type.EVENT, .type.SPAN { background: var(--event); }
    .node-name {
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
    }
    .brief {
      margin-top: 7px;
      color: var(--muted);
      display: -webkit-box;
      -webkit-line-clamp: 2;
      -webkit-box-orient: vertical;
      overflow: hidden;
      line-height: 1.35;
    }
    .node-foot {
      margin-top: 8px;
      display: flex;
      justify-content: space-between;
      gap: 8px;
      color: var(--muted);
      font-size: 11px;
    }
    .status-error {
      color: var(--error);
      font-weight: 650;
    }
    .empty {
      padding: 28px;
      text-align: center;
      color: var(--muted);
      border: 1px dashed var(--line);
      border-radius: 8px;
      background: #fff;
    }
    .diagnostics {
      margin: 0 0 10px;
      padding: 9px 10px;
      border: 1px solid #fde68a;
      border-radius: 8px;
      background: #fffbeb;
      color: #854d0e;
      font-size: 12px;
      line-height: 1.45;
    }
    .detail-header {
      border-bottom: 1px solid var(--line);
      padding-bottom: 12px;
      margin-bottom: 12px;
    }
    .detail-title {
      display: flex;
      align-items: center;
      gap: 8px;
      font-size: 16px;
      font-weight: 680;
      min-width: 0;
    }
    .detail-name {
      min-width: 0;
      overflow-wrap: anywhere;
    }
    .detail-grid {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 8px;
      margin-top: 12px;
    }
    .metric {
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 9px;
      background: #fbfcfe;
    }
    .metric-label {
      color: var(--muted);
      font-size: 11px;
      margin-bottom: 4px;
    }
    .metric-value {
      font-weight: 650;
      overflow-wrap: anywhere;
    }
    .section {
      margin: 16px 0 0;
    }
    .section h2 {
      font-size: 12px;
      text-transform: uppercase;
      color: var(--muted);
      margin: 0 0 6px;
      letter-spacing: 0.04em;
    }
    pre {
      margin: 0;
      padding: 10px;
      background: #0f172a;
      color: #e5e7eb;
      border-radius: 8px;
      overflow: auto;
      max-height: 34vh;
      line-height: 1.45;
      font-size: 12px;
    }
    .reasoning {
      white-space: pre-wrap;
      line-height: 1.45;
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 10px;
      background: #fffaf0;
      max-height: 28vh;
      overflow: auto;
    }
    @media (max-width: 980px) {
      .app { grid-template-columns: 1fr; height: auto; min-width: 0; }
      .left, .right { height: auto; }
      .right { border-top: 1px solid var(--line); }
    }
  </style>
</head>
<body>
  <div class="app">
    <main class="left">
      <div class="topbar">
        <div>
          <h1 id="title"></h1>
          <div class="meta" id="subtitle"></div>
        </div>
        <div class="pills" id="pills"></div>
      </div>
      <div class="graph" id="graph"></div>
    </main>
    <aside class="right" id="details"></aside>
  </div>
  <script nonce="${nonce}">
    const state = ${payload};
    let selectedId = state.observations[0]?.id;

    function esc(value) {
      return String(value ?? "")
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;")
        .replaceAll('"', "&quot;");
    }

    function pretty(value) {
      if (value === undefined || value === null) return "null";
      if (typeof value === "string") return value;
      return JSON.stringify(value, null, 2);
    }

    function render() {
      document.getElementById("title").textContent = state.trace.name || "Agent Trajectory";
      document.getElementById("subtitle").textContent = state.filePath;
      document.getElementById("pills").innerHTML = [
        ["Status", state.trace.status],
        ["Observations", state.observations.length],
        ["Started", state.trace.timestamp]
      ].map(([k, v]) => '<span class="pill">' + esc(k) + ': ' + esc(v) + '</span>').join("");
      renderGraph();
      renderDetails();
    }

    function renderGraph() {
      const graph = document.getElementById("graph");
      if (!state.roots.length) {
        graph.innerHTML = '<div class="empty">No observations found.</div>';
        return;
      }
      const layout = buildGraphLayout();
      graph.innerHTML =
        diagnosticsHtml() +
        '<div class="graph-canvas" style="width:' + layout.width + 'px;height:' + layout.height + 'px">' +
          edgeSvgHtml(layout) +
          layout.nodes.map(item =>
            '<div class="graph-node ' + (item.role === "child" ? "child" : "root") + '" style="left:' + item.x + 'px;top:' + item.y + 'px">' +
              nodeHtml(item.obs, item.role === "child" ? "child-node" : "main-node") +
            '</div>'
          ).join("") +
          arrowSvgHtml(layout) +
        '</div>';
      for (const el of graph.querySelectorAll(".node")) {
        el.addEventListener("click", () => {
          selectedId = el.dataset.id;
          renderGraph();
          renderDetails();
        });
      }
    }

    function diagnosticsHtml() {
      if (!state.diagnostics || !state.diagnostics.length) return "";
      return '<div class="diagnostics">' +
        state.diagnostics.slice(0, 4).map(item => '<div>' + esc(item) + '</div>').join("") +
        (state.diagnostics.length > 4 ? '<div>...' + (state.diagnostics.length - 4) + ' more layout warnings</div>' : "") +
        '</div>';
    }

    function buildGraphLayout() {
      const constants = {
        rootX: 24,
        levelGap: 170,
        top: 12,
        nodeW: 340,
        nodeH: 108,
        minCanvasW: 960,
        rootGap: 52,
        childGap: 18,
        rightPad: 36,
        bottomPad: 24,
        cornerRadius: 10,
        portMargin: 22,
        toolTrunkOffset: 8,
        arrowInset: 0,
      };
      const nodes = [];
      const boxes = {};
      const edges = [];
      const subtreeHeights = {};
      let y = constants.top;
      let maxRight = constants.rootX + constants.nodeW;

      for (let i = 0; i < state.roots.length; i += 1) {
        const root = state.roots[i];
        const height = measureSubtree(root);
        placeTree(root, 0, y);
        y += height + constants.rootGap;
      }

      for (let i = 0; i < state.roots.length - 1; i += 1) {
        edges.push(mainFlowEdge(state.roots[i].id, state.roots[i + 1].id));
      }

      for (const obs of state.observations) {
        const children = state.childrenByParent[obs.id] || [];
        if (!children.length) continue;
        const parent = boxes[obs.id];
        if (!parent) continue;

        let minChildLeft = Infinity;
        for (const child of children) {
          const cb = boxes[child.id];
          if (cb) minChildLeft = Math.min(minChildLeft, cb.x);
        }
        if (!Number.isFinite(minChildLeft) || minChildLeft <= parent.x + parent.w) continue;

        const parentRight = parent.x + parent.w;
        const trunkX = (parentRight + minChildLeft) / 2;
        const usableHeight = parent.h - 2 * constants.portMargin;
        const portSpacing = usableHeight / (children.length + 1);
        const portOffset = Math.max(2, Math.min(7, portSpacing * 0.2));

        children.forEach((child, idx) => {
          const childBox = boxes[child.id];
          if (!childBox) return;
          const portY = parent.y + constants.portMargin + (idx + 1) * portSpacing;

          if (child.type === "TOOL") {
            const inputEnd = { x: childBox.x, y: childBox.y + childBox.h * 0.32 };
            edges.push(makeEdge(
              "tool-input",
              obs.id,
              child.id,
              roundedOrthogonalPath(
                parentRight,
                portY - portOffset,
                childBox.x - constants.arrowInset,
                childBox.y + childBox.h * 0.32,
                trunkX - constants.toolTrunkOffset,
                constants.cornerRadius
              ),
              { x: trunkX - constants.toolTrunkOffset, y: inputEnd.y },
              inputEnd
            ));
            const outputEnd = { x: parentRight, y: portY + portOffset };
            edges.push(makeEdge(
              "tool-output",
              child.id,
              obs.id,
              roundedOrthogonalPath(
                childBox.x,
                childBox.y + childBox.h * 0.68,
                parentRight + constants.arrowInset,
                portY + portOffset,
                trunkX + constants.toolTrunkOffset,
                constants.cornerRadius
              ),
              { x: trunkX + constants.toolTrunkOffset, y: outputEnd.y },
              outputEnd
            ));
          } else {
            const childEnd = { x: childBox.x, y: childBox.y + childBox.h / 2 };
            edges.push(makeEdge(
              "child",
              obs.id,
              child.id,
              roundedOrthogonalPath(
                parentRight,
                portY,
                childBox.x - constants.arrowInset,
                childBox.y + childBox.h / 2,
                trunkX,
                constants.cornerRadius
              ),
              { x: trunkX, y: childEnd.y },
              childEnd
            ));
          }
        });
      }

      return {
        nodes,
        edges,
        width: Math.max(constants.minCanvasW, maxRight + constants.rightPad),
        height: Math.max(240, y - constants.rootGap + constants.bottomPad),
      };

      function measureSubtree(obs) {
        if (subtreeHeights[obs.id] !== undefined) {
          return subtreeHeights[obs.id];
        }
        const children = state.childrenByParent[obs.id] || [];
        if (!children.length) {
          subtreeHeights[obs.id] = constants.nodeH;
          return constants.nodeH;
        }
        const childrenHeight = children.reduce((sum, child, index) => {
          return sum + measureSubtree(child) + (index ? constants.childGap : 0);
        }, 0);
        const height = Math.max(constants.nodeH, childrenHeight);
        subtreeHeights[obs.id] = height;
        return height;
      }

      function placeTree(obs, depth, top) {
        const children = state.childrenByParent[obs.id] || [];
        const subtreeHeight = subtreeHeights[obs.id] ?? measureSubtree(obs);
        const x = constants.rootX + depth * (constants.nodeW + constants.levelGap);
        const ny = top + subtreeHeight / 2 - constants.nodeH / 2;
        addNode(obs, depth === 0 ? "root" : "child", x, ny, constants.nodeW, constants.nodeH);

        let childTop = top;
        for (const child of children) {
          const childHeight = subtreeHeights[child.id] ?? measureSubtree(child);
          placeTree(child, depth + 1, childTop);
          childTop += childHeight + constants.childGap;
        }
      }

      function addNode(obs, role, x, ny, w, h) {
        const box = { x, y: ny, w, h, depth: role === "root" ? 0 : 1 };
        boxes[obs.id] = box;
        maxRight = Math.max(maxRight, x + w);
        nodes.push({ obs, role, x, y: ny, w, h, depth: box.depth });
      }

      function mainFlowEdge(fromId, toId) {
        const from = boxes[fromId];
        const to = boxes[toId];
        const sx = from.x + from.w / 2;
        const sy = from.y + from.h;
        const tx = to.x + to.w / 2;
        const ty = to.y - constants.arrowInset;
        let path;
        if (Math.abs(sx - tx) < 1) {
          path = ['M', sx, sy, 'L', tx, ty].join(' ');
        } else {
          path = roundedZigzagPath(sx, sy, tx, ty, constants.cornerRadius);
        }
        return makeEdge("main", fromId, toId, path, { x: tx, y: sy }, { x: tx, y: ty });
      }
    }

    function makeEdge(kind, fromId, toId, path, arrowFrom, arrowTip) {
      return { kind, fromId, toId, path, arrowFrom, arrowTip };
    }

    function roundedOrthogonalPath(sx, sy, tx, ty, trunkX, radius) {
      if (Math.abs(sy - ty) < 1) return ['M', sx, sy, 'L', tx, ty].join(' ');
      if (Math.abs(sx - tx) < 1) return ['M', sx, sy, 'L', sx, ty].join(' ');

      const xDir1 = Math.sign(trunkX - sx);
      const yDir = Math.sign(ty - sy);
      const xDir2 = Math.sign(tx - trunkX);

      if (xDir1 === 0) {
        const r0 = Math.min(radius, Math.abs(ty - sy) / 2, Math.abs(tx - sx));
        return [
          'M', sx, sy,
          'L', sx, ty - yDir * r0,
          'Q', sx, ty, sx + xDir2 * r0, ty,
          'L', tx, ty
        ].join(' ');
      }
      if (xDir2 === 0) {
        const r0 = Math.min(radius, Math.abs(ty - sy) / 2, Math.abs(tx - sx));
        return [
          'M', sx, sy,
          'L', tx - xDir1 * r0, sy,
          'Q', tx, sy, tx, sy + yDir * r0,
          'L', tx, ty
        ].join(' ');
      }

      const r = Math.min(
        radius,
        Math.abs(trunkX - sx),
        Math.abs(trunkX - tx),
        Math.abs(ty - sy) / 2
      );

      return [
        'M', sx, sy,
        'L', trunkX - xDir1 * r, sy,
        'Q', trunkX, sy, trunkX, sy + yDir * r,
        'L', trunkX, ty - yDir * r,
        'Q', trunkX, ty, trunkX + xDir2 * r, ty,
        'L', tx, ty
      ].join(' ');
    }

    function roundedZigzagPath(sx, sy, tx, ty, radius) {
      const midY = (sy + ty) / 2;
      const xDir = Math.sign(tx - sx);
      const y1Dir = Math.sign(midY - sy);
      const y2Dir = Math.sign(ty - midY);
      const r = Math.min(
        radius,
        Math.abs(tx - sx) / 2,
        Math.abs(midY - sy),
        Math.abs(ty - midY)
      );
      return [
        'M', sx, sy,
        'L', sx, midY - y1Dir * r,
        'Q', sx, midY, sx + xDir * r, midY,
        'L', tx - xDir * r, midY,
        'Q', tx, midY, tx, midY + y2Dir * r,
        'L', tx, ty
      ].join(' ');
    }

    function edgeSvgHtml(layout) {
      const paths = layout.edges.map(edge => {
        let klass = "edge-path";
        if (edge.kind === "tool-output") klass += " tool-output";
        else if (edge.kind === "tool-input") klass += " tool-input";
        else if (edge.kind === "main") klass += " main-edge";
        else if (edge.kind === "child") klass += " child-edge";
        return '<path class="' + klass + '" d="' + esc(edge.path) + '"></path>';
      }).join("");
      return '<svg class="edge-svg" viewBox="0 0 ' + layout.width + ' ' + layout.height + '" preserveAspectRatio="none">' +
        paths +
        '</svg>';
    }

    function arrowSvgHtml(layout) {
      const arrows = layout.edges.map(edge => arrowPolygonHtml(edge)).join("");
      return '<svg class="arrow-svg" viewBox="0 0 ' + layout.width + ' ' + layout.height + '" preserveAspectRatio="none">' +
        arrows +
        '</svg>';
    }

    function arrowPolygonHtml(edge) {
      const tip = edge.arrowTip;
      const from = edge.arrowFrom;
      if (!tip || !from) return "";

      const dx = tip.x - from.x;
      const dy = tip.y - from.y;
      const length = Math.sqrt(dx * dx + dy * dy);
      if (!length) return "";

      const ux = dx / length;
      const uy = dy / length;
      const px = -uy;
      const py = ux;
      const arrowLength = 11;
      const arrowHalfWidth = 5;
      const baseX = tip.x - ux * arrowLength;
      const baseY = tip.y - uy * arrowLength;
      const points = [
        [tip.x, tip.y],
        [baseX + px * arrowHalfWidth, baseY + py * arrowHalfWidth],
        [baseX - px * arrowHalfWidth, baseY - py * arrowHalfWidth],
      ].map(([x, y]) => x.toFixed(2) + "," + y.toFixed(2)).join(" ");
      const fill = edge.kind === "tool-output" ? "#94a3b8" : "#475569";
      return '<polygon points="' + points + '" fill="' + fill + '"></polygon>';
    }

    function nodeHtml(obs, extraClass) {
      const selected = obs.id === selectedId ? " selected" : "";
      const status = obs.level === "ERROR" ? '<span class="status-error">ERROR</span>' : esc(obs.level || "DEFAULT");
      return '<div class="node ' + extraClass + selected + '" data-id="' + esc(obs.id) + '">' +
        '<div class="node-title"><span class="type ' + esc(obs.type) + '">' + esc(obs.type) + '</span><span class="node-name">' + esc(obs.shortName) + '</span></div>' +
        '<div class="brief">' + esc(obs.brief) + '</div>' +
        '<div class="node-foot"><span>' + esc(obs.durationLabel) + '</span><span>' + status + '</span></div>' +
        '</div>';
    }

    function renderDetails() {
      const obs = state.byId[selectedId] || state.observations[0];
      const details = document.getElementById("details");
      if (!obs) {
        details.innerHTML = '<div class="empty">Select an observation.</div>';
        return;
      }
      details.innerHTML =
        '<div class="detail-header">' +
          '<div class="detail-title"><span class="type ' + esc(obs.type) + '">' + esc(obs.type) + '</span><span class="detail-name">' + esc(obs.name || obs.id) + '</span></div>' +
          '<div class="meta">id: ' + esc(obs.id) + '</div>' +
          '<div class="meta">parent: ' + esc(obs.parentLabel || "None") + '</div>' +
          '<div class="detail-grid">' +
            metric("Model", obs.model || "None") +
            metric("Latency", obs.durationLabel) +
            metric("Start", obs.startTime || "None") +
            metric("End", obs.endTime || "None") +
          '</div>' +
        '</div>' +
        section("Input", '<pre>' + esc(pretty(obs.input)) + '</pre>') +
        (obs.reasoning ? section("Reasoning", '<div class="reasoning">' + esc(obs.reasoning) + '</div>') : "") +
        section("Output", '<pre>' + esc(pretty(obs.output)) + '</pre>') +
        section("Token Usage", '<pre>' + esc(pretty(obs.usageDetails || {})) + '</pre>') +
        section("Time Usage", '<pre>' + esc(pretty(obs.timeUsage)) + '</pre>') +
        section("Metadata", '<pre>' + esc(pretty(obs.metadata || {})) + '</pre>');
    }

    function metric(label, value) {
      return '<div class="metric"><div class="metric-label">' + esc(label) + '</div><div class="metric-value">' + esc(value) + '</div></div>';
    }

    function section(title, body) {
      return '<section class="section"><h2>' + esc(title) + '</h2>' + body + '</section>';
    }

    render();
  </script>
</body>
</html>`;
}

function buildViewModel(data, filePath) {
  const diagnostics = [];
  const observations = makeUniqueObservations(data.observations || [], diagnostics);
  const byId = Object.fromEntries(observations.map((obs) => [obs.id, obs]));
  const firstIdByOriginalId = new Map();
  for (const obs of observations) {
    if (!firstIdByOriginalId.has(obs.originalId)) {
      firstIdByOriginalId.set(obs.originalId, obs.id);
    }
  }

  for (const obs of observations) {
    const rawParentId = obs.originalParentObservationId;
    const parentId = rawParentId ? firstIdByOriginalId.get(rawParentId) : null;
    if (rawParentId && !parentId) {
      diagnostics.push(`Missing parent ${rawParentId} for ${obs.originalId}; rendering it as a root.`);
    }
    if (parentId && parentId === obs.id) {
      diagnostics.push(`Self-parent edge on ${obs.originalId} was ignored.`);
    }
    obs.parentObservationId = parentId && parentId !== obs.id ? parentId : null;
  }

  breakCycles(observations, byId, diagnostics);

  for (const obs of observations) {
    const parent = byId[obs.parentObservationId];
    obs.parentLabel = parent ? `${parent.type}: ${parent.shortName}` : obs.parentObservationId || null;
  }

  const childrenByParent = {};
  for (const obs of observations) {
    if (obs.parentObservationId && byId[obs.parentObservationId]) {
      if (!childrenByParent[obs.parentObservationId]) {
        childrenByParent[obs.parentObservationId] = [];
      }
      childrenByParent[obs.parentObservationId].push(obs);
    }
  }

  const roots = observations.filter((obs) => !obs.parentObservationId || !byId[obs.parentObservationId]);
  roots.sort(compareObservationTime);
  for (const children of Object.values(childrenByParent)) {
    children.sort(compareObservationTime);
  }

  return {
    filePath,
    trace: data.trace || {},
    observations,
    byId,
    roots,
    childrenByParent,
    diagnostics,
  };
}

function makeUniqueObservations(rawObservations, diagnostics) {
  const seen = new Map();
  return rawObservations.map((rawObs, index) => {
    const originalId = String(rawObs.id || `observation-${index + 1}`);
    const count = seen.get(originalId) || 0;
    seen.set(originalId, count + 1);
    const id = count ? `${originalId}#${count + 1}` : originalId;
    if (count) {
      diagnostics.push(`Duplicate observation id ${originalId}; renamed duplicate to ${id} for rendering.`);
    }
    return normalizeObservation({
      ...rawObs,
      id,
      originalId,
      originalParentObservationId: rawObs.parentObservationId ? String(rawObs.parentObservationId) : null,
    });
  });
}

function breakCycles(observations, byId, diagnostics) {
  const color = new Map();

  for (const obs of observations) {
    visit(obs);
  }

  function visit(obs) {
    const state = color.get(obs.id);
    if (state === "done") return;
    if (state === "visiting") return;

    color.set(obs.id, "visiting");

    const parent = byId[obs.parentObservationId];
    if (parent) {
      const parentState = color.get(parent.id);
      if (parentState === "visiting") {
        obs.parentObservationId = null;
        diagnostics.push(`Cycle detected at ${obs.originalId}; one parent edge was ignored so the graph can be rendered.`);
      } else {
        visit(parent);
      }
    }

    color.set(obs.id, "done");
  }
}

function normalizeObservation(obs) {
  const latency = typeof obs.latency === "number" ? obs.latency : null;
  const durationLabel = latency === null ? "N/A" : `${latency.toFixed(latency < 1 ? 3 : 2)}s`;
  return {
    ...obs,
    shortName: shortName(obs),
    brief: brief(obs),
    reasoning: reasoning(obs),
    durationLabel,
    timeUsage: {
      startTime: obs.startTime,
      endTime: obs.endTime,
      latencySeconds: obs.latency,
    },
  };
}

function shortName(obs) {
  if (obs.type === "TOOL") {
    return toolShortName(obs);
  }
  if (obs.type === "GENERATION") {
    return obs.name || obs.model || "llm";
  }
  if (obs.type === "EVENT" && obs.name === "claude.thinking") {
    return "claude.thinking";
  }
  return obs.name || obs.id;
}

function toolShortName(obs) {
  const input = obs.input || {};
  const bits = [];
  if (input.command) bits.push(input.command);
  if (input.pattern) bits.push(input.pattern);
  if (input.query) bits.push(input.query);
  if (input.file_path) bits.push(input.file_path);
  if (input.location) bits.push(input.location);
  const suffix = bits.length ? ` ${bits.join(" ")}` : "";
  return `${obs.name || "tool"}${suffix}`;
}

function brief(obs) {
  if (obs.type === "GENERATION") {
    const output = obs.output || {};
    return output.content || output.reasoning_content || obs.name || "LLM generation";
  }
  if (obs.type === "TOOL") {
    return toolShortName(obs);
  }
  if (obs.name === "claude.thinking") {
    return reasoning(obs) || "Reasoning event";
  }
  if (typeof obs.output === "string") return obs.output;
  if (Array.isArray(obs.output)) return obs.output.join(" ");
  if (obs.output && typeof obs.output === "object") {
    if (obs.output.answer) return obs.output.answer;
    if (obs.output.content) return obs.output.content;
    if (obs.output.thinking) return obs.output.thinking;
  }
  return obs.name || obs.type || obs.id;
}

function reasoning(obs) {
  if (obs.output && typeof obs.output === "object") {
    if (typeof obs.output.reasoning_content === "string") return obs.output.reasoning_content;
    if (typeof obs.output.thinking === "string") return obs.output.thinking;
  }
  return "";
}

function compareObservationTime(a, b) {
  return String(a.startTime || "").localeCompare(String(b.startTime || ""));
}

function makeNonce() {
  const chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789";
  let text = "";
  for (let i = 0; i < 32; i += 1) {
    text += chars.charAt(Math.floor(Math.random() * chars.length));
  }
  return text;
}

function deactivate() {}

module.exports = { activate, deactivate };
