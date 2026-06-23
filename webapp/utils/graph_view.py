import html
import json
from pathlib import Path


def graph_stats(graph, landmarks=None, changes=None):
    landmarks = landmarks or []
    changes = changes or {}
    moved = changes.get("moved", [])
    changed = changes.get("changed", [])
    objects = graph.get("objects", graph.get("nodes", []))
    relations = graph.get("relations", graph.get("edges", []))
    return {
        "Nodes": len(objects),
        "Edges": len([
            rel for rel in relations
            if isinstance(rel.get("object"), str) and rel.get("object", "").startswith("obj")
            or rel.get("object") is None
        ]),
        "Landmarks": len(landmarks),
        "Detected Changes": sum(len(changes.get(key, [])) for key in ("added", "removed")) + len(moved) + len(changed),
    }


def _normalized_graph(graph):
    if "objects" in graph:
        nodes = [
            {
                "id": obj.get("id"),
                "label": obj.get("unique_name") or obj.get("pred") or obj.get("name") or obj.get("gt") or obj.get("id"),
                "landmark": obj.get("is_landmark", False),
            }
            for obj in graph.get("objects", [])
        ]
        edges = [
            {
                "source": rel.get("subject"),
                "target": rel.get("object"),
                "label": rel.get("predicate", ""),
            }
            for rel in graph.get("relations", [])
            if isinstance(rel.get("object"), str) and rel.get("object", "").startswith("obj")
        ]
        return nodes, edges
    nodes = [
        {
            "id": node.get("id"),
            "label": node.get("label") or node.get("id"),
            "landmark": node.get("is_landmark", False),
        }
        for node in graph.get("nodes", [])
    ]
    edges = [
        {
            "source": edge.get("subject"),
            "target": edge.get("object"),
            "label": edge.get("predicate", ""),
        }
        for edge in graph.get("edges", [])
    ]
    return nodes, edges


def graph_html(graph, height=560):
    nodes, edges = _normalized_graph(graph)
    payload = html.escape(json.dumps({"nodes": nodes, "edges": edges}))
    return f"""
<div class="labgraph-wrap" data-graph="{payload}">
  <div class="labgraph-toolbar">
    <button type="button" data-zoom="out" title="Zoom out">-</button>
    <button type="button" data-zoom="reset" title="Reset view">Reset</button>
    <button type="button" data-zoom="in" title="Zoom in">+</button>
    <button type="button" data-fullscreen="1" title="Fullscreen">Fullscreen</button>
  </div>
  <svg class="labgraph-svg" viewBox="0 0 980 620" role="img"></svg>
</div>
<style>
.labgraph-wrap {{
  height: {height}px;
  border: 1px solid #e2e8f0;
  border-radius: 18px;
  background:
    linear-gradient(rgba(37, 99, 235, 0.055) 1px, transparent 1px),
    linear-gradient(90deg, rgba(6, 182, 212, 0.055) 1px, transparent 1px),
    linear-gradient(180deg, #ffffff, #f8fafc);
  background-size: 30px 30px;
  overflow: hidden;
  position: relative;
  animation: graphFade 360ms ease-out;
  box-shadow: inset 0 0 54px rgba(37, 99, 235, 0.06);
}}
.labgraph-toolbar {{
  position: absolute;
  z-index: 2;
  top: 10px;
  right: 10px;
  display: flex;
  gap: 6px;
}}
.labgraph-toolbar button {{
  border: 1px solid #dbeafe;
  background: rgba(255, 255, 255, 0.94);
  color: #0f172a;
  border-radius: 999px;
  padding: 6px 10px;
  font: 700 12px system-ui, sans-serif;
  cursor: pointer;
  box-shadow: 0 8px 18px rgba(15, 23, 42, 0.10);
}}
.labgraph-svg {{
  width: 100%;
  height: 100%;
  cursor: grab;
}}
.labgraph-node rect {{
  fill: #ffffff;
  stroke: #2563eb;
  stroke-width: 1.5;
  rx: 8;
  filter: drop-shadow(0 10px 16px rgba(37, 99, 235, 0.16));
}}
.labgraph-node.landmark rect {{
  fill: #ecfeff;
  stroke: #06b6d4;
}}
.labgraph-node text, .labgraph-edge-label {{
  font: 700 12px system-ui, sans-serif;
  fill: #0f172a;
}}
.labgraph-edge {{
  stroke: #64748b;
  stroke-width: 1.4;
  marker-end: url(#arrow);
}}
.labgraph-node {{
  opacity: 0;
  animation: nodeIn 420ms ease-out forwards;
}}
@keyframes graphFade {{
  from {{ opacity: 0; transform: translateY(6px); }}
  to {{ opacity: 1; transform: translateY(0); }}
}}
@keyframes nodeIn {{
  from {{ opacity: 0; transform: translateY(10px); }}
  to {{ opacity: 1; }}
}}
</style>
<script>
(function() {{
  const root = document.currentScript.previousElementSibling.previousElementSibling;
  const data = JSON.parse(root.dataset.graph);
  const svg = root.querySelector("svg");
  const NS = "http://www.w3.org/2000/svg";
  let scale = 1;
  let tx = 0;
  let ty = 0;
  let dragging = false;
  let last = null;

  const defs = document.createElementNS(NS, "defs");
  defs.innerHTML = '<marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path d="M 0 0 L 10 5 L 0 10 z" fill="#64748b"/></marker>';
  svg.appendChild(defs);
  const viewport = document.createElementNS(NS, "g");
  svg.appendChild(viewport);

  const columns = Math.max(2, Math.ceil(Math.sqrt(Math.max(1, data.nodes.length))));
  const positions = new Map();
  data.nodes.forEach((node, index) => {{
    const col = index % columns;
    const row = Math.floor(index / columns);
    positions.set(node.id, {{x: 120 + col * 220, y: 90 + row * 130}});
  }});

  function applyTransform() {{
    viewport.setAttribute("transform", `translate(${{tx}} ${{ty}}) scale(${{scale}})`);
  }}

  data.edges.forEach(edge => {{
    const a = positions.get(edge.source);
    const b = positions.get(edge.target);
    if (!a || !b) return;
    const line = document.createElementNS(NS, "line");
    line.setAttribute("class", "labgraph-edge");
    line.setAttribute("x1", a.x);
    line.setAttribute("y1", a.y);
    line.setAttribute("x2", b.x);
    line.setAttribute("y2", b.y);
    viewport.appendChild(line);
    const label = document.createElementNS(NS, "text");
    label.setAttribute("class", "labgraph-edge-label");
    label.setAttribute("x", (a.x + b.x) / 2);
    label.setAttribute("y", (a.y + b.y) / 2 - 6);
    label.textContent = String(edge.label || "").replaceAll("_", " ");
    viewport.appendChild(label);
  }});

  data.nodes.forEach(node => {{
    const p = positions.get(node.id);
    const g = document.createElementNS(NS, "g");
    g.setAttribute("class", `labgraph-node${{node.landmark ? " landmark" : ""}}`);
    g.setAttribute("transform", `translate(${{p.x - 72}} ${{p.y - 24}})`);
    const rect = document.createElementNS(NS, "rect");
    rect.setAttribute("width", 144);
    rect.setAttribute("height", 48);
    const text = document.createElementNS(NS, "text");
    text.setAttribute("x", 72);
    text.setAttribute("y", 29);
    text.setAttribute("text-anchor", "middle");
    text.textContent = String(node.label).slice(0, 22);
    g.appendChild(rect);
    g.appendChild(text);
    g.style.animationDelay = `${{Math.min(420, Array.from(positions.keys()).indexOf(node.id) * 28)}}ms`;
    viewport.appendChild(g);
  }});

  root.querySelector('[data-zoom="in"]').onclick = () => {{ scale = Math.min(2.8, scale + 0.18); applyTransform(); }};
  root.querySelector('[data-zoom="out"]').onclick = () => {{ scale = Math.max(0.45, scale - 0.18); applyTransform(); }};
  root.querySelector('[data-zoom="reset"]').onclick = () => {{ scale = 1; tx = 0; ty = 0; applyTransform(); }};
  root.querySelector('[data-fullscreen]').onclick = () => root.requestFullscreen && root.requestFullscreen();
  svg.addEventListener("mousedown", e => {{ dragging = true; last = [e.clientX, e.clientY]; }});
  window.addEventListener("mouseup", () => dragging = false);
  window.addEventListener("mousemove", e => {{
    if (!dragging || !last) return;
    tx += e.clientX - last[0];
    ty += e.clientY - last[1];
    last = [e.clientX, e.clientY];
    applyTransform();
  }});
  svg.addEventListener("wheel", e => {{
    e.preventDefault();
    scale = Math.max(0.45, Math.min(2.8, scale + (e.deltaY < 0 ? 0.12 : -0.12)));
    applyTransform();
  }}, {{passive: false}});
  applyTransform();
}})();
</script>
"""


def graph_from_json_file(path):
    with open(Path(path)) as handle:
        data = json.load(handle)
    if "nodes" in data or "objects" in data:
        return data
    return None
