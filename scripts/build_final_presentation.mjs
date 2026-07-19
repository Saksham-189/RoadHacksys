import fs from "node:fs/promises";
import path from "node:path";
import {
  Presentation,
  PresentationFile,
} from "@oai/artifact-tool";

const ROOT = process.cwd();
const OUTPUT = path.join(ROOT, "reports", "final_submission");
const QA =
  process.env.PRESENTATION_QA_DIR ||
  path.join(OUTPUT, "_qa_slides");
const SUMMARY = JSON.parse(
  await fs.readFile(path.join(OUTPUT, "submission_summary.json"), "utf8"),
);
const scenarioRows = await readCsv(
  path.join(ROOT, "runs", "part4", "scenario_scoreboard.csv"),
);

const W = 1280;
const H = 720;
const M = 42;
const INK = "#101417";
const MUTED = "#56636C";
const PANEL = "#EDEDED";
const RULE = "#B8BCC4";
const ORANGE = "#FF6B35";
const GREEN = "#2F6F4E";
const RED = "#B93A32";
const FONT = "Arial";

async function readCsv(filePath) {
  const text = await fs.readFile(filePath, "utf8");
  const lines = text.trim().split(/\r?\n/);
  const headers = lines[0].split(",");
  return lines.slice(1).map((line) => {
    const values = line.split(",");
    return Object.fromEntries(headers.map((header, index) => [header, values[index]]));
  });
}

async function imageBytes(filePath) {
  const bytes = await fs.readFile(filePath);
  return bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength);
}

function addText(slide, text, x, y, width, height, options = {}) {
  const shape = slide.shapes.add({
    geometry: "textbox",
    name: options.name || "text",
    position: { left: x, top: y, width, height },
    fill: options.fill || "none",
    line: options.line || { style: "solid", fill: "none", width: 0 },
  });
  shape.text = text;
  shape.text.style = {
    fontFamily: FONT,
    fontSize: options.fontSize || 22,
    bold: options.bold || false,
    color: options.color || INK,
    alignment: options.alignment || "left",
    verticalAlignment: options.verticalAlignment || "top",
  };
  return shape;
}

function addTitle(slide, title, section, number) {
  addText(slide, section.toUpperCase(), M, 30, 280, 22, {
    fontSize: 13,
    bold: true,
    color: MUTED,
    name: `section-${number}`,
  });
  addText(slide, title, M, 57, 1120, 64, {
    fontSize: 39,
    bold: true,
    name: `title-${number}`,
  });
  addText(slide, String(number).padStart(2, "0"), 1190, 665, 48, 22, {
    fontSize: 13,
    color: MUTED,
    alignment: "right",
    name: `footer-${number}`,
  });
}

function addPanel(slide, x, y, width, height, fill = PANEL, name = "panel") {
  return slide.shapes.add({
    geometry: "rect",
    name,
    position: { left: x, top: y, width, height },
    fill,
    line: { style: "solid", fill: "none", width: 0 },
  });
}

function addRule(slide, x, y, width, height = 2, fill = RULE, name = "rule") {
  return slide.shapes.add({
    geometry: "rect",
    name,
    position: { left: x, top: y, width, height },
    fill,
    line: { style: "solid", fill: "none", width: 0 },
  });
}

function addMetric(slide, value, label, x, y, width, options = {}) {
  addPanel(slide, x, y, width, options.height || 180, options.fill || PANEL, options.name);
  addText(slide, value, x + 22, y + 20, width - 44, 68, {
    fontSize: options.valueSize || 44,
    bold: true,
    color: options.valueColor || INK,
  });
  addText(slide, label, x + 22, y + 105, width - 44, 58, {
    fontSize: 18,
    color: MUTED,
  });
}

async function addImage(slide, filePath, x, y, width, height, fit = "contain", alt = "") {
  slide.images.add({
    blob: await imageBytes(filePath),
    contentType: "image/png",
    alt,
    fit,
    position: { left: x, top: y, width, height },
  });
}

function addBullet(slide, title, body, x, y, width, color = INK) {
  addPanel(slide, x, y + 5, 12, 12, color, `bullet-${title}`);
  addText(slide, title, x + 28, y, width - 28, 28, {
    fontSize: 20,
    bold: true,
  });
  addText(slide, body, x + 28, y + 34, width - 28, 70, {
    fontSize: 17,
    color: MUTED,
  });
}

const deck = Presentation.create({ slideSize: { width: W, height: H } });

// 1. Project-native cover built around the measured relative-flow network.
{
  const slide = deck.slides.add();
  slide.background.fill = "#FFFFFF";
  await addImage(
    slide,
    path.join(ROOT, "runs", "part3", "relative_flow_map.png"),
    650,
    0,
    630,
    720,
    "cover",
    "Relative urban traffic-load network",
  );
  addText(slide, "ISRO HACKATHON", 52, 46, 300, 28, {
    fontSize: 15,
    bold: true,
    color: ORANGE,
  });
  addText(slide, "Route\nResilience", 52, 145, 560, 190, {
    fontSize: 72,
    bold: true,
  });
  addText(
    slide,
    "Occlusion-robust road extraction and graph-theoretic criticality analysis for urban mobility",
    52,
    382,
    520,
    118,
    { fontSize: 25, color: MUTED },
  );
  addRule(slide, 52, 548, 500, 3, ORANGE, "cover-rule");
  addText(
    slide,
    "Satellite imagery  →  road graph  →  critical infrastructure  →  resilience",
    52,
    578,
    540,
    70,
    { fontSize: 18, bold: true },
  );
}

// 2. Four linked capabilities.
{
  const slide = deck.slides.add();
  slide.background.fill = "#FFFFFF";
  addTitle(slide, "The brief demands one connected decision pipeline", "System", 2);
  addRule(slide, 110, 315, 1060, 3, RULE, "process-line");
  const items = [
    ["01", "Recover roads", "Segment RGBN imagery despite clouds, trees, shadows and vehicles."],
    ["02", "Build the graph", "Trace centre lines and heal only plausible road gaps."],
    ["03", "Find gatekeepers", "Combine topology, capacity, demand and rerouting damage."],
    ["04", "Simulate failure", "Measure access loss, detours, burden and resilience."],
  ];
  items.forEach((item, index) => {
    const x = 42 + index * 309;
    addPanel(slide, x, 205, 270, 278, index === 3 ? "#FFF0E8" : PANEL, `stage-${index}`);
    addText(slide, item[0], x + 20, 226, 80, 44, {
      fontSize: 28,
      bold: true,
      color: index === 3 ? ORANGE : MUTED,
    });
    addText(slide, item[1], x + 20, 290, 230, 54, {
      fontSize: 25,
      bold: true,
    });
    addText(slide, item[2], x + 20, 365, 230, 92, {
      fontSize: 17,
      color: MUTED,
    });
  });
  addText(
    slide,
    "Every stage produces evidence consumed by the next; the project is not four isolated models.",
    M,
    540,
    1120,
    50,
    { fontSize: 24, bold: true },
  );
}

// 3. Data and protocol.
{
  const slide = deck.slides.add();
  slide.background.fill = "#FFFFFF";
  addTitle(slide, "The expanded evaluation is geographic, not a random tile split", "Evidence base", 3);
  addMetric(slide, "2,568", "Sentinel-2 RGBN tiles", 42, 185, 270);
  addMetric(slide, "17", "urban and peri-urban AOIs", 351, 185, 270);
  addMetric(slide, "10 m", "spatial resolution", 658, 185, 270);
  addMetric(slide, "304", "held-out test tiles", 968, 185, 270);
  addPanel(slide, 0, 520, W, 200, "#F5F5F5", "data-note");
  addText(
    slide,
    "1,804 train  /  460 validation  /  304 test",
    M,
    555,
    650,
    42,
    { fontSize: 28, bold: true },
  );
  addText(
    slide,
    "Resourcesat remains adapter-ready but is not included in measured claims because no verified LISS-IV scene was available locally.",
    705,
    548,
    500,
    82,
    { fontSize: 18, color: MUTED },
  );
}

// 4. Current Part 1 model.
{
  const slide = deck.slides.add();
  slide.background.fill = "#FFFFFF";
  addTitle(slide, "Pretraining and synthetic occlusion define the current road model", "Part 1", 4);
  await addImage(
    slide,
    path.join(ROOT, "runs", "e013_pretrained_segformer_rgbn", "examples_clean", "bengaluru_edge_00303.png"),
    42,
    140,
    1196,
    330,
    "contain",
    "E013 clean-road segmentation diagnostic",
  );
  addMetric(slide, "0.300", "clean IoU", 42, 505, 220, { height: 130, valueSize: 34 });
  addMetric(slide, "0.462", "clean Dice", 282, 505, 220, { height: 130, valueSize: 34 });
  addMetric(slide, "0.783", "clean recall", 522, 505, 220, { height: 130, valueSize: 34 });
  addMetric(slide, "0.780", "occlusion recall", 762, 505, 220, { height: 130, valueSize: 34 });
  addMetric(slide, "0.493", "expanded-test score", 1002, 505, 236, {
    height: 130,
    valueSize: 34,
    fill: "#FFF0E8",
    valueColor: ORANGE,
  });
}

// 5. Score interpretation.
{
  const slide = deck.slides.add();
  slide.background.fill = "#FFFFFF";
  addTitle(slide, "The 0.573 and 0.493 scores answer different experiments", "Part 1", 5);
  addMetric(slide, "0.573", "E012 historical v1 winner", 74, 185, 470, {
    height: 240,
    valueSize: 64,
  });
  addMetric(slide, "0.493", "E013 current pretrained model", 736, 185, 470, {
    height: 240,
    valueSize: 64,
    fill: "#FFF0E8",
    valueColor: ORANGE,
  });
  addText(slide, "Earlier v1 dataset and scoring protocol", 103, 448, 410, 42, {
    fontSize: 20,
    color: MUTED,
    alignment: "center",
  });
  addText(slide, "Expanded 304-tile held-out protocol", 765, 448, 410, 42, {
    fontSize: 20,
    color: MUTED,
    alignment: "center",
  });
  addRule(slide, 620, 175, 2, 345, RULE, "protocol-divider");
  addText(
    slide,
    "The current threshold favours recall and continuity. It helps graph recovery, but over-segments some dense urban textures.",
    198,
    545,
    884,
    70,
    { fontSize: 24, bold: true, alignment: "center" },
  );
}

// 6. Graph healing.
{
  const slide = deck.slides.add();
  slide.background.fill = "#FFFFFF";
  addTitle(slide, "Graph healing is selected by safety, not raw connectivity", "Part 2", 6);
  await addImage(
    slide,
    path.join(ROOT, "runs", "part2_healing", "part2_comparison.png"),
    42,
    140,
    700,
    490,
    "contain",
    "Comparison of graph-healing experiments",
  );
  addMetric(slide, "96.25%", "route success", 790, 160, 200, { height: 150, valueSize: 36 });
  addMetric(slide, "7.40%", "false bridges", 1010, 160, 200, {
    height: 150,
    valueSize: 36,
    fill: "#EAF4EE",
    valueColor: GREEN,
  });
  addMetric(slide, "63.87%", "routing-node coverage", 790, 340, 420, {
    height: 150,
    valueSize: 42,
    fill: "#FFF0E8",
    valueColor: ORANGE,
  });
  addText(
    slide,
    "The 70% coverage gate was not met safely. Routing therefore uses the largest safe connected component.",
    790,
    525,
    420,
    82,
    { fontSize: 19, bold: true },
  );
}

// 7. Transport model.
{
  const slide = deck.slides.add();
  slide.background.fill = "#FFFFFF";
  addTitle(slide, "Relative capacity and gravity demand create mobility flow", "Part 3", 7);
  await addImage(
    slide,
    path.join(ROOT, "runs", "part3", "relative_flow_map.png"),
    42,
    135,
    720,
    515,
    "contain",
    "Relative assigned-flow map",
  );
  addBullet(slide, "3,345 nodes", "Largest safe routing component", 805, 160, 390, INK);
  addBullet(slide, "3,749 edges", "Relative width, speed and capacity", 805, 275, 390, ORANGE);
  addBullet(slide, "2,000 OD pairs", "Seeded graph-derived gravity demand", 805, 390, 390, GREEN);
  addBullet(slide, "MSA + BPR", "Converged congestion-aware assignment", 805, 505, 390, INK);
}

// 8. Criticality.
{
  const slide = deck.slides.add();
  slide.background.fill = "#FFFFFF";
  addTitle(slide, "Flow-aware ablation exposes failures that degree alone misses", "Part 3", 8);
  await addImage(
    slide,
    path.join(ROOT, "runs", "part3", "node_criticality_map.png"),
    42,
    135,
    745,
    520,
    "contain",
    "Flow-aware node criticality map",
  );
  addMetric(slide, "3900", "top flow-critical node", 840, 170, 345, {
    height: 155,
    valueSize: 48,
  });
  addMetric(slide, "18.18%", "demand disconnected", 840, 355, 345, {
    height: 155,
    valueSize: 48,
    fill: "#FFF0E8",
    valueColor: ORANGE,
  });
  addText(
    slide,
    "The advanced ranking asks what actually happens after removal and rerouting, not only how many roads touch a junction.",
    840,
    550,
    345,
    78,
    { fontSize: 18, color: MUTED },
  );
}

// 9. Scenario engine.
{
  const slide = deck.slides.add();
  slide.background.fill = "#FFFFFF";
  addTitle(slide, "Every disruption follows the same auditable simulation contract", "Part 4", 9);
  addRule(slide, 145, 330, 985, 3, RULE, "scenario-line");
  const steps = [
    ["1", "Resolve actions", "Nodes, edges, capacity, circles or polygons"],
    ["2", "Apply safely", "Copy the baseline; record every graph change"],
    ["3", "Reroute demand", "Preview AON or converged exact MSA"],
    ["4", "Measure impact", "Service, detour, congestion and efficiency"],
    ["5", "Map burden", "Affected cells, OD routes and overloaded links"],
  ];
  steps.forEach((step, index) => {
    const x = 42 + index * 245;
    addPanel(slide, x, 205, 205, 260, index === 4 ? "#FFF0E8" : PANEL, `sim-step-${index}`);
    addText(slide, step[0], x + 18, 222, 45, 40, {
      fontSize: 28,
      bold: true,
      color: index === 4 ? ORANGE : MUTED,
    });
    addText(slide, step[1], x + 18, 285, 169, 52, {
      fontSize: 22,
      bold: true,
    });
    addText(slide, step[2], x + 18, 358, 169, 82, {
      fontSize: 16,
      color: MUTED,
    });
  });
  addText(
    slide,
    "Official resilience = served demand ratio × canonical path resilience",
    160,
    545,
    960,
    52,
    { fontSize: 27, bold: true, alignment: "center" },
  );
}

// 10. Scenario results.
{
  const slide = deck.slides.add();
  slide.background.fill = "#FFFFFF";
  addTitle(slide, "Connectivity loss and congestion reveal different failure modes", "Part 4", 10);
  await addImage(
    slide,
    path.join(ROOT, "runs", "part4", "resilience_comparison.png"),
    42,
    145,
    745,
    500,
    "contain",
    "Resilience comparison for D001 through D009",
  );
  addMetric(slide, "D002", "worst preset scenario", 830, 170, 360, {
    height: 140,
    valueSize: 42,
    fill: "#FFF0E8",
    valueColor: ORANGE,
  });
  addMetric(slide, "19.29%", "estimated demand disconnected", 830, 335, 360, {
    height: 140,
    valueSize: 42,
  });
  addMetric(slide, "625.34%", "D003 travel-time increase", 830, 500, 360, {
    height: 140,
    valueSize: 38,
    fill: "#FCEBE9",
    valueColor: RED,
  });
}

// 11. Affected zones.
{
  const slide = deck.slides.add();
  slide.background.fill = "#FFFFFF";
  addTitle(slide, "D002 concentrates the largest mobility impact in specific zones", "Planning answer", 11);
  await addImage(
    slide,
    path.join(ROOT, "runs", "part4", "scenario_impact_map.png"),
    42,
    130,
    720,
    535,
    "contain",
    "Affected 500 metre mobility-demand zones for D002",
  );
  addText(slide, "If these flood zones fail:", 820, 170, 370, 42, {
    fontSize: 25,
    bold: true,
  });
  addBullet(slide, "19.29% loses service", "Estimated OD demand becomes unreachable.", 820, 245, 370, ORANGE);
  addBullet(slide, "Resilience falls to 0.807", "The service loss dominates the official score.", 820, 365, 370, RED);
  addBullet(slide, "Hotspots are exportable", "Every affected cell, route and rerouted edge is preserved.", 820, 485, 370, GREEN);
}

// 12. Honest close.
{
  const slide = deck.slides.add();
  slide.background.fill = "#FFFFFF";
  addTitle(slide, "The project now answers the planner’s real question", "Conclusion", 12);
  addText(
    slide,
    "Which junction or corridor can fail without paralysing access, and which one cannot?",
    42,
    150,
    1040,
    100,
    { fontSize: 48, bold: true },
  );
  addPanel(slide, 42, 315, 380, 240, "#EAF4EE", "works");
  addText(slide, "What works now", 68, 342, 320, 40, {
    fontSize: 25,
    bold: true,
    color: GREEN,
  });
  addText(
    slide,
    "Satellite-to-graph pipeline\nFlow-aware criticality\nComposable disruption scenarios\nReproducible outputs and tests",
    68,
    405,
    320,
    125,
    { fontSize: 19 },
  );
  addPanel(slide, 450, 315, 380, 240, "#FFF0E8", "limits");
  addText(slide, "What remains honest", 476, 342, 320, 40, {
    fontSize: 25,
    bold: true,
    color: ORANGE,
  });
  addText(
    slide,
    "Relative, not measured, traffic\n63.87% routing coverage\nRecall-oriented over-segmentation\nResourcesat validation pending",
    476,
    405,
    320,
    125,
    { fontSize: 19 },
  );
  addPanel(slide, 858, 315, 380, 240, "#EDEDED", "value");
  addText(slide, "Why it fits the brief", 884, 342, 320, 40, {
    fontSize: 25,
    bold: true,
  });
  addText(
    slide,
    "It turns road visibility into a measurable urban-resilience decision: service loss, detour, burden and affected geography.",
    884,
    405,
    320,
    125,
    { fontSize: 19 },
  );
  addText(slide, "22 automated tests passed", 42, 620, 400, 34, {
    fontSize: 20,
    bold: true,
    color: MUTED,
  });
}

await fs.mkdir(OUTPUT, { recursive: true });
await fs.mkdir(QA, { recursive: true });

for (const [index, slide] of deck.slides.items.entries()) {
  const stem = `slide-${String(index + 1).padStart(2, "0")}`;
  const png = await deck.export({ slide, format: "png", scale: 1 });
  await fs.writeFile(path.join(QA, `${stem}.png`), new Uint8Array(await png.arrayBuffer()));
  const layout = await slide.export({ format: "layout" });
  await fs.writeFile(path.join(QA, `${stem}.layout.json`), await layout.text());
}

const montage = await deck.export({ format: "webp", montage: true, scale: 1 });
await fs.writeFile(
  path.join(QA, "deck-montage.webp"),
  new Uint8Array(await montage.arrayBuffer()),
);
const pptx = await PresentationFile.exportPptx(deck);
await pptx.save(path.join(OUTPUT, "Route_Resilience_Hackathon_Presentation.pptx"));

console.log(`Wrote ${deck.slides.items.length} slides.`);
