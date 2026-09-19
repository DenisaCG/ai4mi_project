const pptxgen = require("pptxgenjs");
const path = require("path");

const LOGO_BLACK = path.join(__dirname, "assets/uva_logo.png");
const LOGO_WHITE = path.join(__dirname, "assets/uva_logo_white.png");
const OUT = path.join(__dirname, "midproject_outline.pptx");
const FIG = path.join(__dirname, "..", "figures");

const IMG = {
  fov: { path: path.join(FIG, "profile/scan_geometry/field_of_view.png"), aspect: 1696 / 1152 },
};

// ---- palette: UvA theme (red/orange/purple/black) + UvA's own unused
// hyperlink blue added in place of green ----
const BLACK = "1F1D21";
const WHITE = "FFFFFF";
const RED = "BC0031";
const ORANGE = "E98300";
const PURPLE = "751B68";
const BLUE = "004E92";
const TEXT = "1F1D21";
const GREY = "6B6B6B";
const LINE = "E3E3E3";

// light tints of each accent, for segthor-style callout cards
const TINT = { [RED]: "F7E8EC", [ORANGE]: "FCEEDC", [PURPLE]: "F1E6EF", [BLUE]: "E3EBF3", [BLACK]: "F0F0F0", [GREY]: "F0F0F0" };

const HEAD_FONT = "Times New Roman";
const BODY_FONT = "Times New Roman";

const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE";
pres.author = "AI for Medical Imaging";

function footer(s, label, logo = LOGO_BLACK) {
  s.addImage({ path: logo, x: 0.5, y: 7.05, w: 1.5, h: 0.15 });
  s.addText(label, {
    x: 10.83, y: 7.05, w: 2, h: 0.3, fontSize: 9, color: GREY, align: "right",
    fontFace: BODY_FONT, isTextBox: true, margin: 0,
  });
}

function header(s, { kicker, time, title, accent }) {
  s.addShape(pres.ShapeType.rect, { x: 0, y: 0, w: 0.18, h: 7.5, fill: { color: accent } });
  s.addText(kicker.toUpperCase(), {
    x: 0.7, y: 0.42, w: 9.5, h: 0.3, fontSize: 12, bold: true, color: accent,
    fontFace: BODY_FONT, isTextBox: true, margin: 0, charSpacing: 1.5,
  });
  if (time) {
    s.addText(time, {
      x: 11.4, y: 0.42, w: 1.2, h: 0.3, fontSize: 12, bold: true, color: GREY, align: "right",
      fontFace: BODY_FONT, isTextBox: true, margin: 0,
    });
  }
  s.addText(title, {
    x: 0.7, y: 0.72, w: 11.9, h: 0.9, fontSize: 26, bold: true, color: TEXT,
    fontFace: HEAD_FONT, isTextBox: true, margin: 0,
  });
  s.addShape(pres.ShapeType.rect, { x: 0.72, y: 1.6, w: 1.1, h: 0.035, fill: { color: accent } });
}

// segthor-style callout card: tinted box, colored left edge, label + body
function card(s, { label, body, x, y, w, h, accent }) {
  s.addShape(pres.ShapeType.rect, { x, y, w, h, fill: { color: TINT[accent] || "F0F0F0" }, line: { type: "none" } });
  s.addShape(pres.ShapeType.rect, { x, y, w: 0.06, h, fill: { color: accent } });
  s.addText(label.toUpperCase(), {
    x: x + 0.3, y: y + 0.2, w: w - 0.6, h: 0.3, fontSize: 11, bold: true, color: accent,
    fontFace: BODY_FONT, isTextBox: true, margin: 0, charSpacing: 1,
  });
  s.addText(body, {
    x: x + 0.3, y: y + 0.55, w: w - 0.6, h: h - 0.75, fontFace: BODY_FONT, fontSize: 13.5,
    color: TEXT, isTextBox: true, margin: 0, valign: "top", lineSpacingMultiple: 1.18,
  });
}

// ---------------------------------------------------------------
// Layout: one large figure (segthor-deck style) — image sized to its
// true aspect ratio on the left, structured annotation panel on the right
// ---------------------------------------------------------------
function bigFigureSlide({ pageLabel, time, kicker, title, subtitle, image, boxes, accent = RED }) {
  const s = pres.addSlide();
  s.background = { color: WHITE };
  s.addShape(pres.ShapeType.rect, { x: 0, y: 0, w: 0.18, h: 7.5, fill: { color: accent } });
  s.addText(kicker.toUpperCase(), {
    x: 0.7, y: 0.42, w: 9.5, h: 0.3, fontSize: 12, bold: true, color: accent,
    fontFace: BODY_FONT, isTextBox: true, margin: 0, charSpacing: 1.5,
  });
  if (time) {
    s.addText(time, {
      x: 11.4, y: 0.42, w: 1.2, h: 0.3, fontSize: 12, bold: true, color: GREY, align: "right",
      fontFace: BODY_FONT, isTextBox: true, margin: 0,
    });
  }
  s.addText(title, {
    x: 0.7, y: 0.72, w: 11.9, h: 0.55, fontSize: 26, bold: true, color: TEXT,
    fontFace: HEAD_FONT, isTextBox: true, margin: 0,
  });
  if (subtitle) {
    s.addText(subtitle, {
      x: 0.7, y: 1.28, w: 11.9, h: 0.35, fontSize: 14, italic: true, color: GREY,
      fontFace: BODY_FONT, isTextBox: true, margin: 0,
    });
  }

  // image sized to its true aspect ratio, never stretched, capped to the
  // available box so it can't overflow into the annotation panel or footer.
  // Left edge and top edge both aligned with the title's own margin (x = 0.7).
  const boxX = 0.7, boxY = 1.35, maxW = 9.3, maxH = 5.3;
  let w = maxW, h = w / image.aspect;
  if (h > maxH) { h = maxH; w = h * image.aspect; }
  s.addImage({ path: image.path, x: boxX, y: boxY, w, h, sizing: { type: "contain", w, h } });

  const rx = boxX + w + 0.4, rw = 12.85 - rx;
  const gap = 0.2, boxH = (maxH - gap * (boxes.length - 1)) / boxes.length;
  boxes.forEach((b, i) => {
    card(s, { ...b, x: rx, y: boxY + i * (boxH + gap), w: rw, h: boxH });
  });

  footer(s, pageLabel);
}

// ---------------------------------------------------------------
// Layout: single column bullets, optional callout card on the right
// ---------------------------------------------------------------
function singleColumn({ pageLabel, time, kicker, title, bullets, cardData, image, accent = RED }) {
  const s = pres.addSlide();
  s.background = { color: WHITE };
  header(s, { kicker, time, title, accent });
  const bodyW = cardData || image ? 6.9 : 11.6;
  s.addText(
    bullets.map((b) => ({
      text: b.text,
      options: {
        bullet: b.sub ? false : { code: "25AA", color: accent, indent: 18 },
        indentLevel: b.sub ? 1 : 0,
        fontSize: b.sub ? 14 : 16,
        italic: !!b.italic,
        color: b.sub ? GREY : TEXT,
        bold: !!b.bold,
        breakLine: true,
        paraSpaceAfter: b.sub ? 6 : 14,
      },
    })),
    { x: 0.7, y: 1.95, w: bodyW, h: 4.6, fontFace: BODY_FONT, isTextBox: true, margin: 0, valign: "top", lineSpacingMultiple: 1.18 }
  );
  if (cardData) card(s, { ...cardData, x: 8.05, y: 1.95, w: 4.55, h: cardData.h || 4.3, accent });
  if (image) {
    const x = 8.05, y = 1.95, w = 4.55, h = 4.3;
    s.addImage({ path: image.path, x, y, w, h, sizing: { type: "contain", w, h } });
    if (image.caption) {
      s.addText(image.caption, {
        x, y: y + h + 0.05, w, h: 0.35, fontSize: 11, italic: true, color: GREY,
        fontFace: BODY_FONT, isTextBox: true, margin: 0, align: "center",
      });
    }
  }
  footer(s, pageLabel);
}

// ---------------------------------------------------------------
// Layout: two-panel split for slides with 2 clear sub-topics
// ---------------------------------------------------------------
function twoPanel({ pageLabel, time, kicker, title, left, right, cardData, image, accent = RED }) {
  const s = pres.addSlide();
  s.background = { color: WHITE };
  header(s, { kicker, time, title, accent });

  const top = 1.95;
  const h = cardData ? 3.15 : image ? 2.55 : 4.6;
  const lx = 0.7, lw = 5.55;
  const rx = 6.85, rw = 5.75;

  [{ x: lx, w: lw, ...left }, { x: rx, w: rw, ...right }].forEach((panel) => {
    s.addText(panel.label.toUpperCase(), {
      x: panel.x, y: top, w: panel.w, h: 0.3, fontSize: 12, bold: true, color: accent,
      fontFace: BODY_FONT, isTextBox: true, margin: 0, charSpacing: 1,
    });
    s.addText(
      panel.bullets.map((t) => ({ text: t, options: { bullet: { code: "25AA", color: accent, indent: 16 }, fontSize: 14, color: TEXT, breakLine: true, paraSpaceAfter: 9 } })),
      { x: panel.x, y: top + 0.4, w: panel.w, h: h - 0.4, fontFace: BODY_FONT, isTextBox: true, margin: 0, valign: "top", lineSpacingMultiple: 1.15 }
    );
  });
  s.addShape(pres.ShapeType.rect, { x: 6.55, y: top, w: 0.015, h, fill: { color: LINE } });

  if (cardData) card(s, { ...cardData, x: 0.7, y: top + h + 0.25, w: 11.9, h: cardData.h || 1.2, accent });

  if (image) {
    const boxW = 6.5, boxH = 2.05;
    const boxX = (13.333 - boxW) / 2;
    const boxY = top + h + 0.15;
    s.addImage({ path: image.path, x: boxX, y: boxY, w: boxW, h: boxH, sizing: { type: "contain", w: boxW, h: boxH } });
    if (image.caption) {
      s.addText(image.caption, {
        x: boxX, y: boxY + boxH + 0.03, w: boxW, h: 0.3, fontSize: 10, italic: true, color: GREY,
        fontFace: BODY_FONT, isTextBox: true, margin: 0, align: "center",
      });
    }
  }

  footer(s, pageLabel);
}

// ---------------------------------------------------------------
// Layout: numbered step-flow — horizontal badge cards, for sequential content
// ---------------------------------------------------------------
function stepFlow({ pageLabel, time, kicker, title, steps, accent = RED }) {
  const s = pres.addSlide();
  s.background = { color: WHITE };
  header(s, { kicker, time, title, accent });

  const n = steps.length;
  const top = 2.15;
  const gap = 0.3;
  const cardW = (11.9 - gap * (n - 1)) / n;
  steps.forEach((step, i) => {
    const x = 0.7 + i * (cardW + gap);
    s.addShape(pres.ShapeType.ellipse, { x: x + cardW / 2 - 0.32, y: top, w: 0.64, h: 0.64, fill: { color: accent }, line: { type: "none" } });
    s.addText(String(i + 1), {
      x: x + cardW / 2 - 0.32, y: top, w: 0.64, h: 0.64, fontSize: 20, bold: true, color: WHITE,
      fontFace: HEAD_FONT, isTextBox: true, margin: 0, align: "center", valign: "middle",
    });
    s.addText(step.label, {
      x, y: top + 0.82, w: cardW, h: 0.45, fontSize: 13.5, bold: true, color: TEXT, align: "center",
      fontFace: BODY_FONT, isTextBox: true, margin: 0, valign: "top",
    });
    s.addText(step.detail, {
      x, y: top + 1.3, w: cardW, h: 3.0, fontSize: 12, color: GREY, align: "center",
      fontFace: BODY_FONT, isTextBox: true, margin: 0, valign: "top", lineSpacingMultiple: 1.15,
    });
    if (i < n - 1) {
      s.addShape(pres.ShapeType.rightArrow, { x: x + cardW + 0.01, y: top + 0.25, w: 0.26, h: 0.14, fill: { color: "D9D9D9" }, line: { type: "none" } });
    }
  });

  footer(s, pageLabel);
}

// ---------------------------------------------------------------
// Cover (not part of the 6:40 spoken budget)
// ---------------------------------------------------------------
function titleSlide() {
  const s = pres.addSlide();
  s.background = { color: BLACK };
  s.addShape(pres.ShapeType.rect, { x: 0, y: 0, w: 13.333, h: 0.12, fill: { color: RED } });
  s.addImage({ path: LOGO_WHITE, x: 0.8, y: 0.55, w: 2.6, h: 0.26 });
  s.addText("Multi-Organ CT Segmentation on SegTHOR", {
    x: 0.8, y: 2.5, w: 11.5, h: 1.1, fontSize: 38, bold: true, color: WHITE,
    fontFace: HEAD_FONT, isTextBox: true, margin: 0,
  });
  s.addShape(pres.ShapeType.rect, { x: 0.82, y: 3.35, w: 1.6, h: 0.05, fill: { color: RED } });
  s.addText("Mid-project presentation — esophagus, heart, trachea and aorta segmentation from thoracic CT", {
    x: 0.8, y: 3.55, w: 10.5, h: 0.5, fontSize: 16, color: "D9D9D9",
    fontFace: BODY_FONT, isTextBox: true, margin: 0,
  });
  s.addText("Team names · Group N", {
    x: 0.8, y: 6.5, w: 8, h: 0.35, fontSize: 13, color: "B5B5B5",
    fontFace: BODY_FONT, isTextBox: true, margin: 0,
  });
  s.addText("September 2026", {
    x: 0.8, y: 6.85, w: 8, h: 0.3, fontSize: 11, color: "B5B5B5",
    fontFace: BODY_FONT, isTextBox: true, margin: 0,
  });
}
titleSlide();

// ---------------------------------------------------------------
// 1. Clinical problem — 0:40 — single column
// ---------------------------------------------------------------
singleColumn({
  pageLabel: "01", time: "0:40", accent: BLACK,
  kicker: "Clinical problem",
  title: "Clinical Motivation: Organs at Risk in Thoracic CT",
  bullets: [
    { text: "[TODO] why this matters clinically" },
    { text: "[TODO] the manual-contouring pain point, if using one", sub: true },
    { text: "Task: 3D thoracic CT → esophagus, heart, trachea, aorta", bold: true },
  ],
});

// ---------------------------------------------------------------
// 2. Data integrity (Step 0) — 0:35 — single column + card
// ---------------------------------------------------------------
singleColumn({
  pageLabel: "02", time: "0:35", accent: RED,
  kicker: "Data integrity — Step 0",
  title: "Dataset Validation: Correcting the Aorta Label",
  bullets: [
    { text: "[TODO] how the mislabeling was found and fixed" },
    { text: "[TODO] real per-class voxel share, now corrected — as numbers, not a bar chart" },
    { text: "[TODO] any other integrity checks run (e.g. alignment)" },
  ],
  cardData: {
    label: "Finding",
    body: "Aorta (label 4) was mislabeled — 0 / 20 patients had it before the fix. Caught and corrected before any preprocessing or training ran on it.",
  },
});

// ---------------------------------------------------------------
// 3. Exploratory Analysis I — 0:35 — single column, large figure
// ---------------------------------------------------------------
bigFigureSlide({
  pageLabel: "03", time: "0:35", accent: ORANGE,
  kicker: "Data exploration",
  title: "Exploratory Analysis I: Scan Geometry for the 20 CT Scans",
  image: IMG.fov,
  boxes: [
    { label: "What it shows", body: "[TODO] voxel spacing across patients, slice count / scan length", accent: ORANGE },
    { label: "How it was made", body: "[TODO] how it was measured", accent: RED },
    { label: "What we learn", body: "[TODO] what this implies", accent: PURPLE },
  ],
});

// ---------------------------------------------------------------
// 4. Exploratory Analysis II — 0:35 — single column
// ---------------------------------------------------------------
singleColumn({
  pageLabel: "04", time: "0:35", accent: ORANGE,
  kicker: "Data exploration",
  title: "Exploratory Analysis II: Whole-Scan & Per-Organ Intensity",
  bullets: [
    { text: "[TODO] whole-scan HU distribution, with each organ's voxels coloured" },
    { text: "[TODO] aorta vs. heart in HU" },
    { text: "[TODO] what this implies" },
  ],
});

// ---------------------------------------------------------------
// 5. Exploratory Analysis III — 0:35 — two panel
// ---------------------------------------------------------------
twoPanel({
  pageLabel: "05", time: "0:35", accent: ORANGE,
  kicker: "Data exploration",
  title: "Exploratory Analysis III: Label Topology & Adjacency",
  left: { label: "Topology", bullets: ["[TODO] connected components per label", "[TODO] fragmentation risk, if any"] },
  right: { label: "Adjacency", bullets: ["[TODO] contact between label pairs", "[TODO] which pairs matter most, now that aorta is real"] },
});

// ---------------------------------------------------------------
// 6. Exploratory Analysis IV — 0:35 — two panel + card
// ---------------------------------------------------------------
twoPanel({
  pageLabel: "06", time: "0:35", accent: ORANGE,
  kicker: "Data exploration",
  title: "Exploratory Analysis IV: Spatial Distribution & Extent",
  left: { label: "Position", bullets: ["[TODO] where each organ sits in 3D"] },
  right: { label: "Extent", bullets: ["[TODO] coverage along the body axis"] },
  cardData: {
    label: "Why this matters",
    body: "[TODO] how position and extent inform the preprocessing decisions on the next slide.",
    h: 1.0,
  },
});

// ---------------------------------------------------------------
// 6. Preprocessing — 0:55 — step flow (fixes the "unclear" complaint)
// ---------------------------------------------------------------
stepFlow({
  pageLabel: "07", time: "0:55", accent: BLUE,
  kicker: "Preprocessing pipeline",
  title: "Preprocessing: From Measurement to Method",
  steps: [
    { label: "Step 1", detail: "[TODO], motivated by the geometry measurement (slide 3)" },
    { label: "Step 2", detail: "[TODO], motivated by the intensity measurement (slide 4)" },
    { label: "Step 3", detail: "[TODO], motivated by the position/extent measurement (slide 6)" },
  ],
});

// ---------------------------------------------------------------
// 7. Baseline — 0:35 — single column + stat card
// ---------------------------------------------------------------
singleColumn({
  pageLabel: "08", time: "0:35", accent: PURPLE,
  kicker: "Baseline",
  title: "Baseline Performance: Unmodified Pipeline (CE Loss, 25 Epochs)",
  bullets: [
    { text: "Split: 15 train / 5 val — val = patients 01, 11, 15, 17, 19", bold: true },
    { text: "Scores below are means over these 5 held-out scans", sub: true },
  ],
  cardData: {
    label: "Positive-slice Dice (patient means)",
    body: "Esophagus: 0.47\nHeart: 0.58\nTrachea: 0.50",
    h: 2.2,
  },
});

// ---------------------------------------------------------------
// 8. Modification & result — 2:10 — step flow, largest rubric weight
// ---------------------------------------------------------------
stepFlow({
  pageLabel: "09", time: "2:10", accent: PURPLE,
  kicker: "Modification & result — largest rubric weight",
  title: "Proposed Modification: Motivation, Method & Preliminary Result",
  steps: [
    { label: "Motivation", detail: "[TODO], tied back to an exploration finding (slides 3–6)" },
    { label: "Method", detail: "[TODO] the modification tried" },
    { label: "Result", detail: "[TODO] vs. baseline — from the run in progress" },
    { label: "Next steps", detail: "[TODO] one or two concrete next experiments" },
  ],
});

// ---------------------------------------------------------------
// Closing — status, outside the 6:40 budget
// ---------------------------------------------------------------
function closingSlide() {
  const s = pres.addSlide();
  s.background = { color: BLACK };
  s.addShape(pres.ShapeType.rect, { x: 0, y: 0, w: 0.18, h: 7.5, fill: { color: RED } });
  s.addText("Status & Next Steps", {
    x: 0.8, y: 0.6, w: 11, h: 0.7, fontSize: 30, bold: true, color: WHITE,
    fontFace: HEAD_FONT, isTextBox: true, margin: 0,
  });
  s.addText(
    [
      { text: "Done: dataset validation, exploration, baseline, preprocessing pipeline", options: { bullet: { code: "25AA", color: RED, indent: 18 }, fontSize: 16, color: WHITE, breakLine: true, paraSpaceAfter: 10 } },
      { text: "In progress: [TODO] first modified-loss / modified-preprocessing run", options: { bullet: { code: "25AA", color: RED, indent: 18 }, fontSize: 16, color: WHITE, breakLine: true, paraSpaceAfter: 10 } },
      { text: "Next: [TODO] one or two concrete next experiments before the final deadline", options: { bullet: { code: "25AA", color: RED, indent: 18 }, fontSize: 16, color: WHITE, breakLine: true, paraSpaceAfter: 10 } },
    ],
    { x: 0.8, y: 1.7, w: 10.5, h: 3, fontFace: BODY_FONT, isTextBox: true, margin: 0, valign: "top", lineSpacingMultiple: 1.15 }
  );
  s.addText("Questions", {
    x: 0.8, y: 6.3, w: 6, h: 0.6, fontSize: 20, bold: true, color: "B5B5B5",
    fontFace: HEAD_FONT, isTextBox: true, margin: 0,
  });
  footer(s, "10", LOGO_WHITE);
}
closingSlide();

// ---------------------------------------------------------------
// Appendix — outside the 6:40 budget, backup material only
// ---------------------------------------------------------------
singleColumn({
  pageLabel: "A1", accent: GREY,
  kicker: "Appendix A — backup, not in the timed talk",
  title: "Appendix A: Label Shape Variability Across Patients",
  bullets: [
    { text: "[TODO] what this figure shows, patient by patient" },
    { text: "[TODO] where aorta fits into that pattern" },
    { text: "Backup only — pull in if a question opens the door", italic: true, sub: true },
  ],
});

pres.writeFile({ fileName: OUT }).then(() => console.log("wrote", OUT));
