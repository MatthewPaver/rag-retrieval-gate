---
name: RAG Regression Gate
description: Restrained paper and ink for inspectable retrieval evidence.
colors:
  paper: "#f5f6f2"
  ink: "#202321"
  muted: "#5a615b"
  line: "#d7dcd5"
  hold: "#a02623"
  current: "#25633b"
  hold-paper: "#f8ece9"
  current-paper: "#e2ece1"
typography:
  display:
    fontFamily: "Arial, Helvetica, sans-serif"
    fontSize: "clamp(40px, 4.55vw, 64px)"
    fontWeight: 700
    lineHeight: 1.08
    letterSpacing: "-0.04em"
  body:
    fontFamily: "Arial, Helvetica, sans-serif"
    fontSize: "17px"
    fontWeight: 400
    lineHeight: 1.5
  code:
    fontFamily: "ui-monospace, SFMono-Regular, Consolas, monospace"
    fontSize: "14px"
rounded:
  compact: "3px"
  control: "4px"
spacing:
  compact: "8px"
  regular: "16px"
  inset: "20px"
  section: "24px"
components:
  button-primary:
    backgroundColor: "{colors.ink}"
    textColor: "{colors.paper}"
    rounded: "{rounded.control}"
    padding: "12px 38px"
    height: "52px"
  source-current:
    backgroundColor: "{colors.current-paper}"
    textColor: "{colors.current}"
    rounded: "{rounded.compact}"
    padding: "3px 9px"
---

# Design System: RAG Regression Gate

## Overview

The approved world is restrained paper and ink: flat, spacious, typographic and grounded in readable evidence. The product entry at `web/index.html` gives nontechnical reviewers a visible consequence before detailed measurement. `web/product.css` and `web/product.js` are its implementation sources.

The portable generated report is a separate engineering read mode. It shares the palette and restraint, but keeps a denser hierarchy and offline, single-file delivery. Do not apply landing-page scale to its tables.

**Key Characteristics:**
- Pale paper, near-black text and thin rules.
- Sparse red and green tied to explicit source and result labels.
- Large introductory type, compact evidence and native disclosures.

## Colors

### Primary

Hold red marks the regressing result and superseded evidence. Current green identifies current evidence, focus and the unchanged result. Tinted paper supports these states without adding depth.

**The Labelled State Rule.** Pair state color with words and source position; color alone does not explain a result.

### Neutral

Colors and font families are declared as CSS custom properties in the implemented stylesheet. Paper is the page background. Ink carries titles and primary actions. Muted text carries context; line separates sections and rows. Primary actions use ink rather than borrowing a result color.

## Typography

The landing page uses Arial with Helvetica and sans-serif fallbacks. Its display is bold and tightly spaced; explanatory prose remains regular weight. The introduction uses larger supporting text (26px desktop); section headings use a measured hierarchy (36px, then 29px ranking titles and 21px source titles). Code and identifiers use monospace.

At the narrow breakpoint, body text becomes 16px, the display becomes 40px, and the question heading becomes 26px. Secondary labels remain subordinate without replacing readable source text.

**The Read-Mode Rule.** The offline report keeps its system sans hierarchy: 28px report title, 17px sections, 14px body, 13px tables and 12px metadata. Monospace is reserved for identifiers and commands.

## Layout

The landing shell uses 92% width with a 1350px maximum. Thin rules divide the masthead, introduction, evidence and setup sections. Paired source columns use equal tracks with a 7.5% gap. Ordered source rows align large tabular rank numbers with titles, state labels and excerpts.

Below 1100px, source headings wrap and the crossing connectors disappear. Below 650px, source columns and setup steps stack; actions form a vertical sequence, and the shell keeps 18px side margins. The footer's source link occupies its own row with separation from the author credit. All-source evidence also stacks in reading order. Commands scroll within their own region.

The offline report follows validity and source identity, bounded verdict and gate mode, primary metrics, other metrics and applicability, question evidence, measured cost, reproduction command and JSON download. Its metric tables become labelled rows on phones.

## Elevation & Depth

The system is flat: no decorative cards, gradients or shadows. Thin dividers, type scale, whitespace and state tints establish hierarchy. The crossing rank connectors are explanatory geometry on wide screens, not raised surfaces.

## Shapes

Controls use gently curved corners; compact labels and result strips use the smaller radius. Borders remain thin. Source rows are open lists separated by rules rather than enclosed cards. Circular connector endpoints and inline SVG result/navigation icons serve the comparison.

## Components

### Actions and navigation

Primary actions are ink-filled links with bold text, a thin border and the control radius. Desktop padding is recorded above; mobile padding reduces to 12px 22px and the minimum height to 48px. Secondary actions use underlined text. Navigation remains plain links and wraps into a second masthead row on phones.

Keyboard focus uses a visible green outline (3px with 5px offset). Previous/next controls are labelled buttons; disabled controls retain their native disabled attribute and use a neutral fill, muted ink and a not-allowed cursor. The direct question selector has a persistent text label and a muted-ink border with at least 3:1 contrast against paper. A skip link leads to the example. Selection uses a pale green tint; caret and scrollbar colors follow the ink, muted and paper palette. Enabled button presses shift by 1px.

### Source rankings and result strip

Each ranking is a semantic ordered list. Current, superseded and context labels accompany readable titles and excerpts. Wide-screen connectors show the opening rank swap. The result strip precedes the paired rankings and combines an icon, a short verdict and a concrete consequence; it reflows into a stacked explanation on phones. Question views distinguish the selected question’s source order from the overall hold, explicitly naming worse data-retention evidence as its cause. The summary reports “N of total question checks pass” separately from the overall source-order result. The masthead retains RAG Regression Gate with the subordinate descriptor “Search checks for staff assistants”. “Set up your own checks” leads to the engineer handoff, while exact source text and technical results remain in native disclosures.

The unchanged-result switch may briefly move candidate rows into place (8px over 280ms). This feedback is optional, uses the browser animation API, and is skipped when reduced motion is requested. Reduced motion also removes smooth scrolling. The report needs no animation.

### Evidence and setup

Native details/summary elements expose all source text, technical results and engineer setup. Their open and hover states use current green. Code blocks scroll locally, while source identifiers wrap. Downloads and the offline report remain ordinary links. The static opening example and source disclosure remain readable if interactive loading fails; the page explains the failure.

## Do's and Don'ts

### Do:
- **Do** retain readable source text and explicit state labels beside ranking position.
- **Do** preserve the offline report's compact read mode, native disclosures and local command scrolling.
- **Do** keep keyboard focus visible and motion optional.

### Don't:
- **Don't** treat the landing page's display scale as the report's type scale.
- **Don't** substitute color or animated movement for a written consequence.
- **Don't** add decorative elevation to this flat evidence world.
