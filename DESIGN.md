# Portable comparison report

Read mode for engineers reviewing a retrieval change. Keep the standard-library CLI; the HTML is one offline file generated from the same JSON comparison. No web framework, hosted service, external font, telemetry or build step.

Use paper #f5f6f2, ink #202321, secondary #5a615b, divider #d7dcd5, fail red #a02623 and bounded pass green #25633b. System sans at 28px report title, 17px section, 14px body, 13px tables, 12px metadata; monospace only identifiers and commands. Spacing uses 8/16/24/32px, thin dividers and no decorative cards, gradients or shadows.

Order: report validity and source identity, bounded verdict and gate mode, primary metric rows, other metrics and applicability, question evidence, measured cost, exact reproduction command and portable JSON download. Passing case checks and regressing ranking metrics are distinct. Missing intervals say Not estimated; tiny fixtures identify themselves as smoke tests. Cost is measured but not gated.

Metric tables become labelled rows on phones. Commands scroll inside their own region. Native disclosures expose all cases, rankings and source fingerprints. Keyboard focus uses a 3px outline. No animation is needed. Invalid comparisons get an explicit No gate verdict report and exit 2. Fingerprints identify recorded inputs; reports remain unsigned and require trusted provenance.
