# RAG Regression Gate

<!-- impeccable:product-schema 1 -->

## Platform

Web product entry and interactive saved-evidence viewer, backed by the existing Python CLI and portable offline reports.

## Users

The user explicitly selected an internal help assistant: a team checking whether staff can still find the right policy information before an update. Nontechnical people must understand the consequence and next review action; technical reviewers must retain exact scores, source identities and reproduction commands. Practitioner demand and measured user benefits remain unvalidated.

## Purpose

Explain whether a proposed search change makes labelled supporting information harder to retrieve, why the gate blocks or passes the comparison, and what evidence a person should inspect next. The browser must distinguish recorded engine results from live evaluation. This product evaluates retrieval and reference-citation checks; it does not generate answers or establish their truth.

## Capabilities

The standard-library Python core scores baseline and candidate on the same labelled questions. With bootstrap enabled, a metric blocks only when its drop exceeds the practical threshold and its paired confidence interval excludes zero. Tiny fixtures use explicitly labelled threshold-only checks. A pass does not prove improvement or production readiness. Cost is measured, not gated. Invalid, incomplete or incomparable inputs produce no gate verdict.

## Existing evidence

The committed synthetic policies fixture has eleven documents and six questions. The saved demonstration has current retention evidence move from first to second place, beneath superseded evidence, while all six case checks still pass. This changes MRR from 0.9167 to 0.8333 and nDCG from 0.9385 to 0.8770 in an explicitly threshold-only smoke test. Separate SciFact evidence in README is not evidence of the internal-assistant scenario or generated-answer quality.

## Constraints

Preserve core scoring, validation, reproducibility, CLI exit semantics, standard-library-only core, opt-in model downloads and unsigned-report provenance limits. No new AI call, account, upload service or telemetry is implied by this design task. Keep portable HTML useful offline and developer evidence inspectable. No invented users, savings, adoption, live test runs or business results.

## Confirmed direction

The user requested the same product treatment as ProjectLens and PolicyLens, using Hallmark and Apple's principles, with clear value for nontechnical users, before/after captures, full interactive journey tests and publication. Preserve the existing restrained paper/ink report identity. The user approved direction B, the before/after rank explorer, on 2026-10-23 after visual mockups. The implemented landing page uses a wide headline, paired source rankings and a plain-language hold explanation. On phones, the rankings stack in reading order. Native disclosures retain exact source text, metrics and reproduction details.


## Built journey and boundaries

The landing page opens on the saved retention example, where a superseded proposal overtakes current guidance. Visitors can move through all six questions, choose a question directly, switch between the deliberate regression and unchanged ranking, inspect every retrieved source, download the selected comparison, and open its portable offline report. Browser controls explore committed results; they do not evaluate uploaded data or run a live assistant.

The own-data path explains the handoff: a reviewer chooses meaningful questions and labelled evidence; an engineer runs the CLI against the team's documents and release checks. The website has no upload form, account or model service. Data stays where the tool runs. Optional model downloads remain explicit and separate.

The offline report remains a distinct, denser engineering read mode. Its validity, bounded verdict, metric applicability, evidence, measured cost and provenance rules remain intact. A passing saved comparison means no detected regression under its settings; it does not establish correct answers or a safe release.
