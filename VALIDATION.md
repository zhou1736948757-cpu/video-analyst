# Publication validation

Checked on 2026-10-11 on Apple Silicon macOS with Python 3.14.7.

- Parsed every Python source file and bundled JSON schema.
- Scanned all publication files for common credential prefixes, private-key
  blocks, and the publisher's private absolute home path; no matches found.
- Ran the portable launcher help and JSON environment doctor.
- Generated a three-second synthetic test video locally and ran `prepare`.
- Confirmed that frame and contact-sheet paths in the manifest exist and that
  the workflow pauses at `waiting_for_coarse_agent_analysis`.
- Ran timestamp inspection at 1.5 seconds.
- Ran URL import in dry-run mode; confirmed playlist/config isolation flags and
  that no import directory was created.

No private media, configuration, generated analysis, caches, or validation media
are included in the repository. These checks do not establish a complete visual
analysis: no agent-authored coarse/fine report was generated. ASR was unavailable
in the check environment. No cloud audio upload or real URL download was tested.
The credential scan is bounded pattern matching, not a comprehensive security
certification.
