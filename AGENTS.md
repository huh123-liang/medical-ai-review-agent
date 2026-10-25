# Repository working rules

- This is an independent medical-output evaluation research agent, not the Doctor/Admin platform.
- Never commit credentials, patient/clinician records, raw public benchmark cases/labels, databases, runtime indexes, caches or model-output logs.
- Use the installed project root; do not assume the author's Windows drive paths.
- Default/offline tests use mocks and block network. Real model calls require explicit user authorization, local credentials, valid prices and budget reservations.
- Preserve original responses, rubric wording and physician labels. Partition by conversation/duplicate groups; never use held-out labels as prompt examples or RAG data.
- Separate protocol validity, citation location, semantic support, applicability, medical judgment and review status.
- Keep all failed, interrupted, undetermined and not-run items in the planned denominator. Never present mock/literature/target metrics as measured clinical performance.
- Default flow/API and old experiment contracts must not change merely because a new research candidate is added.
- Keep research prototypes non-default. Human review, platform integration, MCP, local models and multi-agent expansion need separate scope and validation.
- Current maintained docs: docs/project/. Historical notes and scripts may refer to local archives intentionally excluded from the public repository.
