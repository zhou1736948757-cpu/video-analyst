# Removed provider / dry-run tests

The corresponding runtime mode was intentionally removed from the product.

Deleted or retired test modules:

- test_openai_compatible_provider.py
- test_anthropic_compatible_provider.py
- test_mock_vision_provider.py
- test_fine_provider.py
- test_provider_check_fields.py
- test_vision_provider_base.py
- test_network_scope.py
- test_real_api_decision.py
- test_model_resolution.py
- test_dry_run_no_key.py
- test_byok_security.py (vision-key budget/redaction path)

Product vision path is exclusively `in_session_agent`:
local scripts extract frames; the current conversation Agent inspects images
and writes analysis JSON. External vision APIs and mock vision providers
are not product runtime modes.
