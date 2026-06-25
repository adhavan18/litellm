"""
Unit tests for Router.update_settings() — specifically that retry_policy
can be set dynamically (issue #31308).

Before this fix:
  - retry_policy was absent from _allowed_settings in update_settings()
  - UpdateRouterConfig had no retry_policy field
  → setting it via POST /config/update was silently dropped
"""

import pytest

from litellm import Router
from litellm.types.router import RetryPolicy, UpdateRouterConfig


class TestRouterUpdateSettingsRetryPolicy:
    """Regression tests for retry_policy support in Router.update_settings()."""

    def _make_router(self):
        return Router(
            model_list=[
                {
                    "model_name": "test-model",
                    "litellm_params": {"model": "openai/gpt-4", "api_key": "test-key"},
                }
            ]
        )

    def test_update_settings_retry_policy_dict(self):
        """retry_policy passed as a dict must be stored as a RetryPolicy instance."""
        router = self._make_router()
        assert router.retry_policy is None

        router.update_settings(retry_policy={"RateLimitErrorRetries": 3, "TimeoutErrorRetries": 2})

        assert router.retry_policy is not None
        assert isinstance(router.retry_policy, RetryPolicy)
        assert router.retry_policy.RateLimitErrorRetries == 3
        assert router.retry_policy.TimeoutErrorRetries == 2

    def test_update_settings_retry_policy_visible_in_get_settings(self):
        """After update_settings, retry_policy appears in get_settings()."""
        router = self._make_router()
        router.update_settings(retry_policy={"RateLimitErrorRetries": 5})

        settings = router.get_settings()
        assert "retry_policy" in settings
        rp = settings["retry_policy"]
        assert isinstance(rp, RetryPolicy)
        assert rp.RateLimitErrorRetries == 5

    def test_update_settings_retry_policy_instance(self):
        """retry_policy passed as a RetryPolicy instance is stored unchanged."""
        router = self._make_router()
        policy = RetryPolicy(RateLimitErrorRetries=7)
        router.update_settings(retry_policy=policy)

        assert router.retry_policy is policy
        assert router.retry_policy.RateLimitErrorRetries == 7

    def test_update_router_config_accepts_retry_policy(self):
        """UpdateRouterConfig must accept retry_policy without validation error."""
        cfg = UpdateRouterConfig(retry_policy={"RateLimitErrorRetries": 4})
        assert cfg.retry_policy == {"RateLimitErrorRetries": 4}

    def test_update_router_config_retry_policy_excluded_when_none(self):
        """retry_policy=None must not appear in the serialised config dict."""
        cfg = UpdateRouterConfig(num_retries=3)
        d = cfg.dict(exclude_none=True)
        assert "retry_policy" not in d
        assert d["num_retries"] == 3
