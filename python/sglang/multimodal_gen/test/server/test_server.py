"""
1 GPU tests - All cases combined for auto-balanced partitioning.

This file contains all 1-GPU test cases. The test runner (run_suite.py)
automatically partitions these cases across multiple CI jobs using the LPT
algorithm for load balancing.

Individual cases can still be run directly:
    pytest test_server.py -k qwen_image_t2i
"""

from __future__ import annotations

import pytest

from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger
from sglang.multimodal_gen.test.server.test_server_common import (  # noqa: F401
    DiffusionServerBase,
    diffusion_server,
)
from sglang.multimodal_gen.test.server.testcase_configs import (
    ONE_GPU_CASES,
    DiffusionTestCase,
)

logger = init_logger(__name__)


class TestDiffusionServerOneGpu(DiffusionServerBase):
    """Performance tests for all 1-GPU diffusion cases."""

    @pytest.fixture(params=ONE_GPU_CASES, ids=lambda c: c.id)
    def case(self, request) -> DiffusionTestCase:
        """Provide a DiffusionTestCase for each 1-GPU test."""
        return request.param
