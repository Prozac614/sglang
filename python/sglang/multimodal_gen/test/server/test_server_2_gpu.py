"""
2 GPU tests - All cases combined for auto-balanced partitioning.

This file contains all 2-GPU test cases. The test runner (run_suite.py)
automatically partitions these cases across multiple CI jobs using the LPT
algorithm for load balancing.

Individual cases can still be run directly:
    pytest test_server_2_gpu.py -k wan2_1_t2v_14b_2gpu
"""

from __future__ import annotations

import pytest

from sglang.multimodal_gen.test.server.test_server_common import (  # noqa: F401
    DiffusionServerBase,
    diffusion_server,
)
from sglang.multimodal_gen.test.server.testcase_configs import (
    TWO_GPU_CASES,
    DiffusionTestCase,
)


class TestDiffusionServerTwoGpu(DiffusionServerBase):
    """Performance tests for all 2-GPU diffusion cases."""

    @pytest.fixture(params=TWO_GPU_CASES, ids=lambda c: c.id)
    def case(self, request) -> DiffusionTestCase:
        """Provide a DiffusionTestCase for each 2-GPU test."""
        return request.param
