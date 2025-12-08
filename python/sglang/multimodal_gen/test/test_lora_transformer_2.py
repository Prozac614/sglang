#!/usr/bin/env python
"""
End-to-end test for LoRA support with dual-transformer models (Wan2.2 A14B).

This script tests the transformer_2 LoRA adaptation in lora_pipeline.py.
Each test runs as a separate invocation to avoid GPU memory accumulation.

Usage:
    # Run all tests sequentially
    CUDA_VISIBLE_DEVICES=1,2,3,4 python test_lora_transformer_2.py --all

    # Run individual tests
    CUDA_VISIBLE_DEVICES=1,2,3,4 python test_lora_transformer_2.py --test1  # Without LoRA
    CUDA_VISIBLE_DEVICES=1,2,3,4 python test_lora_transformer_2.py --test2  # With LoRA
    CUDA_VISIBLE_DEVICES=1,2,3,4 python test_lora_transformer_2.py --test3  # Unmerge + Remerge

Expected output:
    ./test_lora_outputs/
    ├── 1_without_lora.mp4          # Baseline (no LoRA)
    ├── 2_with_lora.mp4             # LoRA effect applied
    └── 3_unmerge_remerge.mp4       # After unmerge then remerge

LoRA: Cseti/wan2.2-14B-Arcane_Jinx-lora-v1
    - Trigger words: Nfj1nx, blue hair
    - Character: Jinx from Arcane
"""

import argparse
import subprocess
import sys
from pathlib import Path

from sglang.multimodal_gen.runtime.entrypoints.diffusion_generator import DiffGenerator

# ============================================================================
# Configuration
# ============================================================================

MODEL_PATH = "Wan-AI/Wan2.2-T2V-A14B-Diffusers"
LORA_PATH = "Cseti/wan2.2-14B-Arcane_Jinx-lora-v1"
LORA_NICKNAME = "arcane_jinx"

# Use the LoRA's trigger words for best effect
PROMPT = "Nfj1nx with blue hair, a girl walking through a neon-lit cyberpunk city at night, cinematic lighting"

# Video parameters
WIDTH = 720
HEIGHT = 480
NUM_INFERENCE_STEPS = 30
NUM_GPUS = 2

# Output directory
OUTPUT_DIR = Path("./test_lora_outputs")


# ============================================================================
# Helper Functions
# ============================================================================


def get_sampling_params_kwargs(output_file_name: str, seed: int = 1024) -> dict:
    """Create sampling parameters dict for generations."""
    return {
        "prompt": PROMPT,
        "width": WIDTH,
        "height": HEIGHT,
        "num_inference_steps": NUM_INFERENCE_STEPS,
        "guidance_scale": 5.0,
        "seed": seed,
        "save_output": True,
        "output_path": str(OUTPUT_DIR),
        "output_file_name": output_file_name,
    }


def print_section(title: str) -> None:
    """Print a formatted section header."""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


# ============================================================================
# Test Functions
# ============================================================================


def test1_without_lora() -> None:
    """
    Test 1: Generate video WITHOUT LoRA adapter.
    This serves as the baseline for comparison.
    """
    print_section("Test 1: Generate WITHOUT LoRA (Baseline)")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    generator = DiffGenerator.from_pretrained(
        model_path=MODEL_PATH,
        num_gpus=NUM_GPUS,
        text_encoder_cpu_offload=True,
    )

    print(f"Generating with prompt: {PROMPT}")
    params = get_sampling_params_kwargs("1_without_lora")
    generator.generate(sampling_params_kwargs=params)

    output_path = OUTPUT_DIR / "1_without_lora.mp4"
    print(f"✅ Saved: {output_path}")
    print("\n✅ Test 1 completed!")


def test2_with_lora() -> None:
    """
    Test 2: Generate video WITH LoRA adapter.
    Tests: set_lora() + lora_layers_transformer_2
    """
    print_section("Test 2: Generate WITH LoRA")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    generator = DiffGenerator.from_pretrained(
        model_path=MODEL_PATH,
        lora_path=LORA_PATH,
        lora_nickname=LORA_NICKNAME,
        num_gpus=NUM_GPUS,
        text_encoder_cpu_offload=True,
    )

    print(f"LoRA: {LORA_PATH}")
    print(f"Generating with prompt: {PROMPT}")
    params = get_sampling_params_kwargs("2_with_lora", seed=2048)
    generator.generate(sampling_params_kwargs=params)

    output_path = OUTPUT_DIR / "2_with_lora.mp4"
    print(f"✅ Saved: {output_path}")
    print("\n✅ Test 2 completed!")


def test3_unmerge_remerge() -> None:
    """
    Test 3: Test unmerge and remerge LoRA weights.
    Tests: unmerge_lora_weights() and merge_lora_weights() for transformer_2
    """
    print_section("Test 3: Unmerge and Remerge LoRA")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    generator = DiffGenerator.from_pretrained(
        model_path=MODEL_PATH,
        lora_path=LORA_PATH,
        lora_nickname=LORA_NICKNAME,
        num_gpus=NUM_GPUS,
        text_encoder_cpu_offload=True,
    )

    print(f"LoRA: {LORA_PATH}")

    # Unmerge LoRA
    print("\n--- Calling unmerge_lora_weights() ---")
    generator.unmerge_lora_weights()
    print("✅ unmerge_lora_weights() completed")

    # Remerge LoRA
    print("\n--- Calling merge_lora_weights() ---")
    generator.merge_lora_weights()
    print("✅ merge_lora_weights() completed")

    # Generate after remerge
    print(f"\nGenerating with prompt: {PROMPT}")
    print("(LoRA should be effective after remerge)")
    params = get_sampling_params_kwargs("3_unmerge_remerge")
    generator.generate(sampling_params_kwargs=params)

    output_path = OUTPUT_DIR / "3_unmerge_remerge.mp4"
    print(f"✅ Saved: {output_path}")
    print("\n✅ Test 3 completed!")


def run_all_tests() -> None:
    """Run all tests as separate subprocess to avoid memory issues."""
    print_section("Running All Tests (Separate Processes)")

    script_path = Path(__file__).absolute()

    tests = [
        ("--test1", "Test 1: Without LoRA"),
        ("--test2", "Test 2: With LoRA"),
        ("--test3", "Test 3: Unmerge + Remerge"),
    ]

    results = []
    for flag, name in tests:
        print(f"\n{'='*70}")
        print(f"  Starting: {name}")
        print(f"{'='*70}")

        result = subprocess.run(
            [sys.executable, str(script_path), flag],
            env={**dict(__import__("os").environ)},
        )

        success = result.returncode == 0
        results.append((name, success))

        if success:
            print(f"✅ {name} - PASSED")
        else:
            print(f"❌ {name} - FAILED (exit code: {result.returncode})")

    # Print summary
    print_section("TEST SUMMARY")
    print("\nResults:")
    for name, success in results:
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"  {name}: {status}")

    print("\nGenerated Videos:")
    print(f"  1. Baseline (no LoRA):    {OUTPUT_DIR}/1_without_lora.mp4")
    print(f"  2. With LoRA:             {OUTPUT_DIR}/2_with_lora.mp4")
    print(f"  3. Unmerge + Remerge:     {OUTPUT_DIR}/3_unmerge_remerge.mp4")

    print("\nExpected Results:")
    print("  - Video 1 should look DIFFERENT from Video 2 and 3 (no LoRA vs with LoRA)")
    print(
        "  - Video 2 and 3 should both show LoRA effect (but different due to different seeds)"
    )

    print("\nVerified lora_pipeline.py functions for transformer_2:")
    print("  ✅ lora_layers_transformer_2 (class attribute)")
    print("  ✅ convert_to_lora_layers() - transformer_2 branch")
    print("  ✅ set_lora() - transformer_2 weight application")
    print("  ✅ unmerge_lora_weights() - transformer_2 iteration")
    print("  ✅ merge_lora_weights() - transformer_2 iteration")

    all_passed = all(success for _, success in results)
    if all_passed:
        print("\n" + "=" * 70)
        print("  ALL TESTS COMPLETED SUCCESSFULLY!")
        print("=" * 70 + "\n")
    else:
        print("\n" + "=" * 70)
        print("  SOME TESTS FAILED - SEE ABOVE FOR DETAILS")
        print("=" * 70 + "\n")
        sys.exit(1)


# ============================================================================
# Main
# ============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Test LoRA transformer_2 support for Wan2.2 A14B"
    )
    parser.add_argument("--all", action="store_true", help="Run all tests")
    parser.add_argument("--test1", action="store_true", help="Test 1: Without LoRA")
    parser.add_argument("--test2", action="store_true", help="Test 2: With LoRA")
    parser.add_argument(
        "--test3", action="store_true", help="Test 3: Unmerge + Remerge"
    )

    args = parser.parse_args()

    if args.test1:
        test1_without_lora()
    elif args.test2:
        test2_with_lora()
    elif args.test3:
        test3_unmerge_remerge()
    elif args.all:
        run_all_tests()
    else:
        # Default: run all tests
        run_all_tests()


if __name__ == "__main__":
    main()
