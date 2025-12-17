#!/usr/bin/env python3
"""
Test 6: LoRA Strength Parameter Test for FLUX.2-dev

This test verifies the strength parameter functionality in set_lora and merge_lora_weights:
1. Generate baseline image (no LoRA)
2. Generate images with different strength values via set_lora (0.5, 1.0, 1.5)
3. Test merge_lora_weights with different strength values after unmerge

Model: black-forest-labs/FLUX.2-dev
LoRA: ostris/flux2_berthe_morisot
Prompt: berthe_morisot style, woman with red hair, playing chess at the park, bomb going off in the background

Usage:
    python 6_test_lora_strength.py [--host HOST] [--port PORT]
"""

import argparse
import base64
import json
import os
import time
from datetime import datetime

import requests
from huggingface_hub import hf_hub_download

LORA_REPO = "ostris/flux2_berthe_morisot"
DEFAULT_PROMPT = "berthe_morisot style, woman with red hair, playing chess at the park, bomb going off in the background"
DEFAULT_SIZE = "1024x1024"


def download_lora_file():
    """Download LoRA file from HuggingFace."""
    print(f"\nDownloading LoRA files from {LORA_REPO}...")
    # The repo typically contains a safetensors file
    lora_path = hf_hub_download(
        repo_id=LORA_REPO, filename="flux2_berthe_morisot.safetensors"
    )
    print(f"  LoRA path: {lora_path}")
    return lora_path


def set_lora(
    base_url: str,
    lora_nickname: str,
    lora_path: str,
    target: str = "transformer",
    strength: float = 1.0,
):
    """Set LoRA adapter via REST API with strength parameter."""
    url = f"{base_url}/v1/set_lora"
    payload = {
        "lora_nickname": lora_nickname,
        "lora_path": lora_path,
        "target": target,
        "strength": strength,
    }
    print(f"\nSetting LoRA: {lora_nickname} -> {target} (strength: {strength})")
    print(f"  Path: {lora_path}")

    response = requests.post(url, json=payload, timeout=120)

    if response.status_code != 200:
        print(f"  Error: {response.status_code}")
        try:
            error_detail = response.json()
            print(f"  Detail: {error_detail}")
        except Exception:
            print(f"  Response: {response.text}")
        response.raise_for_status()

    result = response.json()
    print(f"  Result: {result}")
    return result


def unmerge_lora(base_url: str, target: str = "transformer"):
    """Unmerge LoRA weights via REST API."""
    url = f"{base_url}/v1/unmerge_lora_weights"
    payload = {"target": target}
    print(f"\nUnmerging LoRA weights (target: {target})...")

    response = requests.post(url, json=payload, timeout=120)

    if response.status_code != 200:
        # May fail if no LoRA is merged - that's OK
        print(f"  Note: {response.status_code} - {response.text}")
        return None

    result = response.json()
    print(f"  Result: {result}")
    return result


def merge_lora(base_url: str, target: str = "transformer", strength: float = 1.0):
    """Merge LoRA weights via REST API with strength parameter."""
    url = f"{base_url}/v1/merge_lora_weights"
    payload = {
        "target": target,
        "strength": strength,
    }
    print(f"\nMerging LoRA weights (target: {target}, strength: {strength})...")

    response = requests.post(url, json=payload, timeout=120)

    if response.status_code != 200:
        print(f"  Error: {response.status_code}")
        try:
            error_detail = response.json()
            print(f"  Detail: {error_detail}")
        except Exception:
            print(f"  Response: {response.text}")
        response.raise_for_status()

    result = response.json()
    print(f"  Result: {result}")
    return result


def save_image_from_b64(b64_json: str, output_dir: str, filename: str) -> str:
    """Decode base64 image and save to file."""
    image_bytes = base64.b64decode(b64_json)
    output_path = os.path.join(output_dir, filename)
    with open(output_path, "wb") as f:
        f.write(image_bytes)
    print(f"  Image saved to: {output_path}")
    return output_path


def generate_image(
    base_url: str,
    prompt: str,
    output_dir: str,
    size: str = "1024x1024",
    test_name: str = "test",
    seed: int = 42,
    strength: float = None,
) -> str:
    """Generate an image via REST API (synchronous)."""
    url = f"{base_url}/v1/images/generations"
    payload = {
        "prompt": prompt,
        "size": size,
        "n": 1,
        "response_format": "b64_json",
        "seed": seed,
    }

    print(f"\n{'='*60}")
    print(f"Generating image: {test_name}")
    print(f"Prompt: {prompt}")
    print(f"Size: {size}, Seed: {seed}")
    if strength is not None:
        print(f"Strength: {strength}")
    print(f"{'='*60}")

    start_time = time.time()

    response = requests.post(url, json=payload, timeout=600)

    if response.status_code != 200:
        print(f"Error: {response.status_code}")
        try:
            error_detail = response.json()
            print(f"Detail: {error_detail}")
        except Exception:
            print(f"Response: {response.text}")
        response.raise_for_status()

    elapsed = time.time() - start_time
    print(f"Generation completed in {elapsed:.2f}s")

    result = response.json()

    # Save image
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    image_filename = f"{test_name}_{timestamp}.png"

    b64_json = result.get("data", [{}])[0].get("b64_json")
    if b64_json:
        image_path = save_image_from_b64(b64_json, output_dir, image_filename)
    else:
        print("  Warning: No b64_json in response")
        image_path = None

    # Save metadata
    metadata_path = os.path.join(output_dir, f"{test_name}_{timestamp}.json")
    metadata = {
        "test_name": test_name,
        "prompt": prompt,
        "size": size,
        "seed": seed,
        "strength": strength,
        "generation_time": elapsed,
        "image_path": image_path,
    }
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"  Metadata saved to: {metadata_path}")

    return image_path


def main():
    parser = argparse.ArgumentParser(
        description="LoRA Strength Parameter Test for FLUX.2-dev"
    )
    parser.add_argument("--host", type=str, default="localhost", help="Server host")
    parser.add_argument("--port", type=int, default=30010, help="Server port")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./e2e_lora_strength_outputs",
        help="Output directory",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=DEFAULT_PROMPT,
        help="Prompt for generation",
    )
    parser.add_argument("--size", type=str, default=DEFAULT_SIZE, help="Image size WxH")
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility"
    )

    args = parser.parse_args()
    base_url = f"http://{args.host}:{args.port}"

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    print("\n" + "=" * 60)
    print("TEST 6: LoRA STRENGTH PARAMETER TEST (FLUX.2-dev)")
    print("=" * 60)
    print(f"Server: {base_url}")
    print(f"LoRA: {LORA_REPO}")
    print(f"Output: {args.output_dir}")
    print(f"Seed: {args.seed}")

    # Step 3: Download LoRA file
    print("\n" + "-" * 60)
    print("Step 1: Downloading LoRA file")
    print("-" * 60)
    lora_path = download_lora_file()

    # Step 4: Generate baseline image (no LoRA)
    print("\n" + "-" * 60)
    print("Step 2: Generate BASELINE image (no LoRA)")
    print("-" * 60)
    # First unmerge any existing LoRA
    unmerge_lora(base_url, "transformer")

    generate_image(
        base_url=base_url,
        prompt=args.prompt,
        output_dir=args.output_dir,
        size=args.size,
        test_name="1_baseline_no_lora",
        seed=args.seed,
        strength=None,
    )

    # Step 5: Test set_lora with different strength values
    print("\n" + "-" * 60)
    print("Step 3: Test set_lora with different STRENGTH values")
    print("-" * 60)

    # 5a: strength=0.5
    print("\n>>> Test 3a: set_lora with strength=0.5")
    unmerge_lora(base_url, "transformer")
    set_lora(
        base_url=base_url,
        lora_nickname="berthe_morisot",
        lora_path=lora_path,
        target="transformer",
        strength=0.5,
    )
    generate_image(
        base_url=base_url,
        prompt=args.prompt,
        output_dir=args.output_dir,
        size=args.size,
        test_name="2a_set_lora_strength_0.5",
        seed=args.seed,
        strength=0.5,
    )

    # 5b: strength=1.0 (default)
    print("\n>>> Test 3b: set_lora with strength=1.0 (default)")
    unmerge_lora(base_url, "transformer")
    set_lora(
        base_url=base_url,
        lora_nickname="berthe_morisot",
        lora_path=lora_path,
        target="transformer",
        strength=1.0,
    )
    generate_image(
        base_url=base_url,
        prompt=args.prompt,
        output_dir=args.output_dir,
        size=args.size,
        test_name="2b_set_lora_strength_1.0",
        seed=args.seed,
        strength=1.0,
    )

    # 5c: strength=1.5
    print("\n>>> Test 3c: set_lora with strength=1.5")
    unmerge_lora(base_url, "transformer")
    set_lora(
        base_url=base_url,
        lora_nickname="berthe_morisot",
        lora_path=lora_path,
        target="transformer",
        strength=1.5,
    )
    generate_image(
        base_url=base_url,
        prompt=args.prompt,
        output_dir=args.output_dir,
        size=args.size,
        test_name="2c_set_lora_strength_1.5",
        seed=args.seed,
        strength=1.5,
    )

    # Step 6: Test merge_lora_weights with different strength values
    print("\n" + "-" * 60)
    print("Step 4: Test merge_lora_weights with different STRENGTH values")
    print("-" * 60)

    # First, set LoRA with default strength and then unmerge
    print("\n>>> Setup: set_lora with strength=1.0, then unmerge")
    unmerge_lora(base_url, "transformer")
    set_lora(
        base_url=base_url,
        lora_nickname="berthe_morisot",
        lora_path=lora_path,
        target="transformer",
        strength=1.0,
    )
    unmerge_lora(base_url, "transformer")

    # 6a: merge with strength=0.5
    print("\n>>> Test 4a: merge_lora_weights with strength=0.5")
    merge_lora(base_url, "transformer", strength=0.5)
    generate_image(
        base_url=base_url,
        prompt=args.prompt,
        output_dir=args.output_dir,
        size=args.size,
        test_name="3a_merge_strength_0.5",
        seed=args.seed,
        strength=0.5,
    )

    # 6b: unmerge and merge with strength=1.0
    print("\n>>> Test 4b: merge_lora_weights with strength=1.0")
    unmerge_lora(base_url, "transformer")
    merge_lora(base_url, "transformer", strength=1.0)
    generate_image(
        base_url=base_url,
        prompt=args.prompt,
        output_dir=args.output_dir,
        size=args.size,
        test_name="3b_merge_strength_1.0",
        seed=args.seed,
        strength=1.0,
    )

    # 6c: unmerge and merge with strength=1.5
    print("\n>>> Test 4c: merge_lora_weights with strength=1.5")
    unmerge_lora(base_url, "transformer")
    merge_lora(base_url, "transformer", strength=1.5)
    generate_image(
        base_url=base_url,
        prompt=args.prompt,
        output_dir=args.output_dir,
        size=args.size,
        test_name="3c_merge_strength_1.5",
        seed=args.seed,
        strength=1.5,
    )

    # Step 7: Print completion message
    print("\n" + "=" * 60)
    print("TEST 6 COMPLETED: LoRA Strength Parameter Test finished")
    print("=" * 60)
    print("\nExpected results:")
    print("  - 1_baseline_no_lora: No berthe_morisot style")
    print("  - 2a_set_lora_strength_0.5: Subtle berthe_morisot style")
    print("  - 2b_set_lora_strength_1.0: Standard berthe_morisot style")
    print("  - 2c_set_lora_strength_1.5: Strong berthe_morisot style")
    print("  - 3a_merge_strength_0.5: Should match 2a")
    print("  - 3b_merge_strength_1.0: Should match 2b")
    print("  - 3c_merge_strength_1.5: Should match 2c")
    print(f"\nOutput files saved to: {args.output_dir}")
    print("\nCompare the images to verify strength parameter works correctly!")


if __name__ == "__main__":
    main()
