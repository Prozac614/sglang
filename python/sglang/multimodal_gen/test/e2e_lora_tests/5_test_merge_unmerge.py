#!/usr/bin/env python3
"""
Test 5: Merge/Unmerge functionality test

This test verifies that merge and unmerge operations work correctly:
1. Start with LoRA merged (from previous tests)
2. Generate a video (with LoRA)
3. Unmerge the LoRA
4. Generate a video (should be baseline-like)
5. Re-merge the LoRA
6. Generate a video (should match step 2)

This confirms that unmerge truly removes LoRA effect and merge restores it.

Usage:
    python 5_test_merge_unmerge.py [--host HOST] [--port PORT]
"""

import argparse
import os
import time
from datetime import datetime

import requests
from huggingface_hub import hf_hub_download

LORA_REPO = "Cseti/wan2.2-14B-Arcane_Jinx-lora-v1"
HIGH_NOISE_LORA = "985347-wan22_14B-high-Nfj1nx-e71.safetensors"
LOW_NOISE_LORA = "985347-wan22_14B-low-Nfj1nx-e65.safetensors"


def download_lora_files():
    """Download LoRA files from HuggingFace."""
    print(f"\nDownloading LoRA files from {LORA_REPO}...")
    high_path = hf_hub_download(repo_id=LORA_REPO, filename=HIGH_NOISE_LORA)
    low_path = hf_hub_download(repo_id=LORA_REPO, filename=LOW_NOISE_LORA)
    print(f"  High noise LoRA: {high_path}")
    print(f"  Low noise LoRA: {low_path}")
    return high_path, low_path


def set_lora(base_url: str, lora_nickname: str, lora_path: str, target: str):
    """Set LoRA adapter via REST API."""
    url = f"{base_url}/v1/set_lora"
    payload = {
        "lora_nickname": lora_nickname,
        "lora_path": lora_path,
        "target": target,
    }
    print(f"\nSetting LoRA: {lora_nickname} -> {target}")

    response = requests.post(url, json=payload, timeout=120)

    if response.status_code != 200:
        print(f"  Error: {response.status_code}")
        try:
            print(f"  Detail: {response.json()}")
        except Exception:
            print(f"  Response: {response.text}")
        response.raise_for_status()

    result = response.json()
    print(f"  Result: {result}")
    return result


def merge_lora(base_url: str, target: str = "all"):
    """Merge LoRA weights via REST API."""
    url = f"{base_url}/v1/merge_lora_weights"
    payload = {"target": target}
    print(f"\nMerging LoRA weights (target: {target})...")

    response = requests.post(url, json=payload, timeout=120)

    if response.status_code != 200:
        print(f"  Error: {response.status_code}")
        try:
            print(f"  Detail: {response.json()}")
        except Exception:
            print(f"  Response: {response.text}")
        response.raise_for_status()

    result = response.json()
    print(f"  Result: {result}")
    return result


def unmerge_lora(base_url: str, target: str = "all"):
    """Unmerge LoRA weights via REST API."""
    url = f"{base_url}/v1/unmerge_lora_weights"
    payload = {"target": target}
    print(f"\nUnmerging LoRA weights (target: {target})...")

    response = requests.post(url, json=payload, timeout=120)

    if response.status_code != 200:
        print(f"  Error: {response.status_code}")
        try:
            print(f"  Detail: {response.json()}")
        except Exception:
            print(f"  Response: {response.text}")
        response.raise_for_status()

    result = response.json()
    print(f"  Result: {result}")
    return result


def generate_video(
    base_url: str,
    prompt: str,
    output_dir: str,
    size: str = "720x1280",
    num_frames: int = 81,
    test_name: str = "test",
    poll_interval: int = 10,
    max_wait: int = 1800,  # 30 minutes
) -> str:
    """Generate a video via REST API and wait for completion."""
    url = f"{base_url}/v1/videos"
    payload = {
        "prompt": prompt,
        "size": size,
        "num_frames": num_frames,
    }

    print(f"\n{'='*60}")
    print(f"Generating video: {test_name}")
    print(f"Prompt: {prompt}")
    print(f"{'='*60}")

    start_time = time.time()

    # Submit the generation request
    response = requests.post(url, json=payload, timeout=120)

    if response.status_code != 200:
        print(f"Error: {response.status_code}")
        try:
            print(f"Detail: {response.json()}")
        except Exception:
            print(f"Response: {response.text}")
        response.raise_for_status()

    result = response.json()
    video_id = result.get("id")
    status = result.get("status")
    print(f"Video submitted with ID: {video_id}, status: {status}")

    # If already completed (sync mode), return immediately
    if status == "completed":
        elapsed = time.time() - start_time
        print(f"Generation completed in {elapsed:.2f}s")
    else:
        # Poll for completion
        print(f"Waiting for generation to complete (max {max_wait}s)...")
        while True:
            elapsed = time.time() - start_time
            if elapsed > max_wait:
                raise TimeoutError(f"Video generation timed out after {max_wait}s")

            time.sleep(poll_interval)

            # Check status
            try:
                status_response = requests.get(f"{base_url}/v1/videos", timeout=30)
                if status_response.status_code == 200:
                    videos = status_response.json().get("data", [])
                    video_info = next(
                        (v for v in videos if v.get("id") == video_id), None
                    )
                    if video_info:
                        status = video_info.get("status")
                        if status == "completed":
                            print(f"\nGeneration completed in {elapsed:.2f}s")
                            result = video_info
                            break
                        elif status == "failed":
                            error = video_info.get("error", "Unknown error")
                            raise RuntimeError(f"Video generation failed: {error}")
                        else:
                            print(
                                f"  Status: {status}, elapsed: {elapsed:.1f}s", end="\r"
                            )
            except requests.exceptions.RequestException as e:
                print(f"  Warning: Status check failed: {e}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(output_dir, f"{test_name}_{timestamp}.json")

    with open(output_path, "w") as f:
        import json

        json.dump(
            {
                "test_name": test_name,
                "prompt": prompt,
                "generation_time": elapsed,
                "result": result,
            },
            f,
            indent=2,
        )

    print(f"Result saved to: {output_path}")
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Merge/Unmerge functionality test")
    parser.add_argument("--host", type=str, default="localhost", help="Server host")
    parser.add_argument("--port", type=int, default=30010, help="Server port")
    parser.add_argument(
        "--output-dir", type=str, default="./e2e_lora_outputs", help="Output directory"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="A girl with blue braided hair in Arcane style, Jinx character, portrait shot, cinematic lighting",
        help="Prompt for generation",
    )
    parser.add_argument("--size", type=str, default="720x1280", help="Video size WxH")
    parser.add_argument("--num-frames", type=int, default=81, help="Number of frames")

    args = parser.parse_args()
    base_url = f"http://{args.host}:{args.port}"

    os.makedirs(args.output_dir, exist_ok=True)

    print("\n" + "=" * 60)
    print("TEST 5: MERGE/UNMERGE FUNCTIONALITY")
    print("=" * 60)
    print(f"Server: {base_url}")
    print("\nThis test verifies merge/unmerge operations:")
    print("  1. Set LoRA and generate (WITH LoRA)")
    print("  2. Unmerge and generate (WITHOUT LoRA)")
    print("  3. Re-merge and generate (WITH LoRA again)")

    # Step 1: Download and set LoRA
    print("\n" + "-" * 60)
    print("Step 1: Setting up LoRA")
    print("-" * 60)

    # First unmerge any existing LoRA to start fresh
    try:
        unmerge_lora(base_url, "transformer")
    except Exception:
        pass
    try:
        unmerge_lora(base_url, "transformer_2")
    except Exception:
        pass

    high_path, low_path = download_lora_files()

    set_lora(base_url, "arcane_high", high_path, "transformer")
    set_lora(base_url, "arcane_low", low_path, "transformer_2")

    # Step 2: Generate with LoRA merged
    print("\n" + "-" * 60)
    print("Step 2: Generate WITH LoRA (merged)")
    print("-" * 60)
    generate_video(
        base_url=base_url,
        prompt=args.prompt,
        output_dir=args.output_dir,
        size=args.size,
        num_frames=args.num_frames,
        test_name="5a_with_lora_merged",
    )

    # Step 3: Unmerge LoRA
    print("\n" + "-" * 60)
    print("Step 3: Unmerging LoRA weights")
    print("-" * 60)
    unmerge_lora(base_url, "transformer")
    unmerge_lora(base_url, "transformer_2")

    # Step 4: Generate without LoRA (unmerged)
    print("\n" + "-" * 60)
    print("Step 4: Generate WITHOUT LoRA (unmerged)")
    print("-" * 60)
    generate_video(
        base_url=base_url,
        prompt=args.prompt,
        output_dir=args.output_dir,
        size=args.size,
        num_frames=args.num_frames,
        test_name="5b_without_lora_unmerged",
    )

    # Step 5: Re-merge LoRA
    print("\n" + "-" * 60)
    print("Step 5: Re-merging LoRA weights")
    print("-" * 60)
    merge_lora(base_url, "transformer")
    merge_lora(base_url, "transformer_2")

    # Step 6: Generate with LoRA re-merged
    print("\n" + "-" * 60)
    print("Step 6: Generate WITH LoRA (re-merged)")
    print("-" * 60)
    generate_video(
        base_url=base_url,
        prompt=args.prompt,
        output_dir=args.output_dir,
        size=args.size,
        num_frames=args.num_frames,
        test_name="5c_with_lora_remerged",
    )

    print("\n" + "=" * 60)
    print("TEST 5 COMPLETED: Merge/Unmerge test finished")
    print("=" * 60)
    print("\nExpected results:")
    print("  - 5a_with_lora_merged: Should show Arcane/Jinx style")
    print("  - 5b_without_lora_unmerged: Should look like baseline (no style)")
    print("  - 5c_with_lora_remerged: Should match 5a (style restored)")
    print("\nIf 5b looks different from 5a/5c, unmerge is working correctly!")


if __name__ == "__main__":
    main()
