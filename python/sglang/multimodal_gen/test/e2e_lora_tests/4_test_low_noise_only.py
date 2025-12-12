#!/usr/bin/env python3
"""
Test 4: Low-noise transformer LoRA only

This test:
1. Unmerges any existing LoRA
2. Sets LoRA ONLY for the low-noise transformer (target="transformer_2")
3. Generates a video to see the effect of partial LoRA

The low-noise transformer handles later denoising steps, affecting
fine details and textures.

Usage:
    python 4_test_low_noise_only.py [--host HOST] [--port PORT]
"""

import argparse
import os
import time
from datetime import datetime

import requests
from huggingface_hub import hf_hub_download

LORA_REPO = "Cseti/wan2.2-14B-Arcane_Jinx-lora-v1"
LOW_NOISE_LORA = "985347-wan22_14B-low-Nfj1nx-e65.safetensors"


def download_lora_files():
    """Download LoRA files from HuggingFace."""
    print(f"\nDownloading LoRA files from {LORA_REPO}...")
    low_path = hf_hub_download(repo_id=LORA_REPO, filename=LOW_NOISE_LORA)
    print(f"  Low noise LoRA: {low_path}")
    return low_path


def unmerge_lora(base_url: str, target: str = "all"):
    """Unmerge LoRA weights via REST API."""
    url = f"{base_url}/v1/unmerge_lora_weights"
    payload = {"target": target}
    print(f"\nUnmerging LoRA weights (target: {target})...")

    response = requests.post(url, json=payload, timeout=120)

    if response.status_code != 200:
        print(f"  Note: {response.status_code} - {response.text}")
        return None

    result = response.json()
    print(f"  Result: {result}")
    return result


def set_lora(base_url: str, lora_nickname: str, lora_path: str, target: str):
    """Set LoRA adapter via REST API."""
    url = f"{base_url}/v1/set_lora"
    payload = {
        "lora_nickname": lora_nickname,
        "lora_path": lora_path,
        "target": target,
    }
    print(f"\nSetting LoRA: {lora_nickname} -> {target}")
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
            error_detail = response.json()
            print(f"Detail: {error_detail}")
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
    parser = argparse.ArgumentParser(description="Low-noise transformer LoRA only test")
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
    print("TEST 4: LOW-NOISE TRANSFORMER LoRA ONLY")
    print("=" * 60)
    print(f"Server: {base_url}")
    print("\nThis tests LoRA applied ONLY to the low-noise transformer.")
    print("The low-noise transformer affects later denoising steps,")
    print("controlling fine details and textures.")

    # Step 1: Unmerge any existing LoRA
    print("\n" + "-" * 60)
    print("Step 1: Unmerging any existing LoRA")
    print("-" * 60)
    unmerge_lora(base_url, "transformer")
    unmerge_lora(base_url, "transformer_2")

    # Step 2: Download LoRA file
    print("\n" + "-" * 60)
    print("Step 2: Downloading LoRA file")
    print("-" * 60)
    low_path = download_lora_files()

    # Step 3: Set LoRA for low-noise transformer ONLY
    print("\n" + "-" * 60)
    print("Step 3: Setting LoRA for LOW-NOISE transformer only")
    print("-" * 60)
    set_lora(
        base_url=base_url,
        lora_nickname="arcane_low_only",
        lora_path=low_path,
        target="transformer_2",  # ONLY low-noise transformer
    )

    # Step 4: Generate video
    print("\n" + "-" * 60)
    print("Step 4: Generating video with LOW-NOISE LoRA only")
    print("-" * 60)
    generate_video(
        base_url=base_url,
        prompt=args.prompt,
        output_dir=args.output_dir,
        size=args.size,
        num_frames=args.num_frames,
        test_name="4_low_noise_lora_only",
    )

    print("\n" + "=" * 60)
    print("TEST 4 COMPLETED: Low-noise LoRA only video generated")
    print("=" * 60)
    print("\nCompare with:")
    print("  - Baseline (no LoRA): detail/texture differences")
    print("  - Full LoRA: see effect of missing high-noise LoRA")
    print("  - High-noise only: different aspects of style")
    print("\nNext: Run 5_test_merge_unmerge.py to test merge/unmerge functionality")


if __name__ == "__main__":
    main()
