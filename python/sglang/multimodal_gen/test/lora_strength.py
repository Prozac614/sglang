#!/usr/bin/env python3
"""
Test: LoRA Strength Parameter Validation

This test verifies that the strength parameter in set_lora and merge_lora_weights
methods works correctly:

1. Set LoRA with strength=0.5 and generate (weak LoRA effect)
2. Set LoRA with strength=1.0 and generate (normal LoRA effect)
3. Merge LoRA with strength=1.5 and generate (amplified LoRA effect)

The three generated videos should show visibly different style intensities,
confirming the strength parameter is effective.

Usage:
    python lora_strength.py [--host HOST] [--port PORT]

Prerequisites:
    - Server running with model: Wan-AI/Wan2.1-T2V-1.3B-Diffusers
    - Example: sglang serve --model-path Wan-AI/Wan2.1-T2V-1.3B-Diffusers
"""

import argparse
import json
import os
import time
from datetime import datetime

import requests

# Default LoRA for Wan2.1-T2V-1.3B
DEFAULT_LORA_PATH = "Cseti/Wan-LoRA-Arcane-Jinx-v1"
DEFAULT_LORA_NICKNAME = "arcane"


def set_lora(
    base_url: str,
    lora_nickname: str,
    lora_path: str,
    target: str = "all",
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
    print(f"\nSetting LoRA: {lora_nickname} (target: {target}, strength: {strength})")

    response = requests.post(url, json=payload, timeout=300)

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


def merge_lora(base_url: str, target: str = "all", strength: float = 1.0):
    """Merge LoRA weights via REST API with strength parameter."""
    url = f"{base_url}/v1/merge_lora_weights"
    payload = {"target": target, "strength": strength}
    print(f"\nMerging LoRA weights (target: {target}, strength: {strength})...")

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
    size: str = "480x832",
    num_frames: int = 17,
    test_name: str = "test",
    strength: float = 1.0,
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
    print(f"Strength: {strength}")
    print(f"Prompt: {prompt}")
    print(f"Size: {size}, Frames: {num_frames}")
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

    elapsed = time.time() - start_time
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(output_dir, f"{test_name}_{timestamp}.json")

    with open(output_path, "w") as f:
        json.dump(
            {
                "test_name": test_name,
                "prompt": prompt,
                "strength": strength,
                "size": size,
                "num_frames": num_frames,
                "generation_time": elapsed,
                "result": result,
            },
            f,
            indent=2,
        )

    print(f"Result saved to: {output_path}")
    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Test LoRA strength parameter functionality"
    )
    parser.add_argument("--host", type=str, default="localhost", help="Server host")
    parser.add_argument("--port", type=int, default=30010, help="Server port")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./lora_strength_outputs",
        help="Output directory",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="A girl with blue braided hair in Arcane style, Jinx character, portrait shot, cinematic lighting",
        help="Prompt for generation",
    )
    parser.add_argument(
        "--size", type=str, default="480x832", help="Video size WxH (smaller for speed)"
    )
    parser.add_argument(
        "--num-frames", type=int, default=17, help="Number of frames (fewer for speed)"
    )
    parser.add_argument(
        "--lora-path",
        type=str,
        default=DEFAULT_LORA_PATH,
        help="LoRA adapter path (HF repo or local)",
    )

    args = parser.parse_args()
    base_url = f"http://{args.host}:{args.port}"

    os.makedirs(args.output_dir, exist_ok=True)

    print("\n" + "=" * 60)
    print("TEST: LORA STRENGTH PARAMETER VALIDATION")
    print("=" * 60)
    print(f"Server: {base_url}")
    print(f"LoRA: {args.lora_path}")
    print("\nThis test verifies the strength parameter in set_lora/merge_lora:")
    print("  1. strength=0.5 (weak LoRA effect)")
    print("  2. strength=1.0 (normal LoRA effect)")
    print("  3. strength=1.5 via merge_lora_weights (amplified LoRA effect)")

    # Step 1: Clean up any existing LoRA state
    print("\n" + "-" * 60)
    print("Step 0: Cleaning up existing LoRA state")
    print("-" * 60)
    try:
        unmerge_lora(base_url, "all")
    except Exception as e:
        print(f"  Note: Unmerge skipped (may not have been merged): {e}")

    # Step 2: Test strength=0.5
    print("\n" + "-" * 60)
    print("Step 1: Test set_lora with strength=0.5")
    print("-" * 60)
    set_lora(
        base_url=base_url,
        lora_nickname=DEFAULT_LORA_NICKNAME,
        lora_path=args.lora_path,
        target="all",
        strength=0.5,
    )
    generate_video(
        base_url=base_url,
        prompt=args.prompt,
        output_dir=args.output_dir,
        size=args.size,
        num_frames=args.num_frames,
        test_name="strength_0.5",
        strength=0.5,
    )

    # Step 3: Test strength=1.0
    print("\n" + "-" * 60)
    print("Step 2: Test set_lora with strength=1.0")
    print("-" * 60)
    unmerge_lora(base_url, "all")
    set_lora(
        base_url=base_url,
        lora_nickname=DEFAULT_LORA_NICKNAME,
        lora_path=args.lora_path,
        target="all",
        strength=1.0,
    )
    generate_video(
        base_url=base_url,
        prompt=args.prompt,
        output_dir=args.output_dir,
        size=args.size,
        num_frames=args.num_frames,
        test_name="strength_1.0",
        strength=1.0,
    )

    # Step 4: Test strength=1.5 via merge_lora_weights
    print("\n" + "-" * 60)
    print("Step 3: Test merge_lora_weights with strength=1.5")
    print("-" * 60)
    unmerge_lora(base_url, "all")
    # Use merge_lora_weights to apply with different strength
    merge_lora(base_url, target="all", strength=1.5)
    generate_video(
        base_url=base_url,
        prompt=args.prompt,
        output_dir=args.output_dir,
        size=args.size,
        num_frames=args.num_frames,
        test_name="strength_1.5_via_merge",
        strength=1.5,
    )

    # Summary
    print("\n" + "=" * 60)
    print("TEST COMPLETED: LoRA Strength Parameter Validation")
    print("=" * 60)
    print("\nGenerated outputs:")
    print(f"  - strength_0.5: Weak LoRA effect (50%)")
    print(f"  - strength_1.0: Normal LoRA effect (100%)")
    print(f"  - strength_1.5_via_merge: Amplified LoRA effect (150%)")
    print(f"\nOutput directory: {args.output_dir}")
    print("\nExpected results:")
    print("  - The three videos should show visibly different style intensities")
    print("  - strength_0.5 should have subtle Arcane/Jinx style")
    print("  - strength_1.0 should have normal Arcane/Jinx style")
    print("  - strength_1.5 should have exaggerated Arcane/Jinx style")
    print("\nIf the videos look different, the strength parameter is working!")


if __name__ == "__main__":
    main()
