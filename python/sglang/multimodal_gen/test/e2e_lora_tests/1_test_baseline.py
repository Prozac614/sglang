#!/usr/bin/env python3
"""
Test 1: Baseline test without LoRA

This test generates a video WITHOUT any LoRA adapter to establish a baseline
for comparison with LoRA-enabled generations.

Usage:
    python 1_test_baseline.py [--host HOST] [--port PORT]
"""

import argparse
import os
import time
from datetime import datetime

import requests


def generate_video(
    base_url: str,
    prompt: str,
    output_dir: str,
    size: str = "720x1280",
    num_frames: int = 81,
    test_name: str = "baseline",
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
    print(f"Size: {size}, Frames: {num_frames}")
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

    # Save the output info
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
    parser = argparse.ArgumentParser(description="Baseline test without LoRA")
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

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    print("\n" + "=" * 60)
    print("TEST 1: BASELINE (No LoRA)")
    print("=" * 60)
    print(f"Server: {base_url}")
    print(f"Output: {args.output_dir}")

    # Generate baseline video
    generate_video(
        base_url=base_url,
        prompt=args.prompt,
        output_dir=args.output_dir,
        size=args.size,
        num_frames=args.num_frames,
        test_name="1_baseline_no_lora",
    )

    print("\n" + "=" * 60)
    print("TEST 1 COMPLETED: Baseline video generated")
    print("=" * 60)
    print("\nNext: Run 2_test_full_lora.py to compare with LoRA-enabled generation")


if __name__ == "__main__":
    main()
