"""
Build a pre-computed LatentLens index for a model and upload to HuggingFace Hub.

Usage:
    CUDA_VISIBLE_DEVICES=0 python scripts/build_precomputed_index.py \
        --model meta-llama/Llama-3.1-8B \
        --repo McGill-NLP/latentlens-llama3.1-8b
"""

import argparse
import time

import torch

import latentlens


def main():
    parser = argparse.ArgumentParser(description="Build and upload a LatentLens index")
    parser.add_argument("--model", type=str, required=True, help="HuggingFace model ID")
    parser.add_argument("--corpus", type=str, default="concepts.txt", help="Path to corpus file")
    parser.add_argument("--output-dir", type=str, default=None, help="Local save dir (default: indices/{model_short})")
    parser.add_argument("--repo", type=str, default=None, help="HuggingFace repo ID for upload (e.g. McGill-NLP/latentlens-llama3.1-8b)")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-contexts", type=int, default=50)
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["float32", "float16", "bfloat16"])
    parser.add_argument("--storage-dtype", type=str, default="float16", choices=["float32", "float16"])
    args = parser.parse_args()

    dtype_map = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}
    model_dtype = dtype_map[args.dtype]
    storage_dtype = dtype_map[args.storage_dtype]

    # Default output dir
    if args.output_dir is None:
        model_short = args.model.split("/")[-1].lower()
        args.output_dir = f"indices/{model_short}"

    print(f"Model: {args.model}")
    print(f"Corpus: {args.corpus}")
    print(f"Output: {args.output_dir}")
    print(f"Model dtype: {args.dtype}")
    print(f"Storage dtype: {args.storage_dtype}")
    print(f"Batch size: {args.batch_size}")
    print(f"Max contexts per token: {args.max_contexts}")
    print()

    # Build index
    t0 = time.time()
    index = latentlens.build_index(
        args.model,
        corpus=args.corpus,
        batch_size=args.batch_size,
        max_contexts_per_token=args.max_contexts,
        dtype=model_dtype,
    )
    elapsed = time.time() - t0
    print(f"\nIndex built in {elapsed:.0f}s ({elapsed/60:.1f} min)")
    print(f"  {index}")

    # Save locally
    print(f"\nSaving to {args.output_dir} (storage_dtype={args.storage_dtype})...")
    index.save(args.output_dir, storage_dtype=storage_dtype)
    print("Saved.")

    # Upload to HuggingFace
    if args.repo:
        print(f"\nUploading to {args.repo}...")
        from huggingface_hub import HfApi
        api = HfApi()

        # Create repo if needed
        api.create_repo(args.repo, repo_type="model", exist_ok=True)

        # Upload the entire directory
        api.upload_folder(
            folder_path=args.output_dir,
            repo_id=args.repo,
            repo_type="model",
            commit_message=f"Upload LatentLens index for {args.model}",
        )
        print(f"Uploaded to https://huggingface.co/{args.repo}")

    print("\nDone!")


if __name__ == "__main__":
    main()
