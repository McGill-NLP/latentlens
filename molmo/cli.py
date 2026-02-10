"""
Command-line interface for LatentLens.

Usage:
    latentlens --help
    latentlens extract-embeddings --model <model> --output <dir>
    latentlens run-latentlens --ckpt-path <path> --contextual-dir <dir>
    latentlens run-logitlens --ckpt-path <path>
"""

import argparse
import sys


def main():
    parser = argparse.ArgumentParser(
        prog="latentlens",
        description="LatentLens: Interpreting Visual Tokens in Vision-Language Models",
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Extract embeddings
    extract_parser = subparsers.add_parser(
        "extract-embeddings",
        help="Extract contextual embeddings from an LLM"
    )
    extract_parser.add_argument("--model", required=True, help="HuggingFace model ID")
    extract_parser.add_argument("--layers", nargs="+", type=int, help="Layers to extract")
    extract_parser.add_argument("--output", required=True, help="Output directory")

    # Run LatentLens
    latentlens_parser = subparsers.add_parser(
        "run-latentlens",
        help="Run LatentLens analysis (contextual nearest neighbors)"
    )
    latentlens_parser.add_argument("--ckpt-path", required=True, help="Checkpoint path")
    latentlens_parser.add_argument("--contextual-dir", required=True, help="Contextual embeddings directory")
    latentlens_parser.add_argument("--num-images", type=int, default=100, help="Number of images")
    latentlens_parser.add_argument("--output-dir", required=True, help="Output directory")

    # Run LogitLens
    logitlens_parser = subparsers.add_parser(
        "run-logitlens",
        help="Run LogitLens analysis"
    )
    logitlens_parser.add_argument("--ckpt-path", required=True, help="Checkpoint path")
    logitlens_parser.add_argument("--num-images", type=int, default=100, help="Number of images")
    logitlens_parser.add_argument("--output-dir", required=True, help="Output directory")

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    # Dispatch to scripts
    if args.command == "extract-embeddings":
        from reproduce.scripts.extract_embeddings import main as extract_main
        sys.argv = [
            "extract_embeddings.py",
            "--model", args.model,
            "--layers", *[str(l) for l in args.layers],
            "--output", args.output
        ]
        extract_main()
    elif args.command == "run-latentlens":
        print(f"To run LatentLens analysis, use:")
        print(f"  python scripts/run_latentlens.py --ckpt-path {args.ckpt_path} --contextual-dir {args.contextual_dir} --num-images {args.num_images} --output-dir {args.output_dir}")
    elif args.command == "run-logitlens":
        print(f"To run LogitLens analysis, use:")
        print(f"  torchrun --nproc_per_node=4 scripts/run_logitlens.py --ckpt-path {args.ckpt_path} --num-images {args.num_images} --output-dir {args.output_dir}")


if __name__ == "__main__":
    main()
