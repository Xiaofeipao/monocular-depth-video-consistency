"""Keep each Part's model choice explicit while sharing the inference CLI."""
import argparse
import sys


def run_model(model, argv=None):
    from src.infer import main

    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument("--model", choices=[model], default=model)
    _, remaining = parser.parse_known_args(sys.argv[1:] if argv is None else argv)
    return main(["--model", model, *remaining])
