"""Part 2B: official offline Video Depth Anything; --metric selects metric weights."""
from pipelines._model_entry import run_model


def main(argv=None):
    return run_model("vda", argv)


if __name__ == "__main__":
    main()
