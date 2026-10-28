"""Part 1B: independent Depth Anything V2 relative prediction per frame."""
from pipelines._model_entry import run_model


def main(argv=None):
    return run_model("dav2", argv)


if __name__ == "__main__":
    main()
