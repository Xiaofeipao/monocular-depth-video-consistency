"""Part 2A: native metric Depth Pro inference, without test-GT alignment."""
from pipelines._model_entry import run_model


def main(argv=None):
    return run_model("depth-pro", argv)


if __name__ == "__main__":
    main()
