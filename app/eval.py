from src.eval.predictor import main


def _main(argv: list[str] | None = None) -> None:
    main(argv)

if __name__ == "__main__":
    # Allows running via: python -m app.api
    _main()