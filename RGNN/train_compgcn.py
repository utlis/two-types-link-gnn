import sys

from train import main


def ensure_compgcn_encoder(argv):
    """Use CompGCN unless the caller explicitly sets an encoder."""
    if "--encoder" in argv:
        return argv
    return argv + ["--encoder", "compgcn"]


if __name__ == "__main__":
    sys.argv = ensure_compgcn_encoder(sys.argv)
    main()
