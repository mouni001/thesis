# train.py
import argparse
import torch

from loaddatasets import loadmagic, loadinsects
from model import OLD3S_Shallow


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-DataName", type=str, required=True,
                        help="magic, insects")
    parser.add_argument("-AutoEncoder", type=str, default="AE")
    parser.add_argument("-beta", type=float, default=0.9)
    parser.add_argument("-eta", type=float, default=-0.001)
    parser.add_argument("-learningrate", type=float, default=1e-3)
    parser.add_argument("-RecLossFunc", type=str, default="bce")
    parser.add_argument("-T1", type=int, default=5000)
    parser.add_argument("-t", type=int, default=1000)
    parser.add_argument("-insects_csv", type=str, default="./data/incremental-abrupt_imbalanced.csv")
    args = parser.parse_args()

    dataname = args.DataName.strip().lower()

    if dataname == "magic":
        x_S1, y_S1, x_S2, y_S2 = loadmagic()
        path = "parameter_magic"
    elif dataname == "insects":
        x_S1, y_S1, x_S2, y_S2 = loadinsects(args.insects_csv, split_ratio=0.8)
        path = "parameter_insects"
    else:
        raise ValueError(f"Unsupported DataName: {args.DataName}")

    # IMPORTANT: dimension1 is S1 feature dim, dimension2 is S2 feature dim
    dimension1 = int(x_S1.shape[1])
    dimension2 = int(x_S2.shape[1])

    # make sure labels are long
    y_S1 = y_S1.view(-1).long()
    y_S2 = y_S2.view(-1).long()

    # Choose T1/t safely relative to available data:
    # S1 available = len(x_S1), S2 available = len(x_S2)
    # We run B = T1 - t from S1, then t from S2.
    # So must have B <= len(S1) and t <= len(S2).
    t = min(args.t, len(x_S2))
    B = min(args.T1 - t, len(x_S1))
    T1 = B + t

    print(f"[INFO] Using T1={T1} (B={B} from S1, t={t} from S2)")
    print(f"[INFO] dims: S1={dimension1}, S2={dimension2}")

    model = OLD3S_Shallow(
        x_S1, y_S1,
        x_S2, y_S2,
        T1=T1, t=t,
        dimension1=dimension1, dimension2=dimension2,
        path=path,
        lr=args.learningrate,
        b=args.beta,
        eta=args.eta,
        RecLossFunc=args.RecLossFunc,
        detector_type="adwin",
    )

    model.FirstPeriod()
    print("[OK] Done.")


if __name__ == "__main__":
    main()
