# train.py
import argparse
import os
import torch
from collections import Counter
import evaluator_stream

from loaddatasets import loadmagic, loadinsects
from model import OLD3S_Shallow

def filter_insects_pair(x, y, class_a: int, class_b: int):
    y = y.view(-1).long()
    mask = (y == class_a) | (y == class_b)
    x2 = x[mask]
    y2 = y[mask]
    # relabel: class_a -> 0, class_b -> 1
    ybin = (y2 == class_b).long()
    return x2, ybin



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
    parser.add_argument("-class_a", type=int, default=None, help="INSECTS class id A for one-vs-one binary")
    parser.add_argument("-class_b", type=int, default=None, help="INSECTS class id B for one-vs-one binary")

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

    if dataname == "insects" and args.class_a is not None and args.class_b is not None:
        a = int(args.class_a)
        b = int(args.class_b)

        x_S1, y_S1 = filter_insects_pair(x_S1, y_S1, a, b)
        x_S2, y_S2 = filter_insects_pair(x_S2, y_S2, a, b)
        counts = Counter(list(y_S1) + list(y_S2))
        maj_class = max(counts, key=counts.get)
        min_class = min(counts, key=counts.get)

        evaluator_stream.set_global_min_maj(min_class, maj_class)
        print("[INFO] Global maj/min:", maj_class, min_class, dict(counts))

        print(f"[INFO] INSECTS one-vs-one enabled: {a} vs {b}")
        print("[INFO] S1 counts:", torch.bincount(y_S1).tolist())
        print("[INFO] S2 counts:", torch.bincount(y_S2).tolist())

        # hard safety check
        assert int(torch.unique(torch.cat([y_S1, y_S2])).numel()) == 2

        # ✅ unique output folder per pair (prevents overwriting results)
        csv_stem = os.path.splitext(os.path.basename(args.insects_csv))[0]
        path = f"parameter_insects__{csv_stem}__pair_{a}vs{b}__adwin"
        print(f"[INFO] Output path set to: {path}")


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
