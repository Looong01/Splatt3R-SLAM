"""One head against base on every family, per-image, with bootstrap CIs.

Kimi's round-22 correction: the 5x5 matrix's replica column used the head that
was trained at the wrong resolution (512x384 instead of the deployment's
512x288, 17.55), so that column has to be recomputed before it can be read. And
the interesting pattern there is not one cell in twenty -- it is a ROW: the
replica head improved psnr on 4/4 foreign families. A row can be signal; a cell
is noise. But neither can be called without a per-cell floor, and the SLAM
polish floor (17.51) does not apply to this harness.

So this scores per image and bootstraps the per-image deltas, which turns "is
this noise?" from a judgement into a measurement, for free.
"""
import argparse, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import torch
import exp_head_only as E


def per_image(model, loader):
    out = []
    with torch.no_grad():
        for batch in loader:
            b = E.batch_to_dev(batch)
            v1, v2 = b["context"]
            _, _, h, w = v1["img"].shape
            p1, p2 = model.forward(v1, v2)
            color, _ = model.decoder(b, p1, p2, (h, w))
            tgt = b["target"][0]["original_img"]
            mse = float(torch.mean((color[0] - tgt) ** 2))
            lp = float(model.lpips_criterion(color[0] * 2 - 1, tgt * 2 - 1).mean())
            out.append((mse, lp))
    return np.array(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--head", required=True)
    ap.add_argument("--label", default="head")
    ap.add_argument("--families", default="tum,7-scenes,euroc,eth3d,replica")
    ap.add_argument("--boot", type=int, default=2000)
    args = ap.parse_args()
    head = args.head if os.path.isabs(args.head) else os.path.join(
        E.REPO_ROOT, args.head)

    print(f"{'family':10s} {'n':>4} {'base psnr/lpips':>18} {args.label+' psnr/lpips':>18}"
          f" {'d psnr [95% CI]':>26} {'d lpips %':>20}")
    for fam in args.families.split(","):
        E.configure_family(fam, 2, 1e-5)
        _, va = E.build_loaders()
        model = E.MAST3RGaussiansHeadOnly.load_from_checkpoint(
            E.BASE_CKPT, map_location=E.DEV).to(E.DEV)
        model.decoder.spatial_stride = E.STRIDE
        model.eval()
        base_sd = {k: v.clone() for k, v in model.encoder.state_dict().items()
                   if "gaussian_dpt" in k}
        a = per_image(model, va)
        sd = torch.load(head, map_location=E.DEV)
        model.encoder.load_state_dict(sd, strict=False)
        model.eval()
        b = per_image(model, va)
        model.encoder.load_state_dict(base_sd, strict=False)

        n = len(a)
        rng = np.random.default_rng(0)
        idx = rng.integers(0, n, (args.boot, n))
        dp = np.array([-10*np.log10(b[i, 0].mean()) + 10*np.log10(a[i, 0].mean())
                       for i in idx])
        dl = np.array([(b[i, 1].mean() - a[i, 1].mean()) / a[i, 1].mean() * 100
                       for i in idx])
        bp = -10*np.log10(a[:, 0].mean()); hp = -10*np.log10(b[:, 0].mean())
        print(f"{fam:10s} {n:>4} {bp:9.2f}/{a[:,1].mean():.3f}"
              f" {hp:12.2f}/{b[:,1].mean():.3f}"
              f" {hp-bp:+9.2f} [{np.percentile(dp,2.5):+.2f},{np.percentile(dp,97.5):+.2f}]"
              f" {(b[:,1].mean()-a[:,1].mean())/a[:,1].mean()*100:+8.1f}"
              f" [{np.percentile(dl,2.5):+.1f},{np.percentile(dl,97.5):+.1f}]", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
