import torch
for tag in ["5.0pct_n100_tc75_s6", "1.0pct_n100_tc200_s6", "0.1pct_n100_tc200_s6"]:
    path = f"experiments/07_ddpm_composite/bulk_eval/results/{tag}/summary.pt"
    s = torch.load(path, map_location="cpu", weights_only=False)
    pct = tag.split("pct")[0]
    print(f"\n=== {pct}% coverage, 100 samples ===")
    for m in s["summary"]:
        st = s["summary"][m]
        print(f"  {m:>12}: {st}  wins={s['wins'][m]}")
