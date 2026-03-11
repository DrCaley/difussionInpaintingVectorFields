"""Count parameters per section of Helmholtz UNet."""
import torch
from ddpm.neural_networks.unets.unet_helmholtz import MyUNet_Helmholtz

net = MyUNet_Helmholtz(n_steps=250, time_emb_dim=256)

sections = {
    "time_embed": ["time_embed_table", "time_mlp"],
    "enc1+down1": ["enc1", "down1"],
    "enc2+down2": ["enc2", "down2"],
    "enc3+down3": ["enc3", "down3"],
    "enc4+down4": ["enc4", "down4"],
    "bottleneck": ["mid"],
    "dec4+up4": ["up4", "dec4"],
    "dec3+up3": ["up3", "dec3"],
    "dec2+up2": ["up2", "dec2"],
    "dec1+up1": ["up1", "dec1"],
    "psi_head": ["psi_norm", "psi_conv"],
    "phi_head": ["phi_norm", "phi_conv"],
}

total = 0
encoder_p = 0
decoder_p = 0
dec_hi_p = 0
dec_lo_p = 0
head_p = 0

for sec_name, mod_names in sections.items():
    sec_p = 0
    for mn in mod_names:
        if hasattr(net, mn):
            m = getattr(net, mn)
            sec_p += sum(p.numel() for p in m.parameters() if p.requires_grad)
    total += sec_p
    is_enc = sec_name.startswith("enc") or sec_name in ("time_embed", "bottleneck")
    if is_enc:
        encoder_p += sec_p
    else:
        decoder_p += sec_p
    if "dec4" in sec_name or "dec3" in sec_name:
        dec_lo_p += sec_p
    if "dec2" in sec_name or "dec1" in sec_name:
        dec_hi_p += sec_p
    if "head" in sec_name:
        head_p += sec_p
    print(f"  {sec_name:15s}: {sec_p:>10,}  ({sec_p/1e6:.2f}M)")

print()
print(f"  ENCODER+MID:     {encoder_p:>10,}  ({encoder_p/1e6:.2f}M)")
print(f"  DECODER+HEADS:   {decoder_p:>10,}  ({decoder_p/1e6:.2f}M)")
print(f"  TOTAL:           {total:>10,}  ({total/1e6:.2f}M)")
print()
print(f"  Dec low-res  (dec4+dec3, 256ch, attn): {dec_lo_p:>10,}  ({dec_lo_p/1e6:.2f}M)")
print(f"  Dec high-res (dec2+dec1, 128/64ch):    {dec_hi_p:>10,}  ({dec_hi_p/1e6:.2f}M)")
print(f"  Heads (psi+phi norms+convs):           {head_p:>10,}  ({head_p/1e6:.2f}M)")
print()

opt_b_extra = decoder_p
opt_b = total + opt_b_extra
print(f"  Option B (full dup decoder):    {opt_b:>10,}  ({opt_b/1e6:.2f}M)  +{opt_b_extra/1e6:.1f}M  (+{opt_b_extra/total*100:.0f}%)")

opt_c_extra = dec_hi_p + head_p
opt_c = total + opt_c_extra
print(f"  Option C (dup dec2+dec1+heads): {opt_c:>10,}  ({opt_c/1e6:.2f}M)  +{opt_c_extra/1e6:.1f}M  (+{opt_c_extra/total*100:.0f}%)")

print()
print("Decoder resolution map:")
print("  dec4: [B,256, 8,16]  → [B,256, 8,16]   self-attn, 256ch — expensive")
print("  dec3: [B,256,16,32]  → [B,128,16,32]   self-attn, 256→128ch")
print("  dec2: [B,128,32,64]  → [B, 64,32,64]   no attn, cheap")
print("  dec1: [B, 64,64,128] → [B, 64,64,128]  no attn, cheap ← full res")
