# Named DINOv2-L edit specs for the downstream scripts (applied to stock via wsp_edits.build_variant).
# Round 2: attenuate detail instead of deleting it (damped texture units, weak anchoring).
import json
from pathlib import Path
W = Path(__file__).resolve().parent.parent
U14 = [tuple(u) for u in json.load(open(W / "proxy/regunits.json"))]
Z, P = ("zero_units", {"units": U14}), ("pos_lowpass", {"sigma": 1.5})
E12 = [P, Z]
TEX = lambda keep: ("mean_ablate_texture", {"blocks": list(range(7, 24)), "k": 4096, "ratio_min": 0.8, "keep": keep})
MS = lambda g: ("append_mean_subtract_block", {"gamma": g})
AN = lambda g: ("append_guided_block", {"src": 13, "temp": 5, "gamma": g})
VARIANTS = {
    "e12": E12,
    "tex0 (old texture edit)": E12 + [TEX(0.0), MS(1.0)],
    "tex0.3": E12 + [TEX(0.3), MS(1.0)],
    "tex0.5": E12 + [TEX(0.5), MS(1.0)],
    "e12+anchor1": E12 + [AN(1.0), MS(0.5)],
    "e12+anchor2": E12 + [AN(2.0), MS(1.0)],
    "e12+anchor4": E12 + [AN(4.0), MS(2.0)],
    "tex0.5+anchor2": E12 + [TEX(0.5), AN(2.0), MS(1.0)],
}
# Round 3: which base edit costs SPair? Rebuild on the harmless one.
VARIANTS.update({
    "z14": [Z], "poslp1.5": [P], "z7": [("zero_units", {})],
    "z14+tex0.5": [Z, TEX(0.5), MS(1.0)],
    "z14+tex0.5+anchor2": [Z, TEX(0.5), AN(2.0), MS(1.0)],
    "z14+anchor2": [Z, AN(2.0), MS(1.0)],
    "stock+tex0.5+anchor2": [TEX(0.5), AN(2.0), MS(1.0)],
})
# Round 4: 7-neuron base (SPair-neutral), damped texture units, no position blur.
Z7 = ("zero_units", {})
VARIANTS.update({
    "z7+tex0.5": [Z7, TEX(0.5), MS(1.0)],
    "z7+tex0.3": [Z7, TEX(0.3), MS(1.0)],
    "stock+tex0.5": [TEX(0.5), MS(1.0)],
    "stock+tex0.3": [TEX(0.3), MS(1.0)],
    "z7+tex0.5+anchor1": [Z7, TEX(0.5), AN(1.0), MS(0.5)],
})
# Round 5: sharper (higher-temperature) weak anchor on the z7+tex0.5 base.
ANT = lambda g, T: ("append_guided_block", {"src": 13, "temp": T, "gamma": g})
for T in (10, 20):
    for g in (0.5, 1.0):
        VARIANTS[f"z7+tex0.5+anchorT{T}g{g}"] = [Z7, TEX(0.5), ANT(g, T), MS(g / 2)]
VARIANTS["z7+anchorT20g1"] = [Z7, ANT(1.0, 20), MS(0.5)]
VARIANTS["z7+tex0.7"] = [Z7, TEX(0.7), MS(1.0)]
# Round 6: drop the mean-subtract block (it removes image-global context the COCO probe uses).
for k in (0.3, 0.5, 0.7):
    VARIANTS[f"z7+tex{k} noMS"] = [Z7, TEX(k)]
VARIANTS["stock+tex0.3 noMS"] = [TEX(0.3)]
# Round 7: per-block damping (mid 7-16 keep a, late 17-23 keep b), ratio threshold r; no appended blocks.
TEXB = lambda blocks, keep, r: ("mean_ablate_texture", {"blocks": blocks, "k": 4096, "ratio_min": r, "keep": keep})
MID, LATE = list(range(7, 17)), list(range(17, 24))
for r in (0.8, 0.9):
    for a in (1.0, 0.7, 0.5):
        for b in (0.1, 0.3, 0.5):
            spec = ([TEXB(MID, a, r)] if a < 1 else []) + [TEXB(LATE, b, r)]
            VARIANTS[f"mid{a}late{b}r{r}"] = spec
VARIANTS["round1: stock+tex0.3"] = [TEX(0.3)]
# Round 8: positional side channel + local bilateral block on the round-2 pick (TD).
TD = [TEXB(MID, 0.5, 0.9), TEXB(LATE, 0.5, 0.9)]
VARIANTS["TD"] = TD
VARIANTS["TD+reserve"] = TD + [("side_channel", {"A": 2.0, "beta": 0.0, "gamma": 0.0})]
for A in (1.0, 2.0, 4.0):
    for beta in (0.0, 4.0, 8.0):
        for gamma in (1.0, 3.0):
            VARIANTS[f"TD+bil A{A} b{beta} g{gamma}"] = TD + [("side_channel", {"A": A, "beta": beta, "gamma": gamma})]
# Round 9: milder bilateral blends, on TD and on TD2 (mid0.5/late0.3 r0.9, also round-2 eligible).
TD2 = [TEXB(MID, 0.5, 0.9), TEXB(LATE, 0.3, 0.9)]
VARIANTS["TD2"] = TD2
for bn, b in (("TD", TD), ("TD2", TD2)):
    for A in (2.0, 4.0, 8.0):
        for beta in (0.0, 4.0):
            for gamma in (0.3, 0.5):
                VARIANTS[f"{bn}+bil A{A} b{beta} g{gamma}"] = b + [("side_channel", {"A": A, "beta": beta, "gamma": gamma})]
# Round 10 (round 4 of the look effort): rolling-guidance = more programmed bilateral blocks, guide re-read per block.
SCM = lambda more, A=4.0, b=0.0, g=0.5: ("side_channel", {"A": A, "beta": b, "gamma": g, "more": list(more)})
RG = {
    "RG A2b8g1": [(2.0, 8.0, 1.0)],
    "RG A2b8g1x2": [(2.0, 8.0, 1.0)] * 2,
    "RG A2b4g1x2": [(2.0, 4.0, 1.0)] * 2,
    "RG A2b16g1x2": [(2.0, 16.0, 1.0)] * 2,
    "RG A2b8g1x3": [(2.0, 8.0, 1.0)] * 3,
}
for k, more in RG.items():
    VARIANTS[k] = TD2 + [SCM(more)]
VARIANTS["RG A4b0g1+A2b16g1"] = TD2 + [SCM([(2.0, 16.0, 1.0)], g=1.0)]
# Round 11: test-time register (Jiang et al. 2025) programmed through a reserved flag channel (14 register neurons).
TTRU = U14
SCT = lambda more=(), A=4.0, b=0.0, g=0.5: ("side_channel", {"A": A, "beta": b, "gamma": g, "more": list(more), "ttr_units": TTRU})
VARIANTS["TTR TD2 (no filter)"] = TD2 + [SCT(g=0.0)]
VARIANTS["TTR PICK6"] = TD2 + [SCT()]
for k, more in RG.items():
    VARIANTS["TTR " + k] = TD2 + [SCT(more)]
for g in (0.3, 0.5):
    VARIANTS[f"TTR mild A4b8g{g}"] = TD2 + [SCT([(4.0, 8.0, g)])]
# Round 12 (blog-style surgical direction removal): project the positional subspace out of the output stream with an
# exact linear MLP programmed into the last appended block.
P7S = lambda **kw: ("side_channel", {"A": 4.0, "beta": 0.0, "gamma": 0.5, "more": [(4.0, 8.0, 0.3)], "ttr_units": TTRU, **kw})
for k in (4, 16, 64, 256):
    VARIANTS[f"PICK7+pos{k}"] = TD2 + [P7S(pos_k=k)]
VARIANTS["PICK7+pos16 a0.5"] = TD2 + [P7S(pos_k=16, pos_alpha=0.5)]
for k in (8, 16, 32, 64):
    for a in (0.3, 0.5, 0.7):
        VARIANTS.setdefault(f"PICK7+pos{k} a{a}", TD2 + [P7S(pos_k=k, pos_alpha=a)])
VARIANTS["TTR TD2+pos16 a0.5 (no filter)"] = TD2 + [("side_channel", {"A": 4.0, "beta": 0.0, "gamma": 0.0, "ttr_units": TTRU, "pos_k": 16, "pos_alpha": 0.5})]
# Round 13: programmed partial whitening of the output stream (linmap.py) on the round-3 pick (PICK6) and on TD2+TTR.
SCW = lambda b, e, g=0.5: ("side_channel", {"A": 4.0, "beta": 0.0, "gamma": g, "white_beta": b, "white_eps": e})
for b in (0.25, 0.5):
    for e in (0.1, 1.0):
        VARIANTS[f"PICK6+white b{b} e{e}"] = TD2 + [SCW(b, e)]
VARIANTS["TD2+white b0.25 e1.0 (no filter)"] = TD2 + [SCW(0.25, 1.0, g=0.0)]
# Round 14: programmed Fisher discriminant (closed-form metric from SPair TRAIN keypoint pairs) on PICK6.
for b in (0.25, 0.5):
    VARIANTS[f"PICK6+fisher b{b}"] = TD2 + [("side_channel", {"A": 4.0, "beta": 0.0, "gamma": 0.5, "fisher_beta": b})]
for _s, _t in (("video", "v"), ("aug", "a")):                 # label-free Fisher (addendum 5)
    VARIANTS[f"PICK6+{_t}fisher b0.5"] = TD2 + [("side_channel", {"A": 4.0, "beta": 0.0, "gamma": 0.5, "fisher_beta": 0.5,
                                                                 "fisher_src": _s})]
# Round 16 (SELECTION_RULE_v5.md): SUPERVISED Fisher from DAVIS-train object masks, alone and pooled with SPair.
for _src, _nm, _b in (("dmask", "dfisher b0.25", 0.25), ("dmask", "dfisher b0.5", 0.5),
                      ("spair+dmask0.5", "sp+d0.5 fisher b0.5", 0.5), ("spair+dmask1", "sp+d1 fisher b0.5", 0.5)):
    VARIANTS[f"PICK6+{_nm}"] = TD2 + [("side_channel", {"A": 4.0, "beta": 0.0, "gamma": 0.5, "fisher_beta": _b, "fisher_src": _src})]
# Round 16 addendum 1: test-time register + Fisher.
for _src, _nm in (("spair", "fisher b0.5"), ("spair+dmask0.5", "sp+d0.5 fisher b0.5")):
    VARIANTS[f"TTR PICK6+{_nm}"] = TD2 + [("side_channel", {"A": 4.0, "beta": 0.0, "gamma": 0.5, "ttr_units": TTRU,
                                                           "fisher_beta": 0.5, "fisher_src": _src})]
# Round 16 addendum 2: Fisher + AMPLIFIED positional subspace (pos projector with alpha < 0, tail hidden units).
for _a in (-0.5, -1.0):
    VARIANTS[f"PICK6+fisher b0.5+pos16 a{_a}"] = TD2 + [("side_channel", {"A": 4.0, "beta": 0.0, "gamma": 0.5, "fisher_beta": 0.5,
                                                                         "pos_k": 16, "pos_alpha": _a, "pos_tail": True})]
# Round 16 addendum 4: LayerScale damping of the last original blocks (layer sweep: block 21 propagates best on DAVIS dev).
for _bl, _a, _nm in (([23], 0.0, "D23x0"), ([22, 23], 0.5, "D22-23x0.5"), ([22, 23], 0.0, "D22-23x0")):
    VARIANTS[f"PICK6+fisher b0.5+{_nm}"] = TD2 + [("damp_late", {"blocks": _bl, "alpha": _a}),
                                                 ("side_channel", {"A": 4.0, "beta": 0.0, "gamma": 0.5, "fisher_beta": 0.5})]
# Round 16 addendum 5: turn off only block 23's attention or only its MLP.
for _ag, _mg, _nm in ((0.0, 1.0, "D23attn0"), (1.0, 0.0, "D23mlp0")):
    VARIANTS[f"PICK6+fisher b0.5+{_nm}"] = TD2 + [("attn_dominant", {"blocks": [23], "attn_gain": _ag, "mlp_gain": _mg}),
                                                 ("side_channel", {"A": 4.0, "beta": 0.0, "gamma": 0.5, "fisher_beta": 0.5})]
# Round 16 addendum 6: remove the MLPs of the last 2 / 3 original blocks.
for _bl, _nm in (([22, 23], "D22-23mlp0"), ([21, 22, 23], "D21-23mlp0")):
    VARIANTS[f"PICK6+fisher b0.5+{_nm}"] = TD2 + [("attn_dominant", {"blocks": _bl, "attn_gain": 1.0, "mlp_gain": 0.0}),
                                                 ("side_channel", {"A": 4.0, "beta": 0.0, "gamma": 0.5, "fisher_beta": 0.5})]
# Round 17 (SELECTION_RULE_v6.md): smoother look on top of PICK9.
_P9 = lambda **kw: TD2 + [("attn_dominant", {"blocks": [23], "attn_gain": 1.0, "mlp_gain": 0.0}),
                          ("side_channel", {"A": 4.0, "beta": 0.0, "gamma": 0.5, "fisher_beta": 0.5, **kw})]
VARIANTS["P9+mild g0.3"] = _P9(more=[(4.0, 8.0, 0.3)])
VARIANTS["P9+mild g0.5"] = _P9(more=[(4.0, 8.0, 0.5)])
VARIANTS["P9+RG A2b8g1x2"] = _P9(more=[(2.0, 8.0, 1.0), (2.0, 8.0, 1.0)])
VARIANTS["P9 main g1.0"] = _P9(gamma=1.0)
# Round 16 addendum 7: zero the most GLOBAL attention heads of blocks 23 / 22 (global_heads.py, label-free) on PICK9.
_P9H = lambda heads: TD2 + [("attn_dominant", {"blocks": [23], "attn_gain": 1.0, "mlp_gain": 0.0}), ("zero_heads", {"heads": heads}),
                            ("side_channel", {"A": 4.0, "beta": 0.0, "gamma": 0.5, "fisher_beta": 0.5})]
VARIANTS["P9-g23h4"] = _P9H([(23, 12), (23, 8), (23, 3), (23, 1)])
VARIANTS["P9-g23h4+g22h3"] = _P9H([(23, 12), (23, 8), (23, 3), (23, 1), (22, 1), (22, 3), (22, 0)])
# Round 18 (SELECTION_RULE_v7.md): DINOv3-style Gram anchoring as a closed-form linear map after the final norm, on PICK9.
_P9b = TD2 + [("attn_dominant", {"blocks": [23], "attn_gain": 1.0, "mlp_gain": 0.0}),
              ("side_channel", {"A": 4.0, "beta": 0.0, "gamma": 0.5, "fisher_beta": 0.5})]
VARIANTS["P9+gram L21"] = _P9b + [("gram_anchor", {"layer": 21})]
VARIANTS["P9+gram L21 href"] = _P9b + [("gram_anchor", {"layer": 21, "hires": True})]
VARIANTS["P9+gram L21 href a0.5"] = _P9b + [("gram_anchor", {"layer": 21, "hires": True, "alpha": 0.5})]
VARIANTS["P9+gramreg L21"] = _P9b + [("gram_readout", {"layer": 21})]
VARIANTS["P9+gramreg L21 href"] = _P9b + [("gram_readout", {"layer": 21, "hires": True})]
for _a in (0.25, 0.5):
    VARIANTS[f"P9+gramreg L21 a{_a}"] = _P9b + [("gram_readout", {"layer": 21, "alpha": _a})]
