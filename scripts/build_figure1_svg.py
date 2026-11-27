"""Build publication-grade Figure 1 SVG (+ vector PDF & 600-DPI PNG) for
Infrared Physics & Technology (IPT).

Inherits the clean editorial design language, recorded setup photographs,
orthographic 3D push-broom acquisition schematic, and golden kernel icons from
the read-only archived V4/JSPR Figure 1 SVG while showcasing the complete IPT
six-condition perturbation protocol and dual-scale SWIR aggregation framework.
Also embeds a real data-driven mini SWIR reflectance + within-kernel SD plot
directly from Path B+ stage4_optical outputs.

Writes exclusively inside analysis/IPT_submisson/figures/.
"""
from __future__ import annotations

import copy
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parents[1]
V4_SVG = PROJECT / "analysis" / "manuscript_v4" / "figures" / "main" / "Figure_1_experimental_design.svg"
SPECTRA_NPZ = PROJECT / "analysis" / "path_b_plus" / "stage4_optical" / "cultivar_spectra_summary.npz"
FIG_DIR = ROOT / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

NS = "http://www.w3.org/2000/svg"
XLINK = "http://www.w3.org/1999/xlink"
ET.register_namespace("", NS)
ET.register_namespace("xlink", XLINK)

# Refined Elsevier / IPT color system (matching JSPR aesthetic)
C_INK = "#222222"
C_SUB = "#4A4A4A"
C_MUTE = "#666666"
C_HDR = "#F1F1F1"
C_RGB = "#2F6FA3"
C_RGB_BG = "#EEF6FC"
C_SWIR = "#D48216"
C_SWIR_BG = "#FFF6E8"
C_EARLY = "#1F8A65"
C_EARLY_BG = "#EBF8F2"
C_LATE = "#6B4E9B"
C_LATE_BG = "#F4EFFA"
C_FUSE = "#C64D35"
C_FUSE_BG = "#FDF2EF"


def _frag(xml_str: str) -> ET.Element:
    return ET.fromstring(f'<g xmlns="{NS}" xmlns:xlink="{XLINK}">{xml_str}</g>')


def _build_mini_spectrum_svg(x: float, y: float, w: float, h: float) -> str:
    """Create a data-faithful mini SWIR mean reflectance + within-kernel ±1 SD plot."""
    spec = np.load(SPECTRA_NPZ)
    waves = spec["waves"]
    means = spec["mean_spectra"]
    within_sd = spec["within_kernel_sd_spectra"]

    # Subsample smoothly to ~85 points for razor-sharp lightweight vector paths
    idx = np.linspace(0, len(waves) - 1, 85).round().astype(int)
    wv = waves[idx]
    g_mean = means.mean(axis=0)[idx]
    g_sd = within_sd.mean(axis=0)[idx]
    lo = np.clip(g_mean - g_sd, 0.05, 0.72)
    hi = np.clip(g_mean + g_sd, 0.05, 0.72)

    pad_l, pad_r, pad_t, pad_b = 14, 6, 8, 16
    pw, ph = w - pad_l - pad_r, h - pad_t - pad_b
    xmin, xmax = 950.0, 1650.0
    ymin, ymax = 0.08, 0.66

    def sx(val: float) -> float:
        return x + pad_l + (val - xmin) / (xmax - xmin) * pw

    def sy(val: float) -> float:
        return y + pad_t + (1.0 - (val - ymin) / (ymax - ymin)) * ph

    # Envelope polygon
    pts_hi = [f"{sx(w_i):.1f},{sy(v_i):.1f}" for w_i, v_i in zip(wv, hi)]
    pts_lo = [f"{sx(w_i):.1f},{sy(v_i):.1f}" for w_i, v_i in zip(wv[::-1], lo[::-1])]
    poly_pts = " ".join(pts_hi + pts_lo)

    # Selected representative cultivar curves + grand mean
    mean_pts = " ".join(f"{sx(w_i):.1f},{sy(v_i):.1f}" for w_i, v_i in zip(wv, g_mean))
    cv_paths = []
    for ci, col in [(0, "#B86B11"), (4, "#E09A3E"), (10, "#8C4C06")]:
        c_pts = " ".join(f"{sx(w_i):.1f},{sy(v_i):.1f}" for w_i, v_i in zip(wv, means[ci, idx]))
        cv_paths.append(
            f'<polyline points="{c_pts}" fill="none" stroke="{col}" stroke-width="0.95" opacity="0.72"/>'
        )

    ax_x0, ax_x1 = x + pad_l, x + w - pad_r
    ax_y0, ax_y1 = y + pad_t, y + h - pad_b

    return f"""
      <rect x="{x}" y="{y}" width="{w}" height="{h}" rx="3" fill="#FFFDF9" stroke="#EBD8BC" stroke-width="0.9"/>
      <polygon points="{poly_pts}" fill="#E6A94E" opacity="0.28"/>
      {''.join(cv_paths)}
      <polyline points="{mean_pts}" fill="none" stroke="#9C4E07" stroke-width="1.35"/>
      <line x1="{ax_x0}" y1="{ax_y1}" x2="{ax_x1}" y2="{ax_y1}" stroke="#7A7A7A" stroke-width="0.9"/>
      <line x1="{ax_x0}" y1="{ax_y0}" x2="{ax_x0}" y2="{ax_y1}" stroke="#7A7A7A" stroke-width="0.9"/>
      <text x="{ax_x0 + 1}" y="{ax_y1 + 10.5}" font-size="8.2" fill="#555">950</text>
      <text x="{ax_x1}" y="{ax_y1 + 10.5}" font-size="8.2" fill="#555" text-anchor="end">1,650 nm</text>
      <text x="{(ax_x0 + ax_x1)/2:.1f}" y="{ax_y0 + 7.5}" font-size="8.2" fill="#8A4F08" text-anchor="middle" font-weight="600">±1 SD (~9× tray SD)</text>
      <text x="{x + 9}" y="{(ax_y0 + ax_y1)/2:.1f}" font-size="8.0" fill="#666" text-anchor="middle" transform="rotate(-90 {x + 9} {(ax_y0 + ax_y1)/2:.1f})">Refl.</text>
    """


def _perturbation_icon(kind: str, cx: float, cy: float) -> str:
    """Draw a crisp 38x52 micro-icon depicting each of the 6 August imaging conditions."""
    # Mini dark board centered at (cx, cy+10)
    board = (
        f'<polygon points="{cx-15},{cy+18} {cx+15},{cy+18} {cx+18},{cy+8} {cx-12},{cy+8}" '
        f'fill="#3A424D" stroke="#222" stroke-width="0.7"/>'
    )
    k1 = f'<use href="#kernel" transform="translate({cx-4.5} {cy+12.5}) scale(0.38)"/>'
    k2 = f'<use href="#kernel" transform="translate({cx+5.5} {cy+12.5}) scale(0.38)"/>'

    def lamp_left(on: bool) -> str:
        col = "#E8A317" if on else "#CFD5DC"
        if on:
            return (
                f'<line x1="{cx-12}" y1="{cy-8}" x2="{cx-5}" y2="{cy+8}" stroke="#E8A317" stroke-width="0.95" stroke-dasharray="2 1.5" opacity="0.88"/>'
                f'<line x1="{cx-10}" y1="{cy-9}" x2="{cx+1}" y2="{cy+8}" stroke="#E8A317" stroke-width="0.8" stroke-dasharray="2 1.5" opacity="0.62"/>'
                f'<circle cx="{cx-14}" cy="{cy-12}" r="4.3" fill="{col}" stroke="#666" stroke-width="0.7"/>'
            )
        return (
            f'<circle cx="{cx-14}" cy="{cy-12}" r="4.3" fill="{col}" stroke="#7A828C" stroke-width="0.7"/>'
            f'<line x1="{cx-18.5}" y1="{cy-16.5}" x2="{cx-9.5}" y2="{cy-7.5}" stroke="#C64D35" stroke-width="1.6" stroke-linecap="round"/>'
            f'<line x1="{cx-9.5}" y1="{cy-16.5}" x2="{cx-18.5}" y2="{cy-7.5}" stroke="#C64D35" stroke-width="1.6" stroke-linecap="round"/>'
        )

    def lamp_right(on: bool) -> str:
        col = "#E8A317" if on else "#CFD5DC"
        if on:
            return (
                f'<line x1="{cx+12}" y1="{cy-8}" x2="{cx+5}" y2="{cy+8}" stroke="#E8A317" stroke-width="0.95" stroke-dasharray="2 1.5" opacity="0.88"/>'
                f'<line x1="{cx+10}" y1="{cy-9}" x2="{cx-1}" y2="{cy+8}" stroke="#E8A317" stroke-width="0.8" stroke-dasharray="2 1.5" opacity="0.62"/>'
                f'<circle cx="{cx+14}" cy="{cy-12}" r="4.3" fill="{col}" stroke="#666" stroke-width="0.7"/>'
            )
        return (
            f'<circle cx="{cx+14}" cy="{cy-12}" r="4.3" fill="{col}" stroke="#7A828C" stroke-width="0.7"/>'
            f'<line x1="{cx+9.5}" y1="{cy-16.5}" x2="{cx+18.5}" y2="{cy-7.5}" stroke="#C64D35" stroke-width="1.6" stroke-linecap="round"/>'
            f'<line x1="{cx+18.5}" y1="{cy-16.5}" x2="{cx+9.5}" y2="{cy-7.5}" stroke="#C64D35" stroke-width="1.6" stroke-linecap="round"/>'
        )

    if kind == "BASE":
        return board + k1 + k2 + lamp_left(True) + lamp_right(True)
    if kind == "REP":
        # Two circular repeat arcs above mini board
        arcs = (
            f'<path d="M {cx-9} {cy-6} A 9 9 0 0 1 {cx+8} {cy-9}" fill="none" stroke="#2F6FA3" stroke-width="1.5" marker-end="url(#arr-rgb-sm)"/>'
            f'<path d="M {cx+9} {cy-3} A 9 9 0 0 1 {cx-8} {cy}" fill="none" stroke="#2F6FA3" stroke-width="1.5" marker-end="url(#arr-rgb-sm)"/>'
        )
        return board + k1 + k2 + arcs
    if kind == "REPOSITION":
        # Ghost kernel + shifted kernel + horizontal shift arrow
        ghost = f'<use href="#kernel" transform="translate({cx-6.5} {cy-4}) scale(0.62)" opacity="0.32"/>'
        moved = f'<use href="#kernel" transform="translate({cx+7.5} {cy-4}) scale(0.62)"/>'
        arr = f'<line x1="{cx-1.5}" y1="{cy-4}" x2="{cx+2.8}" y2="{cy-4}" stroke="#2F6FA3" stroke-width="1.4" marker-end="url(#arr-rgb-sm)"/>'
        nudge = f'<path d="M {cx-8} {cy+11} H {cx+8}" stroke="#666" stroke-width="0.9" stroke-dasharray="2 1.5"/>'
        return board + nudge + ghost + moved + arr
    if kind == "FLIP":
        # Larger kernel with curved 180° ventral-dorsal flip arrow
        k_mid = f'<use href="#kernel" transform="translate({cx} {cy+1}) scale(0.76)"/>'
        # Subtle anatomical ventral line highlight
        ventral = f'<ellipse cx="{cx+1.5}" cy="{cy+2}" rx="3.2" ry="5.5" fill="#C64D35" opacity="0.28"/>'
        arc_top = f'<path d="M {cx-13} {cy-3} C {cx-11} {cy-18}, {cx+11} {cy-18}, {cx+13} {cy-3}" fill="none" stroke="#C64D35" stroke-width="1.5" marker-end="url(#arr-fuse-sm)"/>'
        arc_bot = f'<path d="M {cx+13} {cy+9} C {cx+11} {cy+20}, {cx-11} {cy+20}, {cx-13} {cy+9}" fill="none" stroke="#C64D35" stroke-width="1.5" marker-end="url(#arr-fuse-sm)"/>'
        lbl = f'<text x="{cx}" y="{cy-15}" font-size="8.0" fill="#C64D35" text-anchor="middle" font-weight="700">180°</text>'
        return k_mid + ventral + arc_top + arc_bot + lbl
    if kind == "LEFT_OFF":
        return board + k1 + k2 + lamp_left(False) + lamp_right(True)
    if kind == "RIGHT_OFF":
        return board + k1 + k2 + lamp_left(True) + lamp_right(False)
    raise ValueError(kind)


def build_figure_1() -> None:
    """Build Figure 1 SVG and render vector PDF + 600-DPI PNG via Playwright/Edge."""
    v4_root = ET.parse(V4_SVG).getroot()

    # Extract reusable kernel definition and setup elements [12..25] read-only
    kernel_def = None
    for el in v4_root.iter(f"{{{NS}}}g"):
        if el.attrib.get("id") == "kernel":
            kernel_def = copy.deepcopy(el)
            break
    assert kernel_def is not None, "Could not find #kernel definition in v4 SVG"

    # Also extract arrow-gray marker needed by v4 push-broom arrow inside setup panel
    v4_defs = v4_root.find(f"{{{NS}}}defs")
    arrow_gray_v4 = None
    if v4_defs is not None:
        for child in v4_defs:
            if child.attrib.get("id") == "arrow-gray":
                arrow_gray_v4 = copy.deepcopy(child)
                break

    # Canvas: 180 mm x 145.5 mm (960 x 776 user units, matching Elsevier 2-col width)
    W, H = 960, 776
    svg = ET.Element(
        f"{{{NS}}}svg",
        {
            "width": "180mm",
            "height": "145.5mm",
            "viewBox": f"0 0 {W} {H}",
            "role": "img",
        },
    )

    svg.append(_frag(f"""
      <title>Figure 1. Paired RGB–SWIR acquisition setup and dual-scale SWIR optical aggregation framework</title>
      <style>text {{ font-family: Arial, Helvetica, sans-serif; }}</style>
    """))

    # Definitions: kernel icon + crisp color-coded arrow markers
    defs = ET.SubElement(svg, f"{{{NS}}}defs")
    defs.append(kernel_def)
    if arrow_gray_v4 is not None:
        defs.append(arrow_gray_v4)

    marker_specs = [
        ("arr-gray", C_MUTE, 6, 5),
        ("arr-rgb", C_RGB, 6, 5),
        ("arr-rgb-sm", C_RGB, 4.5, 4),
        ("arr-swir", C_SWIR, 6, 5),
        ("arr-early", C_EARLY, 6, 5),
        ("arr-late", C_LATE, 6, 5),
        ("arr-fuse", C_FUSE, 6, 5),
        ("arr-fuse-sm", C_FUSE, 4.5, 4),
    ]
    for mid, col, mw, mh in marker_specs:
        m = ET.SubElement(
            defs,
            f"{{{NS}}}marker",
            {
                "id": mid,
                "viewBox": "0 0 10 8",
                "refX": "8.8",
                "refY": "4",
                "markerWidth": str(mw),
                "markerHeight": str(mh),
                "orient": "auto",
            },
        )
        ET.SubElement(m, f"{{{NS}}}path", {"d": "M 0 0.8 L 9.2 4 L 0 7.2 z", "fill": col})

    # Clean white background
    svg.append(ET.Element(f"{{{NS}}}rect", {"width": str(W), "height": str(H), "fill": "white"}))

    # =========================================================================
    # PANEL A: Paired RGB-SWIR acquisition apparatus & 6-condition perturbation
    # =========================================================================
    svg.append(_frag(f"""
      <rect x="18" y="14" width="924" height="28" fill="{C_HDR}"/>
      <text x="27" y="35" font-size="20" font-weight="700" fill="{C_INK}">A</text>
      <text x="55" y="34" font-size="15.5" font-weight="600" fill="{C_INK}">Paired RGB–SWIR acquisition apparatus, coregistration, and controlled perturbation protocol</text>

      <!-- Left subheader -->
      <text x="26" y="63" font-size="13.5" font-weight="700" fill="{C_INK}">Acquisition setup and push-broom imaging geometry</text>
    """))

    # Nested setup SVG reusing v4 recorded photos + 3D orthographic push-broom schematic (elements 12..25)
    setup_nested = ET.SubElement(
        svg,
        f"{{{NS}}}svg",
        {
            "x": "18",
            "y": "73",
            "width": "454",
            "height": "170",
            "viewBox": "22 72 454 170",
            "overflow": "hidden",
        },
    )
    for idx in range(12, 26):
        setup_nested.append(copy.deepcopy(v4_root[idx]))

    # Left bottom cohort summary pill + vertical divider
    svg.append(_frag(f"""
      <rect x="20" y="247" width="454" height="25" rx="3" fill="#F8F9FA" stroke="#D5DCE2" stroke-width="1"/>
      <text x="247" y="263" font-size="9.4" fill="#2D3748" text-anchor="middle">
        <tspan font-weight="700">Primary cohort:</tspan> 3,627 kernels · 67 trays · 12 cultivars · 951.5–1,650.0 nm (421 bands) · mask_inner
      </text>

      <line x1="486" y1="50" x2="486" y2="275" stroke="#D2D6DC" stroke-width="1.2" stroke-dasharray="4 4"/>

      <!-- Right subheader -->
      <text x="502" y="63" font-size="13.5" font-weight="700" fill="{C_INK}">Later-date paired perturbation cohort</text>
      <text x="502" y="80" font-size="11.0" fill="{C_SUB}">280 physical kernels (10 entity trays) × 6 paired conditions = 1,680 observations</text>
    """))

    # 3 x 2 Perturbation Condition Visual Cards
    cond_specs = [
        ("BASE", "Standard 2-lamp", "baseline", "#CBD2D9", "#FFFFFF", C_INK),
        ("REP", "Immediate", "technical repeat", "#CBD2D9", "#FFFFFF", C_INK),
        ("REPOSITION", "In-place kernel", "repositioning", "#CBD2D9", "#FFFFFF", C_INK),
        ("FLIP", "Ventral–dorsal", "kernel flip", "#DF9B8C", "#FFFDFC", "#942B18"),
        ("LEFT_OFF", "Left halogen", "lamp off", "#E4BE80", "#FFFDF8", "#8A4F08"),
        ("RIGHT_OFF", "Right halogen", "lamp off", "#E4BE80", "#FFFDF8", "#8A4F08"),
    ]
    cards_xml = []
    for i, (code, l1, l2, border_col, bg_col, title_col) in enumerate(cond_specs):
        col_i = i % 3
        row_i = i // 3
        cx0 = 502 + col_i * 148
        cy0 = 90 + row_i * 75
        cw, ch = 140, 68
        icon_svg = _perturbation_icon(code, cx0 + 24, cy0 + 34)
        cards_xml.append(f"""
          <rect x="{cx0}" y="{cy0}" width="{cw}" height="{ch}" rx="4" fill="{bg_col}" stroke="{border_col}" stroke-width="1.2"/>
          {icon_svg}
          <line x1="{cx0 + 47}" y1="{cy0 + 10}" x2="{cx0 + 47}" y2="{cy0 + ch - 10}" stroke="#ECEFF2" stroke-width="1"/>
          <text x="{cx0 + 54}" y="{cy0 + 24}" font-size="11.0" font-weight="700" fill="{title_col}">{code}</text>
          <text x="{cx0 + 54}" y="{cy0 + 40}" font-size="10.0" fill="{C_SUB}">{l1}</text>
          <text x="{cx0 + 54}" y="{cy0 + 53}" font-size="10.0" fill="{C_SUB}">{l2}</text>
        """)
    svg.append(_frag("".join(cards_xml)))

    svg.append(_frag(f"""
      <rect x="502" y="247" width="438" height="25" rx="3" fill="{C_SWIR_BG}" stroke="#E5C087" stroke-width="1"/>
      <text x="721" y="263" font-size="9.5" fill="#784408" text-anchor="middle">
        Fixed pre-session dual-lamp white reference shared across all 6 August conditions
      </text>
    """))

    # =========================================================================
    # PANEL B: Dual-scale SWIR optical aggregation & multimodal evaluation
    # =========================================================================
    svg.append(_frag(f"""
      <rect x="18" y="286" width="924" height="28" fill="{C_HDR}"/>
      <text x="27" y="307" font-size="20" font-weight="700" fill="{C_INK}">B</text>
      <text x="55" y="306" font-size="15.5" font-weight="600" fill="{C_INK}">Dual-scale SWIR optical information-aggregation and multimodal evaluation workflow</text>

      <!-- Column Subheaders -->
      <text x="20" y="335" font-size="13.0" font-weight="700" fill="{C_INK}">Scale 1: Intra-kernel summary</text>
      <text x="302" y="335" font-size="13.0" font-weight="700" fill="{C_INK}">Scale 2: Nested k subsets</text>
      <text x="518" y="335" font-size="13.0" font-weight="700" fill="{C_INK}">Inter-kernel routes &amp; diagnostics</text>
      <text x="798" y="335" font-size="13.0" font-weight="700" fill="{C_INK}">Sample-level output</text>
    """))

    # -------------------------------------------------------------------------
    # ZONE 1: Scale 1 SWIR (with zoomed mask_inner + real mini spectrum) & RGB
    # -------------------------------------------------------------------------
    mini_spec_svg = _build_mini_spectrum_svg(132, 376, 141, 74)
    # Pixel dots inside zoomed kernel mask_inner
    px_dots = []
    for dx in (-10, -4, 2, 8):
        for dy in (-10, -4, 2, 8):
            if dx * dx + dy * dy <= 115:
                px_dots.append(f'<rect x="{70 + dx - 1.5}" y="{409 + dy - 1.5}" width="3" height="3" fill="#B86B11" opacity="0.65"/>')

    svg.append(_frag(f"""
      <!-- SWIR Scale 1 Card -->
      <rect x="20" y="344" width="262" height="206" rx="4" fill="white" stroke="{C_SWIR}" stroke-width="1.6"/>
      <path d="M 24 344 H 278 Q 282 344 282 348 V 368 H 20 V 348 Q 20 344 24 344 Z" fill="{C_SWIR_BG}"/>
      <line x1="20" y1="368" x2="282" y2="368" stroke="#EDD3A8" stroke-width="1"/>
      <text x="30" y="360" font-size="11.3" font-weight="700" fill="#8A4F08">SWIR inner-mask pixels (mask_inner)</text>

      <!-- Zoomed eroded mask micro-diagram -->
      <ellipse cx="70" cy="410" rx="26" ry="21" fill="#FBF2DE" stroke="#8C96A0" stroke-width="1.0" stroke-dasharray="3 2"/>
      <ellipse cx="70" cy="410" rx="18" ry="14" fill="#F6DFB0" stroke="{C_SWIR}" stroke-width="1.5"/>
      {''.join(px_dots)}
      <text x="70" y="441" font-size="8.7" fill="#555" text-anchor="middle">2× eroded mask_inner</text>
      <text x="70" y="451" font-size="8.5" fill="#777" text-anchor="middle">(~350–579 px/kernel)</text>
      <line x1="101" y1="410" x2="127" y2="410" stroke="{C_SWIR}" stroke-width="1.5" marker-end="url(#arr-swir)"/>

      <!-- Real data-driven mini SWIR spectrum -->
      {mini_spec_svg}

      <line x1="28" y1="460" x2="274" y2="460" stroke="#EFE2CC" stroke-width="1"/>
      <text x="30" y="476" font-size="9.9" fill="{C_INK}"><tspan font-weight="700">Summaries:</tspan> median, mean, mean+SD, distribution</text>
      <text x="30" y="494" font-size="9.9" fill="{C_INK}"><tspan font-weight="700">Ablation:</tspan> SNV vs. non-SNV (421 SWIR bands)</text>
      <text x="30" y="512" font-size="9.9" fill="{C_INK}"><tspan font-weight="700">Perturbation optics:</tspan> per-kernel OLS affine fit,</text>
      <text x="30" y="526" font-size="9.8" fill="{C_SUB}">421-band non-affine residuals &amp; SAM (°)</text>
      <text x="30" y="542" font-size="9.5" fill="#8A4F08" font-style="italic">Homography-mapped from verified RGB masks</text>

      <!-- Paired RGB Comparators Card -->
      <rect x="20" y="564" width="262" height="132" rx="4" fill="white" stroke="{C_RGB}" stroke-width="1.6"/>
      <path d="M 24 564 H 278 Q 282 564 282 568 V 588 H 20 V 568 Q 20 564 24 564 Z" fill="{C_RGB_BG}"/>
      <line x1="20" y1="588" x2="282" y2="588" stroke="#C5DCEB" stroke-width="1"/>
      <text x="30" y="580" font-size="11.3" font-weight="700" fill="#1D4E78">Paired RGB branches &amp; visible comparators</text>

      <!-- Clean JSPR-style RGB node fork -->
      <rect x="29" y="611" width="58" height="42" rx="3" fill="white" stroke="{C_RGB}" stroke-width="1.3"/>
      <text x="58" y="629" font-size="11.0" font-weight="700" fill="{C_INK}" text-anchor="middle">RGB</text>
      <text x="58" y="644" font-size="9.6" fill="{C_SUB}" text-anchor="middle">paired</text>
      <polyline points="87,632 98,632 98,611 109,611" fill="none" stroke="{C_RGB}" stroke-width="1.4" marker-end="url(#arr-rgb-sm)"/>
      <polyline points="98,632 98,651 109,651" fill="none" stroke="{C_RGB}" stroke-width="1.4" marker-end="url(#arr-rgb-sm)"/>

      <rect x="112" y="595" width="161" height="31" rx="3" fill="white" stroke="{C_RGB}" stroke-width="1.2"/>
      <text x="119" y="608" font-size="10.2" fill="{C_INK}"><tspan font-weight="700">M:</tspan> 28 morph. + color features</text>
      <text x="119" y="621" font-size="9.5" fill="{C_SUB}">M_jpeg vs. uniform M_raw (CR3)</text>

      <rect x="112" y="634" width="161" height="31" rx="3" fill="white" stroke="{C_RGB}" stroke-width="1.2"/>
      <text x="119" y="647" font-size="10.2" fill="{C_INK}"><tspan font-weight="700">D:</tspan> fixed 17×17 mm JPEG</text>
      <text x="119" y="660" font-size="9.5" fill="{C_SUB}">EfficientNet-B0 (held fixed)</text>

      <text x="151" y="684" font-size="10.2" font-weight="600" fill="#1D4E78" text-anchor="middle">Visible baseline: MD_raw = (M_raw + D) / 2</text>
    """))

    # -------------------------------------------------------------------------
    # ZONE 2: JSPR-inspired Nested k-kernel subsets + Validation guardrails
    # -------------------------------------------------------------------------
    nested_rows = []
    # Pool + k = 1, 3, 5, 10 with 10 golden kernel icons per row
    x_k0 = 357.0
    dx_k = 12.6
    row_defs = [
        ("Pool", 386, 10, False),
        ("k = 1", 412, 1, True),
        ("k = 3", 438, 3, True),
        ("k = 5", 464, 5, True),
        ("k = 10", 490, 10, True),
    ]
    for lbl, ry, n_sel, draw_box in row_defs:
        nested_rows.append(
            f'<text x="345" y="{ry + 3.5}" font-size="10.2" font-weight="{"700" if lbl=="Pool" else "600"}" fill="{C_INK}" text-anchor="end">{lbl}</text>'
        )
        if draw_box:
            bx = x_k0 - 6.8
            bw = (n_sel - 1) * dx_k + 13.6
            nested_rows.append(
                f'<rect x="{bx:.1f}" y="{ry - 9.5}" width="{bw:.1f}" height="19" rx="2.5" fill="{C_RGB_BG}" stroke="{C_RGB}" stroke-width="1.2"/>'
            )
        for j in range(10):
            kx = x_k0 + j * dx_k
            op = "1.0" if j < n_sel else "0.20"
            nested_rows.append(
                f'<use href="#kernel" transform="translate({kx:.1f} {ry}) scale(0.50)" opacity="{op}"/>'
            )

    svg.append(_frag(f"""
      <!-- Nested k-kernel subsets card -->
      <rect x="302" y="344" width="188" height="224" rx="4" fill="white" stroke="#B8C2CC" stroke-width="1.25"/>
      <text x="396" y="363" font-size="10.4" font-weight="600" fill="{C_SUB}" text-anchor="middle">500 shared permutations / tray</text>
      <line x1="312" y1="371" x2="480" y2="371" stroke="#ECEFF2" stroke-width="1"/>
      {''.join(nested_rows)}

      <!-- k = 20 / All nested badge -->
      <rect x="313" y="508" width="166" height="22" rx="3" fill="{C_RGB_BG}" stroke="{C_RGB}" stroke-width="1.2"/>
      <text x="396" y="522.5" font-size="10.2" font-weight="700" fill="#1D4E78" text-anchor="middle">k = 20  ⊂  All (20–72 kernels)</text>

      <text x="396" y="546" font-size="9.8" fill="{C_SUB}" text-anchor="middle">Nested prefixes: identical physical</text>
      <text x="396" y="559" font-size="9.8" fill="{C_SUB}" text-anchor="middle">subsets across all routes &amp; models</text>

      <!-- Validation guardrails card -->
      <rect x="302" y="582" width="188" height="114" rx="4" fill="#FAFAFB" stroke="#8C96A0" stroke-width="1.2"/>
      <text x="313" y="601" font-size="11.0" font-weight="700" fill="{C_INK}">Evaluation guardrails</text>
      <line x1="312" y1="608" x2="480" y2="608" stroke="#E2E6EA" stroke-width="1"/>
      <text x="313" y="625" font-size="10.0" fill="{C_SUB}">• 4 outer × 3 inner tray folds</text>
      <text x="313" y="643" font-size="10.0" fill="{C_SUB}">• Zero tray or kernel leakage</text>
      <text x="313" y="661" font-size="10.0" fill="{C_SUB}">• Unified tray-equal weighting</text>
      <text x="313" y="679" font-size="10.0" fill="{C_SUB}">• 2,000 paired bootstrap draws</text>
    """))

    # -------------------------------------------------------------------------
    # ZONE 3: Route A (Early Spectral Pooling) & Route B (Late Probability Pooling)
    # -------------------------------------------------------------------------
    svg.append(_frag(f"""
      <!-- Route A Card -->
      <rect x="518" y="344" width="256" height="164" rx="4" fill="white" stroke="{C_EARLY}" stroke-width="1.6"/>
      <path d="M 522 344 H 770 Q 774 344 774 348 V 368 H 518 V 348 Q 518 344 522 344 Z" fill="{C_EARLY_BG}"/>
      <line x1="518" y1="368" x2="774" y2="368" stroke="#BCE4D3" stroke-width="1"/>
      <text x="528" y="360" font-size="11.3" font-weight="700" fill="#115E43">Route A: Spectral-level early pooling</text>

      <!-- Operator order equation box -->
      <rect x="528" y="376" width="236" height="58" rx="3" fill="#F7FCFA" stroke="#C5E8D9" stroke-width="1"/>
      <text x="536" y="392" font-size="10.2" fill="{C_INK}"><tspan font-weight="700" fill="#115E43">early1:</tspan>  SNV( mean<tspan baseline-shift="sub" font-size="7.8">k</tspan>(x<tspan baseline-shift="sub" font-size="7.8">i</tspan>) ) → f(·)</text>
      <text x="536" y="409" font-size="10.2" fill="{C_INK}"><tspan font-weight="700" fill="#115E43">early2:</tspan>  mean<tspan baseline-shift="sub" font-size="7.8">k</tspan>( SNV(x<tspan baseline-shift="sub" font-size="7.8">i</tspan>) ) → f(·)</text>
      <text x="536" y="426" font-size="10.2" fill="{C_INK}"><tspan font-weight="700" fill="#115E43">early3:</tspan>  SNV( mean<tspan baseline-shift="sub" font-size="7.8">k</tspan>(SNV(x<tspan baseline-shift="sub" font-size="7.8">i</tspan>)) ) → f(·)</text>

      <text x="528" y="451" font-size="9.9" fill="{C_INK}"><tspan font-weight="700">Protocol 1:</tspan> fixed single-kernel f (PCA shift)</text>
      <text x="528" y="468" font-size="9.9" fill="{C_INK}"><tspan font-weight="700">Protocol 2:</tspan> scale-matched multi-kernel training</text>
      <text x="538" y="483" font-size="9.6" fill="{C_SUB}">k20→All vs. All_exact; PCA, 1/M wt &amp; C</text>
      <text x="538" y="497" font-size="9.6" fill="{C_SUB}">controls + algebraic s<tspan baseline-shift="sub" font-size="7.4">i</tspan>-weighting check</text>

      <!-- Route B Card -->
      <rect x="518" y="520" width="256" height="176" rx="4" fill="white" stroke="{C_LATE}" stroke-width="1.6"/>
      <path d="M 522 520 H 770 Q 774 520 774 524 V 544 H 518 V 524 Q 518 520 522 520 Z" fill="{C_LATE_BG}"/>
      <line x1="518" y1="544" x2="774" y2="544" stroke="#D5C6EC" stroke-width="1"/>
      <text x="528" y="536" font-size="11.3" font-weight="700" fill="#4A2F78">Route B: Probability-level late pooling</text>

      <text x="528" y="559" font-size="10.0" fill="{C_INK}"><tspan font-weight="700">14 SWIR configs</tspan> (4 families: Logistic,</text>
      <text x="528" y="572" font-size="10.0" fill="{C_SUB}">RBF-SVM, ExtraTrees, 1D-CNN)</text>

      <rect x="528" y="580" width="236" height="51" rx="3" fill="#FAF8FD" stroke="#DDD1F2" stroke-width="1"/>
      <text x="536" y="594" font-size="10.0" font-weight="700" fill="#4A2F78">7 pooling &amp; nested calibration rules</text>
      <text x="536" y="608" font-size="9.6" fill="{C_SUB}">arith_raw, majority_vote, log_prob_pool,</text>
      <text x="536" y="622" font-size="9.6" fill="{C_SUB}">pre-pool T*<tspan baseline-shift="sub" font-size="7.2">kernel</tspan>/T*<tspan baseline-shift="sub" font-size="7.2">group</tspan>, post-pool T<tspan baseline-shift="sub" font-size="7.2">post</tspan>(k)</text>

      <rect x="528" y="638" width="236" height="50" rx="3" fill="#FAF8FD" stroke="#DDD1F2" stroke-width="1"/>
      <text x="536" y="652" font-size="10.0" font-weight="700" fill="#4A2F78">Zero-retraining margin diagnostics</text>
      <text x="536" y="666" font-size="9.6" fill="{C_SUB}">Worst-rival margin μ* = μ*<tspan baseline-shift="sub" font-size="7.2">corr</tspan> + μ*<tspan baseline-shift="sub" font-size="7.2">mis</tspan></text>
      <text x="536" y="680" font-size="9.6" fill="{C_SUB}">Rival-slot permutation &amp; sharpness sweep</text>
    """))

    # -------------------------------------------------------------------------
    # ZONE 4: Sample-level decisions & multimodal fusion
    # -------------------------------------------------------------------------
    svg.append(_frag(f"""
      <rect x="798" y="344" width="144" height="352" rx="4" fill="white" stroke="{C_FUSE}" stroke-width="1.7"/>
      <path d="M 802 344 H 938 Q 942 344 942 348 V 382 H 798 V 348 Q 798 344 802 344 Z" fill="{C_FUSE_BG}"/>
      <line x1="798" y1="382" x2="942" y2="382" stroke="#E9BFB5" stroke-width="1"/>
      <text x="870" y="361" font-size="11.4" font-weight="700" fill="#942B18" text-anchor="middle">Sample-level</text>
      <text x="870" y="375" font-size="11.4" font-weight="700" fill="#942B18" text-anchor="middle">decisions &amp; fusion</text>

      <!-- Block 1: SWIR-only endpoints -->
      <text x="808" y="401" font-size="10.4" font-weight="700" fill="{C_INK}">SWIR-only (k = 1..All)</text>
      <text x="808" y="418" font-size="9.9" fill="{C_SUB}">• Cross-scale rank ρ(k)</text>
      <text x="808" y="434" font-size="9.9" fill="{C_SUB}">• Group BA &amp; tray counts</text>
      <text x="808" y="450" font-size="9.9" fill="{C_SUB}">• Group NLL, Brier, ECE</text>
      <text x="808" y="466" font-size="9.9" fill="{C_SUB}">• Variance vs. bias error</text>

      <line x1="808" y1="480" x2="932" y2="480" stroke="#E8CFC9" stroke-width="1" stroke-dasharray="3 2"/>

      <!-- Block 2: Multimodal fusion -->
      <text x="808" y="499" font-size="10.4" font-weight="700" fill="#942B18">Multimodal fusion</text>
      <rect x="808" y="507" width="124" height="36" rx="3" fill="{C_FUSE_BG}" stroke="#E3AEA2" stroke-width="1.1"/>
      <text x="870" y="522" font-size="10.2" font-weight="700" fill="#942B18" text-anchor="middle">MSD_raw</text>
      <text x="870" y="536" font-size="9.8" fill="{C_INK}" text-anchor="middle">= (M_raw + D + S) / 3</text>
      <text x="808" y="559" font-size="9.8" fill="{C_SUB}">• Additive SWIR shift Δd<tspan baseline-shift="sub" font-size="7.3">c</tspan></text>
      <text x="808" y="574" font-size="9.8" fill="{C_SUB}">• Rescued vs. harmed</text>

      <line x1="808" y1="588" x2="932" y2="588" stroke="#E8CFC9" stroke-width="1" stroke-dasharray="3 2"/>

      <!-- Block 3: Operational regimes -->
      <text x="808" y="607" font-size="10.4" font-weight="700" fill="{C_INK}">Operational regimes</text>
      <text x="808" y="624" font-size="9.9" fill="{C_SUB}">• Primary (67 trays)</text>
      <text x="808" y="640" font-size="9.9" fill="{C_SUB}">• 6-cond. August shift</text>
      <text x="808" y="656" font-size="9.9" fill="{C_SUB}">• SNV lamp-loss recovery</text>
      <text x="808" y="672" font-size="9.9" fill="{C_SUB}">• Small-k SWIR gain</text>
    """))

    # -------------------------------------------------------------------------
    # CLEAN ORTHOGONAL FLOW ARROWS (ZERO CROSSINGS)
    # -------------------------------------------------------------------------
    svg.append(_frag(f"""
      <!-- Scale 1 SWIR -> Scale 2 Nested k -->
      <line x1="282" y1="447" x2="300" y2="447" stroke="{C_SWIR}" stroke-width="1.8" marker-end="url(#arr-swir)"/>

      <!-- Scale 2 Nested k neutral trunk -> Route A (emerald) & Route B (violet) -->
      <line x1="490" y1="456" x2="504" y2="456" stroke="#555555" stroke-width="1.8"/>
      <circle cx="504" cy="456" r="2.1" fill="#555555"/>
      <polyline points="504,456 504,426 516,426" fill="none" stroke="{C_EARLY}" stroke-width="1.8" stroke-linejoin="round" marker-end="url(#arr-early)"/>
      <polyline points="504,456 504,608 516,608" fill="none" stroke="{C_LATE}" stroke-width="1.8" stroke-linejoin="round" marker-end="url(#arr-late)"/>

      <!-- Route A -> Output -->
      <line x1="774" y1="426" x2="796" y2="426" stroke="{C_EARLY}" stroke-width="1.8" marker-end="url(#arr-early)"/>

      <!-- Route B -> Output -->
      <line x1="774" y1="608" x2="796" y2="608" stroke="{C_LATE}" stroke-width="1.8" marker-end="url(#arr-late)"/>

      <!-- Paired RGB -> bottom channel -> Multimodal Output (zero crossings!) -->
      <polyline points="151,696 151,719 870,719 870,698" fill="none" stroke="{C_RGB}" stroke-width="1.8" stroke-linejoin="round" marker-end="url(#arr-rgb)"/>
      <rect x="336" y="710" width="356" height="17" rx="3" fill="white"/>
      <text x="514" y="722" font-size="9.7" fill="{C_RGB}" text-anchor="middle" font-weight="600">Paired visible comparators (M_raw, D, MD_raw) → multimodal fusion</text>
    """))

    # =========================================================================
    # FOOTER LEGEND STRIP (JSPR signature clean legend)
    # =========================================================================
    svg.append(_frag(f"""
      <line x1="18" y1="738" x2="942" y2="738" stroke="#E2E6EA" stroke-width="1"/>

      <use href="#kernel" transform="translate(30 756) scale(0.60)"/>
      <text x="43" y="760" font-size="11.2" fill="{C_INK}">Physical kernel</text>

      <line x1="142" y1="756" x2="164" y2="756" stroke="{C_RGB}" stroke-width="2.2"/>
      <text x="170" y="760" font-size="11.2" fill="{C_INK}">RGB branch</text>

      <line x1="256" y1="756" x2="278" y2="756" stroke="{C_SWIR}" stroke-width="2.2"/>
      <text x="284" y="760" font-size="11.2" fill="{C_INK}">SWIR Scale 1</text>

      <line x1="378" y1="756" x2="400" y2="756" stroke="{C_EARLY}" stroke-width="2.2"/>
      <text x="406" y="760" font-size="11.2" fill="{C_INK}">Early pooling</text>

      <line x1="496" y1="756" x2="518" y2="756" stroke="{C_LATE}" stroke-width="2.2"/>
      <text x="524" y="760" font-size="11.2" fill="{C_INK}">Late pooling</text>

      <line x1="610" y1="756" x2="632" y2="756" stroke="{C_FUSE}" stroke-width="2.2"/>
      <text x="638" y="760" font-size="11.2" fill="{C_INK}">Multimodal output</text>

      <text x="940" y="760" font-size="11.2" fill="{C_MUTE}" font-style="italic" text-anchor="end">Schematics not to scale</text>
    """))

    out_svg = FIG_DIR / "Figure_1_experimental_framework.svg"
    ET.ElementTree(svg).write(out_svg, encoding="utf-8", xml_declaration=True)
    subprocess.run(
        ["node", str(ROOT / "export_figure1.cjs"), str(out_svg)],
        check=True,
    )


if __name__ == "__main__":
    build_figure_1()
