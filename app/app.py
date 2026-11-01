"""
Electrode Voltage Predictor -- interactive web app (Gradio)
============================================================
Predict, compare and screen battery-electrode host materials for Li, Na, K, Mg, Ca and Zn
insertion from the chemical formula alone.

Tabs
  1. Predict            -- one host + one ion, with the material placed on the voltage-capacity map
  2. Compare ions       -- the same host with every working ion
  3. Insertion explorer -- how average voltage, capacity and energy change with the amount inserted
  4. Batch screening    -- many formulas at once, sortable table and CSV download
  5. Explore the data   -- the 2,880 Materials Project electrodes behind the model
  6. About              -- method, accuracy and limitations

Run locally:   pip install -r requirements.txt  &&  python app.py
"""
from __future__ import annotations

import os
import tempfile

import gradio as gr
import numpy as np
import pandas as pd
import plotly.graph_objects as go

import core
from core import ALL_IONS, E0, META, REF, TRAINED_IONS, InputError

# ---------------------------------------------------------------------------
# Look and feel
# ---------------------------------------------------------------------------
ION_COLORS = {"Li": "#1f77b4", "Na": "#ff7f0e", "K": "#2ca02c", "Mg": "#d62728", "Ca": "#9467bd",
              "Zn": "#8c564b", "Rb": "#e377c2", "Cs": "#7f7f7f", "Al": "#bcbd22"}
ACCENT = "#1F6F8B"
INK = "#1B2A41"

CSS = """
.result-card {border-radius: 14px; padding: 18px 22px; background: linear-gradient(135deg,#1B2A41 0%,#1F6F8B 100%);
              color: #ffffff !important;}
.result-card div, .result-card span, .result-card sup {color: #ffffff !important;}
.result-card .badge.trained {color:#065f46 !important;} .result-card .badge.zero-shot {color:#92400e !important;}
.result-card .badge.experimental {color:#991b1b !important;}
.result-card .v {font-size: 46px; font-weight: 700; line-height: 1.05;}
.result-card .sub {opacity: .85; font-size: 15px; margin-top: 4px;}
.result-card .grid {display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px; margin-top: 14px;}
.result-card .cell {background: rgba(255,255,255,.12); border-radius: 10px; padding: 8px 10px;}
.result-card .cell .k {font-size: 12px; opacity: .8; text-transform: uppercase; letter-spacing: .04em;}
.result-card .cell .val {font-size: 20px; font-weight: 600;}
.badge {display:inline-block; border-radius: 999px; padding: 2px 10px; font-size: 12px; font-weight: 600; margin-left: 8px;}
.badge.trained {background:#d1fae5; color:#065f46;}
.badge.zero-shot {background:#fef3c7; color:#92400e;}
.badge.experimental {background:#fee2e2; color:#991b1b;}
.note-box {border-left: 4px solid #1F6F8B; background: rgba(31,111,139,.07); padding: 10px 14px; border-radius: 6px;
           margin-top: 10px; font-size: 14px;}
.err-box {border-left: 4px solid #C8553D; background: rgba(200,85,61,.08); padding: 12px 14px; border-radius: 6px;}
footer {visibility: hidden;}
"""

THEME = gr.themes.Soft(primary_hue="cyan", secondary_hue="slate", neutral_hue="slate",
                       font=[gr.themes.GoogleFont("Inter"), "ui-sans-serif", "sans-serif"])

ION_CHOICES = ([f"{i} (trained)" for i in TRAINED_IONS]
               + ["Rb (zero-shot)", "Cs (zero-shot)", "Al (experimental)"])
EXAMPLE_HOSTS = ["FePO4", "CoO2", "MnO2", "V2O5", "Fe2(SO4)3", "NiO2", "FeF3", "TiS2", "MoO3", "VOPO4"]


def _layout(fig: go.Figure, title: str, height: int = 430) -> go.Figure:
    fig.update_layout(title=dict(text=title, x=0.02, font=dict(size=16, color=INK)), height=height,
                      template="plotly_white", margin=dict(l=60, r=30, t=60, b=55),
                      legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="right", x=1))
    return fig


def _error_html(msg: str) -> str:
    return f"<div class='err-box'>⚠️ {msg}</div>"


def _empty_fig(msg: str = "") -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(text=msg, showarrow=False, font=dict(size=14, color="#888"))
    fig.update_xaxes(visible=False); fig.update_yaxes(visible=False)
    return _layout(fig, "", 300)


# ---------------------------------------------------------------------------
# Tab 1: single prediction
# ---------------------------------------------------------------------------
def _result_card(p: core.Prediction) -> str:
    err = f"± {p.typical_error:.2f} V typical error" if p.typical_error else "rough estimate only"
    ox = ("—" if np.isnan(p.ox_charged) or np.isnan(p.ox_discharged)
          else f"{p.ox_charged:+.2f} → {p.ox_discharged:+.2f}")
    q = core.ION_CHARGE[p.ion]
    return f"""
<div class='result-card'>
  <div class='sub'>{p.host} + {p.x:g} {p.ion} → {p.discharged} &nbsp;(vs {p.ion}/{p.ion}<sup>{'' if q == 1 else q}+</sup>)
    <span class='badge {p.label}'>{p.label}</span></div>
  <div class='v'>{p.voltage:.2f} V</div>
  <div class='sub'>{err}</div>
  <div class='grid'>
    <div class='cell'><div class='k'>Capacity</div><div class='val'>{p.capacity:.0f} mAh/g</div></div>
    <div class='cell'><div class='k'>Specific energy</div><div class='val'>{p.energy:.0f} Wh/kg</div></div>
    <div class='cell'><div class='k'>Redox-metal ox. state</div><div class='val'>{ox}</div></div>
  </div>
</div>"""


def _notes_html(p: core.Prediction) -> str:
    parts = []
    m = core.reference_matches(p.host, p.ion)
    if len(m):
        vals = ", ".join(f"{v:.2f} V" for v in m["average_voltage"].head(3))
        parts.append(f"📚 <b>In the Materials Project data</b> for {p.ion}: DFT average voltage {vals} "
                     f"(this host may have been in the training set).")
    parts += [f"• {n}" for n in p.notes]
    if not parts:
        return ""
    return "<div class='note-box'>" + "<br>".join(parts) + "</div>"


def map_figure(p: core.Prediction | None, ion: str) -> go.Figure:
    fig = go.Figure()
    # iso-energy curves
    caps = np.linspace(20, 600, 200)
    for e in (200, 400, 600, 800, 1000):
        fig.add_trace(go.Scatter(x=caps, y=e / caps, mode="lines", line=dict(color="#c9d3da", dash="dot", width=1),
                                 hoverinfo="skip", showlegend=False))
        fig.add_annotation(x=min(590, e / 5.6), y=e / min(590, e / 5.6), text=f"{e} Wh/kg", showarrow=False,
                           font=dict(size=10, color="#9aa7b0"), xanchor="left", yanchor="bottom")
    if REF is not None:
        d = REF[REF["working_ion"] == ion]
        if len(d):
            fig.add_trace(go.Scatter(
                x=d["capacity_mAh_g"], y=d["average_voltage"], mode="markers",
                marker=dict(size=6, color=ION_COLORS.get(ion, ACCENT), opacity=0.35),
                name=f"Materials Project ({ion}, {len(d)})",
                customdata=np.stack([d["framework_formula"], d["formula_discharge"]], axis=1),
                hovertemplate="<b>%{customdata[0]}</b> → %{customdata[1]}<br>%{y:.2f} V, %{x:.0f} mAh/g<extra></extra>"))
    if p is not None:
        fig.add_trace(go.Scatter(
            x=[p.capacity], y=[p.voltage], mode="markers+text", name="Your material",
            marker=dict(symbol="star", size=22, color="#C8553D", line=dict(color="white", width=1.5)),
            error_y=dict(type="constant", value=p.typical_error or 0, color="#C8553D", thickness=1.5),
            text=[f"  {p.host}"], textposition="middle right", textfont=dict(size=13, color="#C8553D"),
            hovertemplate=f"<b>{p.host}</b><br>{p.voltage:.2f} V, {p.capacity:.0f} mAh/g, {p.energy:.0f} Wh/kg<extra></extra>"))
    fig.update_xaxes(title="Theoretical capacity (mAh/g)", range=[0, 600])
    fig.update_yaxes(title=f"Average voltage vs {ion} (V)", range=[0, 6.2])
    return _layout(fig, "Where does your material sit?", 470)


def run_predict(host, ion_choice, x):
    ion = core.ion_from_choice(ion_choice)
    try:
        p = core.predict_one(host, ion, x)
    except InputError as e:
        return _error_html(str(e)), "", map_figure(None, ion)
    return _result_card(p), _notes_html(p), map_figure(p, ion)


# ---------------------------------------------------------------------------
# Tab 2: compare ions
# ---------------------------------------------------------------------------
def run_compare(host, n_e, include_extra):
    """Compare ions at the same number of electrons per formula unit (x = n_e / ion charge)."""
    x = n_e
    ions = TRAINED_IONS + (["Rb", "Cs", "Al"] if include_extra else [])
    try:
        core.parse_host(host)
        if x is None or float(x) <= 0:
            raise InputError("The number of inserted ions per formula unit must be positive.")
    except InputError as e:
        return _error_html(str(e)), _empty_fig(), pd.DataFrame()
    present = {str(e) for e in core.Composition(host).elements}
    ions = [i for i in ions if i not in present]
    try:
        preds = core.predict_many([(host.strip(), i, float(x) / core.ION_CHARGE[i]) for i in ions])
    except Exception as e:
        return _error_html(f"Features could not be computed ({type(e).__name__})."), _empty_fig(), pd.DataFrame()

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=[p.ion for p in preds], y=[p.voltage for p in preds],
        marker_color=[ION_COLORS[p.ion] for p in preds],
        marker_pattern_shape=["" if p.label == "trained" else "/" for p in preds],
        error_y=dict(type="data", array=[p.typical_error or 0 for p in preds], color="#555"),
        text=[f"{p.voltage:.2f} V" for p in preds], textposition="outside",
        customdata=np.stack([[p.capacity for p in preds], [p.energy for p in preds], [p.label for p in preds]], axis=1),
        hovertemplate="<b>%{x}</b>: %{y:.2f} V<br>%{customdata[0]:.0f} mAh/g · %{customdata[1]:.0f} Wh/kg"
                      "<br>%{customdata[2]}<extra></extra>", showlegend=False))
    # DFT reference voltages where available
    m = core.reference_matches(host)
    if len(m):
        m = m[m["working_ion"].isin(ions)]
        fig.add_trace(go.Scatter(x=m["working_ion"], y=m["average_voltage"], mode="markers", name="Materials Project DFT",
                                 marker=dict(symbol="diamond", size=12, color="white", line=dict(color=INK, width=2))))
    fig.update_yaxes(title="Average voltage vs ion metal (V)", rangemode="tozero")
    fig.update_xaxes(title="Working ion  (hatched = not in training data)")
    _layout(fig, f"{core.Composition(host).reduced_formula}: predicted voltage for each working ion", 440)

    df = pd.DataFrame([p.as_row() for p in preds])
    df.insert(4, "E° (V vs SHE)", [E0[p.ion] for p in preds])
    best = max(preds, key=lambda p: p.energy)
    summary = (f"<div class='note-box'>Highest voltage: <b>{max(preds, key=lambda p: p.voltage).ion}</b> · "
               f"highest specific energy: <b>{best.ion}</b> ({best.energy:.0f} Wh/kg). "
               f"All ions are compared at {float(n_e):g} electron(s) per formula unit (so {float(n_e)/2:g} Mg, Ca or Zn). "
               "Voltages are measured against each ion's own metal, so they are not directly comparable "
               "between ions in a full cell; see the E° column.</div>")
    return summary, fig, df


# ---------------------------------------------------------------------------
# Tab 3: insertion explorer
# ---------------------------------------------------------------------------
def run_insertion(host, ion_choice, x_max):
    ion = core.ion_from_choice(ion_choice)
    try:
        core.parse_host(host, ion)
    except InputError as e:
        return _error_html(str(e)), _empty_fig()
    xs = np.round(np.linspace(0.1, float(x_max), 30), 3)
    try:
        preds = core.predict_many([(host.strip(), ion, float(x)) for x in xs])
    except Exception as e:
        return _error_html(f"Features could not be computed ({type(e).__name__})."), _empty_fig()
    v = np.array([p.voltage for p in preds]); c = np.array([p.capacity for p in preds])
    en = np.array([p.energy for p in preds])
    err = preds[0].typical_error or 0

    from plotly.subplots import make_subplots
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scatter(x=np.r_[xs, xs[::-1]], y=np.r_[v + err, (v - err)[::-1]], fill="toself",
                             fillcolor="rgba(31,111,139,.15)", line=dict(width=0), hoverinfo="skip",
                             name="± typical error"), secondary_y=False)
    fig.add_trace(go.Scatter(x=xs, y=v, mode="lines+markers", line=dict(color=ACCENT, width=3), name="Average voltage",
                             customdata=c, hovertemplate="x = %{x:.2f}<br>%{y:.2f} V<br>%{customdata:.0f} mAh/g<extra></extra>"),
                  secondary_y=False)
    fig.add_trace(go.Scatter(x=xs, y=en, mode="lines", line=dict(color="#C8553D", width=2, dash="dash"),
                             name="Specific energy", hovertemplate="x = %{x:.2f}<br>%{y:.0f} Wh/kg<extra></extra>"),
                  secondary_y=True)
    fig.update_xaxes(title=f"{ion} inserted per formula unit of {preds[0].host} (x)")
    fig.update_yaxes(title="Average voltage (V)", secondary_y=False, rangemode="tozero")
    fig.update_yaxes(title="Specific energy (Wh/kg)", secondary_y=True, rangemode="tozero", showgrid=False)
    _layout(fig, f"{preds[0].host} + x {ion}", 460)

    i = int(np.argmax(en))
    msg = (f"<div class='note-box'>Maximum theoretical specific energy at <b>x ≈ {xs[i]:.2f}</b>: "
           f"{en[i]:.0f} Wh/kg ({v[i]:.2f} V, {c[i]:.0f} mAh/g). "
           "This is the <i>average</i> voltage from x = 0 to the chosen x, not a full voltage profile, "
           "and large x may not be structurally possible.</div>")
    return msg, fig


# ---------------------------------------------------------------------------
# Tab 4: batch screening
# ---------------------------------------------------------------------------
def run_batch(text, file, ion_choices, x, min_v):
    hosts = core.parse_batch(text)                      # list of (formula, x or None)
    if file is not None:
        try:
            path = file if isinstance(file, str) else file.name
            up = pd.read_csv(path)
            col = next((c for c in up.columns if c.lower() in ("host", "formula", "framework", "framework_formula")),
                       up.columns[0])
            xcol = next((c for c in up.columns if c.lower() in ("x", "ions", "x_per_fu")), None)
            for _, r in up.dropna(subset=[col]).iterrows():
                xv = pd.to_numeric(r[xcol], errors="coerce") if xcol else np.nan
                hosts.append((str(r[col]).strip(), float(xv) if np.isfinite(xv) and xv > 0 else None))
        except Exception as e:
            return _error_html(f"Could not read the CSV file ({type(e).__name__})."), pd.DataFrame(), None, _empty_fig()
    hosts = list(dict.fromkeys(hosts))[:500]
    ions = [core.ion_from_choice(c) for c in (ion_choices or [])]
    if not hosts:
        return _error_html("Paste at least one formula, or upload a CSV with a 'host' column."), pd.DataFrame(), None, _empty_fig()
    if not ions:
        return _error_html("Choose at least one working ion."), pd.DataFrame(), None, _empty_fig()

    rows, skipped = [], []
    for h, hx in hosts:
        for ion in ions:
            try:
                p = core.predict_one(h, ion, hx if hx is not None else x)
                r = p.as_row(); r["Notes"] = "; ".join(n.split(";")[0] for n in p.notes)
                rows.append(r)
            except InputError as e:
                skipped.append(f"{h} ({ion}): {e}")
    df = pd.DataFrame(rows)
    if len(df):
        df = df[df["Voltage (V)"] >= float(min_v or 0)].sort_values("Energy (Wh/kg)", ascending=False).reset_index(drop=True)
        df.index = df.index + 1
    out = os.path.join(tempfile.gettempdir(), "voltage_screening_results.csv")
    df.to_csv(out, index=False)

    fig = go.Figure()
    for ion in ions:
        d = df[df["Ion"] == ion] if len(df) else df
        if len(d):
            fig.add_trace(go.Scatter(x=d["Capacity (mAh/g)"], y=d["Voltage (V)"], mode="markers", name=ion,
                                     marker=dict(size=11, color=ION_COLORS[ion], line=dict(color="white", width=1)),
                                     text=d["Host"], hovertemplate="<b>%{text}</b><br>%{y:.2f} V, %{x:.0f} mAh/g<extra></extra>"))
    fig.update_xaxes(title="Theoretical capacity (mAh/g)"); fig.update_yaxes(title="Average voltage (V)")
    _layout(fig, "Screening results", 420)

    msg = f"<div class='note-box'>Screened <b>{len(hosts)}</b> formula(s) × <b>{len(ions)}</b> ion(s): " \
          f"<b>{len(df)}</b> result(s) shown, sorted by specific energy."
    if skipped:
        msg += f" Skipped {len(skipped)}: " + "; ".join(skipped[:5]) + (" …" if len(skipped) > 5 else "")
    msg += "</div>"
    return msg, df, out, fig


# ---------------------------------------------------------------------------
# Tab 5: data explorer
# ---------------------------------------------------------------------------
def run_explore(ions, vrange_lo, vrange_hi, element, polyanion_only):
    if REF is None:
        return _empty_fig("Reference data not found."), pd.DataFrame(), ""
    d = REF[REF["working_ion"].isin(ions or [])]
    d = d[(d["average_voltage"] >= vrange_lo) & (d["average_voltage"] <= vrange_hi)]
    el = (element or "").strip()
    if el:
        wanted = [e.strip().capitalize() for e in el.replace(",", " ").split()]
        d = d[d["elements"].apply(lambda s: all(w in s.split() for w in wanted))]
    if polyanion_only:
        d = d[d["elements"].apply(lambda s: "O" in s.split() and any(p in s.split() for p in ("P", "S", "Si", "B")))]
    fig = go.Figure()
    for ion in ions or []:
        di = d[d["working_ion"] == ion]
        if len(di):
            fig.add_trace(go.Scatter(
                x=di["capacity_mAh_g"], y=di["average_voltage"], mode="markers", name=f"{ion} ({len(di)})",
                marker=dict(size=7, color=ION_COLORS[ion], opacity=0.7),
                customdata=np.stack([di["framework_formula"], di["formula_discharge"], di["energy_Wh_kg"]], axis=1),
                hovertemplate="<b>%{customdata[0]}</b> → %{customdata[1]}<br>%{y:.2f} V · %{x:.0f} mAh/g"
                              " · %{customdata[2]:.0f} Wh/kg<extra></extra>"))
    fig.update_xaxes(title="Capacity (mAh/g)", range=[0, 700]); fig.update_yaxes(title="Average voltage (V)")
    _layout(fig, "Materials Project insertion electrodes", 480)
    table = (d[["framework_formula", "working_ion", "average_voltage", "capacity_mAh_g", "energy_Wh_kg", "formula_discharge"]]
             .rename(columns={"framework_formula": "Host", "working_ion": "Ion", "average_voltage": "Voltage (V)",
                              "capacity_mAh_g": "Capacity (mAh/g)", "energy_Wh_kg": "Energy (Wh/kg)",
                              "formula_discharge": "Discharged"})
             .sort_values("Voltage (V)", ascending=False).round(2).head(300).reset_index(drop=True))
    info = f"<div class='note-box'><b>{len(d)}</b> electrodes match. Click a row to send it to the <b>Predict</b> tab.</div>"
    return fig, table, info


def table_to_predict(table: pd.DataFrame, evt: gr.SelectData):
    try:
        row = table.iloc[evt.index[0]]
        host, ion = str(row["Host"]), str(row["Ion"])
        choice = next(c for c in ION_CHOICES if c.startswith(ion + " "))
        x = 1.0
        if REF is not None:
            m = REF[(REF["framework_formula"] == host) & (REF["working_ion"] == ion)]
            if len(m) and np.isfinite(m["x_per_fu"].iloc[0]):
                x = float(round(m["x_per_fu"].iloc[0], 3))
        card, notes, fig = run_predict(host, choice, x)
        return host, choice, x, card, notes, fig, gr.Tabs(selected="predict")
    except Exception:
        return (gr.update(),) * 6 + (gr.Tabs(),)


# ---------------------------------------------------------------------------
# About text
# ---------------------------------------------------------------------------
def about_md() -> str:
    zs = META.get("mae_zero_shot_per_ion", {})
    zs_rows = "\n".join(f"| {i} | {E0[i]:+.2f} | {zs.get(i, float('nan')):.2f} |" for i in TRAINED_IONS)
    return f"""
## How the model works

**Data.** {META['n_training']:,} DFT-computed insertion electrodes for Li, Na, K, Mg, Ca and Zn from the
[Materials Project](https://materialsproject.org) (CC BY 4.0), cleaned to 0–6 V and near-stable discharged phases.

**Features (60).** Composition statistics of the host (electronegativity, radius, atomic number, mass, row,
group, Mendeleev number, ionisation energy), anion and polyanion fractions, working-ion properties, and the
average redox-metal oxidation state in the charged and discharged host from charge balance.

**Physics-informed target.** A voltage is measured against the metal of its own working ion, so every ion has
a different zero point. The model learns the host potential

$$E_\\text{{host}} = V + E^\\circ(\\text{{ion}})$$

on the common hydrogen-electrode scale, and converts back with $V = E_\\text{{host}} - E^\\circ(\\text{{ion}})$.
This roughly halves the error for ions the model has never seen, at no cost for known ions.

**Model.** XGBoost, {META['xgb_params']['n_estimators']} trees of depth {META['xgb_params']['max_depth']}.

## Accuracy

* Unseen host materials, all six ions: **MAE {META['mae_in_distribution']:.2f} ± {META.get('mae_in_distribution_std', 0):.2f} V**
  (10 framework-grouped splits; no host appears in both training and test sets).
* Ion left out of training entirely (zero-shot, known hosts): **MAE {META['mae_zero_shot_known_hosts']:.2f} V** on average.

| Ion | E° (V vs SHE) | Zero-shot MAE (V) |
|---|---|---|
{zs_rows}

## Limitations

* Predictions reproduce **DFT (GGA/GGA+U) average voltages**, not experimental values or voltage profiles.
* Anode-type hosts (graphite, Si, Sb), voltages above 5 V and rare chemistries are less reliable.
* Rb and Cs are zero-shot predictions; trivalent Al is outside the training data.
* Capacity and energy are theoretical values for the chosen x and ignore structural limits.

## Cite

Saddique, S. (2026). *Physics-Informed Machine Learning of Battery Electrode Voltages: From Composition to
Cross-Ion Transfer.* Code and report: see the GitHub repository.
"""


# ---------------------------------------------------------------------------
# Interface
# ---------------------------------------------------------------------------
with gr.Blocks(title="Electrode Voltage Predictor") as demo:
    gr.HTML(f"""
<div style="display:flex;align-items:center;gap:14px;padding:6px 2px 2px">
  <div style="font-size:40px">🔋</div>
  <div><div style="font-size:28px;font-weight:700;color:{INK}">Electrode Voltage Predictor</div>
  <div style="color:#5b6b78">Physics-informed machine learning for Li, Na, K, Mg, Ca and Zn battery electrodes ·
  trained on {META['n_training']:,} DFT calculations · typical error ± {META['mae_in_distribution']:.2f} V</div></div>
</div>""")

    with gr.Tabs() as tabs:
        # ---- Predict ----
        with gr.Tab("🔋 Predict", id="predict"):
            with gr.Row():
                with gr.Column(scale=4):
                    p_host = gr.Textbox(label="Host formula (charged, ion-free)", value="FePO4",
                                        placeholder="e.g. FePO4, MnO2, V2O5, Fe2(SO4)3")
                    p_ion = gr.Radio(ION_CHOICES, value="Li (trained)", label="Working ion")
                    p_x = gr.Slider(0.1, 4, value=1, step=0.05, label="Ions inserted per formula unit (x)")
                    gr.Markdown("<small>Enter the <b>charged</b> host without the working ion, e.g. FePO4 for LiFePO4.</small>")
                    p_btn = gr.Button("Predict voltage", variant="primary", size="lg")
                    gr.Examples([["FePO4", "Li (trained)", 1], ["FePO4", "Na (trained)", 1],
                                 ["CoO2", "Li (trained)", 1], ["MnO2", "Zn (trained)", 0.5],
                                 ["V2O5", "Mg (trained)", 1], ["Fe2(SO4)3", "Na (trained)", 2],
                                 ["TiS2", "Li (trained)", 1], ["FePO4", "Rb (zero-shot)", 1]],
                                inputs=[p_host, p_ion, p_x], label="Try an example")
                with gr.Column(scale=5):
                    p_card = gr.HTML()
                    p_notes = gr.HTML()
            p_plot = gr.Plot(label="Voltage–capacity map")
            p_inputs = [p_host, p_ion, p_x]
            p_outputs = [p_card, p_notes, p_plot]
            p_btn.click(run_predict, p_inputs, p_outputs)
            p_host.submit(run_predict, p_inputs, p_outputs)
            p_ion.change(run_predict, p_inputs, p_outputs)
            p_x.release(run_predict, p_inputs, p_outputs)

        # ---- Compare ions ----
        with gr.Tab("⚖️ Compare ions"):
            gr.Markdown("Which working ion suits this host best? The same host is predicted with every ion.")
            with gr.Row():
                c_host = gr.Dropdown(EXAMPLE_HOSTS, value="FePO4", allow_custom_value=True, label="Host formula", scale=3)
                c_x = gr.Number(value=1, label="Electrons per formula unit", minimum=0.01, scale=1)
                c_extra = gr.Checkbox(value=False, label="Include Rb, Cs, Al (not in training data)", scale=2)
                c_btn = gr.Button("Compare", variant="primary", scale=1)
            c_msg = gr.HTML()
            c_plot = gr.Plot()
            c_table = gr.Dataframe(interactive=False, wrap=True)
            c_btn.click(run_compare, [c_host, c_x, c_extra], [c_msg, c_plot, c_table])
            c_host.change(run_compare, [c_host, c_x, c_extra], [c_msg, c_plot, c_table])

        # ---- Insertion explorer ----
        with gr.Tab("📈 Insertion explorer"):
            gr.Markdown("How do average voltage and specific energy change as more ions are inserted?")
            with gr.Row():
                i_host = gr.Dropdown(EXAMPLE_HOSTS, value="V2O5", allow_custom_value=True, label="Host formula", scale=3)
                i_ion = gr.Dropdown(ION_CHOICES, value="Li (trained)", label="Working ion", scale=2)
                i_xmax = gr.Slider(0.5, 6, value=3, step=0.5, label="Maximum x", scale=2)
                i_btn = gr.Button("Plot", variant="primary", scale=1)
            i_msg = gr.HTML()
            i_plot = gr.Plot()
            for trig in (i_btn.click, i_ion.change, i_xmax.release, i_host.change):
                trig(run_insertion, [i_host, i_ion, i_xmax], [i_msg, i_plot])

        # ---- Batch screening ----
        with gr.Tab("🧪 Batch screening"):
            gr.Markdown("Screen many candidate hosts at once. Paste formulas (one per line) and/or upload a CSV "
                        "with a `host` column. To give a host its own x, write it after the formula "
                        "(e.g. `Fe2(SO4)3 2`) or add an `x` column to the CSV; otherwise the default x is used. "
                        "Up to 500 formulas.")
            with gr.Row():
                with gr.Column(scale=3):
                    b_text = gr.Textbox(lines=8, label="Host formulas",
                                        value="FePO4\nMnPO4\nCoPO4\nVOPO4\nFe2(SO4)3 2\nMnO2\nNiO2\nV2O5\nFeF3\nTiS2")
                    b_file = gr.File(label="…or upload a CSV", file_types=[".csv"])
                with gr.Column(scale=2):
                    b_ions = gr.CheckboxGroup(ION_CHOICES, value=["Li (trained)", "Na (trained)"], label="Working ions")
                    b_x = gr.Number(value=1, label="Default ions per formula unit (x)", minimum=0.01)
                    b_min = gr.Slider(0, 5, value=0, step=0.1, label="Show only voltages above (V)")
                    b_btn = gr.Button("Run screening", variant="primary", size="lg")
            b_msg = gr.HTML()
            b_plot = gr.Plot()
            b_table = gr.Dataframe(interactive=False, wrap=True)
            b_download = gr.File(label="Download results (CSV)")
            b_btn.click(run_batch, [b_text, b_file, b_ions, b_x, b_min], [b_msg, b_table, b_download, b_plot])

        # ---- Data explorer ----
        with gr.Tab("📚 Explore the data"):
            gr.Markdown("Browse the Materials Project electrodes the model learned from. Hover for details; "
                        "click a table row to predict it.")
            with gr.Row():
                e_ions = gr.CheckboxGroup(TRAINED_IONS, value=["Li", "Na", "Mg"], label="Working ions", scale=3)
                e_el = gr.Textbox(label="Must contain element(s)", placeholder="e.g. Fe  or  Mn P", scale=2)
                e_poly = gr.Checkbox(label="Polyanion hosts only", value=False, scale=1)
            with gr.Row():
                e_lo = gr.Slider(0, 6, value=0, step=0.1, label="Minimum voltage (V)")
                e_hi = gr.Slider(0, 6, value=6, step=0.1, label="Maximum voltage (V)")
            e_info = gr.HTML()
            e_plot = gr.Plot()
            e_table = gr.Dataframe(interactive=False, wrap=True, max_height=420)
            e_inputs = [e_ions, e_lo, e_hi, e_el, e_poly]
            for comp in (e_ions, e_poly):
                comp.change(run_explore, e_inputs, [e_plot, e_table, e_info])
            for comp in (e_lo, e_hi):
                comp.release(run_explore, e_inputs, [e_plot, e_table, e_info])
            e_el.submit(run_explore, e_inputs, [e_plot, e_table, e_info])
            e_table.select(table_to_predict, [e_table], [p_host, p_ion, p_x, p_card, p_notes, p_plot, tabs]).then(
                None, None, None, js="() => window.scrollTo({top: 0, behavior: 'smooth'})")

        # ---- About ----
        with gr.Tab("ℹ️ About"):
            gr.Markdown(about_md(), latex_delimiters=[{"left": "$$", "right": "$$", "display": True},
                                                       {"left": "$", "right": "$", "display": False}])

    gr.HTML("<div style='text-align:center;color:#8a99a6;font-size:13px;padding:10px'>"
            "Shaista Saddique · Data: Materials Project (CC BY 4.0) · Predictions are DFT-level estimates, "
            "not experimental voltages.</div>")

    demo.load(run_predict, p_inputs, p_outputs)
    demo.load(run_compare, [c_host, c_x, c_extra], [c_msg, c_plot, c_table])
    demo.load(run_insertion, [i_host, i_ion, i_xmax], [i_msg, i_plot])
    demo.load(run_explore, e_inputs, [e_plot, e_table, e_info])


if __name__ == "__main__":
    demo.launch(theme=THEME, css=CSS, share=bool(os.environ.get("GRADIO_SHARE")))
