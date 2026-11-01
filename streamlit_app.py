"""
Electrode Voltage Predictor -- Streamlit version
=================================================
Same model and prediction code as the Gradio app in app/ (app/core.py), packaged for
Streamlit Community Cloud.

Run locally:   pip install -r requirements.txt  &&  streamlit run streamlit_app.py
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "app"))
import core  # noqa: E402
from core import E0, META, REF, TRAINED_IONS, InputError  # noqa: E402

st.set_page_config(page_title="Electrode Voltage Predictor", page_icon="🔋", layout="wide")

ION_COLORS = {"Li": "#1f77b4", "Na": "#ff7f0e", "K": "#2ca02c", "Mg": "#d62728", "Ca": "#9467bd",
              "Zn": "#8c564b", "Rb": "#e377c2", "Cs": "#7f7f7f", "Al": "#bcbd22"}
ACCENT, INK, WARM = "#1F6F8B", "#1B2A41", "#C8553D"
ION_LABELS = {i: f"{i} (trained)" for i in TRAINED_IONS} | {"Rb": "Rb (zero-shot)", "Cs": "Cs (zero-shot)",
                                                            "Al": "Al (experimental)"}
ALL_IONS = list(ION_LABELS)
EXAMPLE_HOSTS = ["FePO4", "CoO2", "MnO2", "V2O5", "Fe2(SO4)3", "NiO2", "FeF3", "TiS2", "MoO3", "VOPO4"]

st.markdown("""
<style>
.result-card {border-radius: 14px; padding: 18px 22px; background: linear-gradient(135deg,#1B2A41 0%,#1F6F8B 100%);}
.result-card, .result-card * {color: #ffffff !important;}
.result-card .v {font-size: 46px; font-weight: 700; line-height: 1.05;}
.result-card .sub {opacity: .85; font-size: 15px; margin-top: 4px;}
.result-card .grid {display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px; margin-top: 14px;}
.result-card .cell {background: rgba(255,255,255,.12); border-radius: 10px; padding: 8px 10px;}
.result-card .k {font-size: 12px; opacity: .8; text-transform: uppercase; letter-spacing: .04em;}
.result-card .val {font-size: 20px; font-weight: 600;}
.badge {display:inline-block; border-radius:999px; padding:2px 10px; font-size:12px; font-weight:600; margin-left:8px;}
.result-card .badge.trained {background:#d1fae5; color:#065f46 !important;}
.result-card .badge.zero-shot {background:#fef3c7; color:#92400e !important;}
.result-card .badge.experimental {background:#fee2e2; color:#991b1b !important;}
</style>""", unsafe_allow_html=True)


def layout(fig: go.Figure, title: str, height: int = 440) -> go.Figure:
    fig.update_layout(title=dict(text=title, x=0.01, font=dict(size=16)), height=height,
                      template="plotly_white", margin=dict(l=60, r=30, t=60, b=55),
                      legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="right", x=1))
    return fig


def show(fig: go.Figure):
    st.plotly_chart(fig, width="stretch", config={"displaylogo": False})


@st.cache_data(show_spinner=False)
def cached_predict(host: str, ion: str, x: float) -> core.Prediction:
    return core.predict_one(host, ion, x)


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.markdown(f"""
<div style="display:flex;align-items:center;gap:14px;padding:4px 0 6px">
  <div style="font-size:42px">🔋</div>
  <div><div style="font-size:30px;font-weight:700">Electrode Voltage Predictor</div>
  <div style="opacity:.75">Physics-informed machine learning for Li, Na, K, Mg, Ca and Zn battery electrodes ·
  trained on {META['n_training']:,} DFT calculations · typical error ± {META['mae_in_distribution']:.2f} V</div></div>
</div>""", unsafe_allow_html=True)

tab_pred, tab_cmp, tab_ins, tab_batch, tab_data, tab_about = st.tabs(
    ["🔋 Predict", "⚖️ Compare ions", "📈 Insertion explorer", "🧪 Batch screening", "📚 Explore the data", "ℹ️ About"])


# ---------------------------------------------------------------------------
# Tab 1: Predict
# ---------------------------------------------------------------------------
def result_card(p: core.Prediction) -> str:
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


def map_figure(p: core.Prediction | None, ion: str) -> go.Figure:
    fig = go.Figure()
    caps = np.linspace(20, 600, 200)
    for e in (200, 400, 600, 800, 1000):
        fig.add_trace(go.Scatter(x=caps, y=e / caps, mode="lines", line=dict(color="#c9d3da", dash="dot", width=1),
                                 hoverinfo="skip", showlegend=False))
        xa = min(590, e / 5.6)
        fig.add_annotation(x=xa, y=e / xa, text=f"{e} Wh/kg", showarrow=False, font=dict(size=10, color="#9aa7b0"),
                           xanchor="left", yanchor="bottom")
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
            marker=dict(symbol="star", size=22, color=WARM, line=dict(color="white", width=1.5)),
            error_y=dict(type="constant", value=p.typical_error or 0, color=WARM, thickness=1.5),
            text=[f"  {p.host}"], textposition="middle right", textfont=dict(size=13, color=WARM),
            hovertemplate=f"<b>{p.host}</b><br>{p.voltage:.2f} V, {p.capacity:.0f} mAh/g<extra></extra>"))
    fig.update_xaxes(title="Theoretical capacity (mAh/g)", range=[0, 600])
    fig.update_yaxes(title=f"Average voltage vs {ion} (V)", range=[0, 6.2])
    return layout(fig, "Where does your material sit?", 470)


with tab_pred:
    left, right = st.columns([4, 5], gap="large")
    with left:
        host = st.text_input("Host formula (charged, ion-free)", value=st.session_state.get("pred_host", "FePO4"),
                             key="pred_host_input", help="Enter the charged host without the working ion, "
                                                         "e.g. FePO4 for LiFePO4.")
        ion = st.radio("Working ion", ALL_IONS, format_func=ION_LABELS.get, horizontal=True,
                       index=ALL_IONS.index(st.session_state.get("pred_ion", "Li")), key="pred_ion_input")
        x = st.slider("Ions inserted per formula unit (x)", 0.1, 4.0,
                      float(st.session_state.get("pred_x", 1.0)), 0.05, key="pred_x_input")
        st.caption("Try: FePO4 + Li · CoO2 + Li · MnO2 + Zn (x = 0.5) · V2O5 + Mg · Fe2(SO4)3 + Na (x = 2) · "
                   "FePO4 + Rb")
    pred = None
    with right:
        try:
            pred = cached_predict(host, ion, x)
            st.markdown(result_card(pred), unsafe_allow_html=True)
            m = core.reference_matches(pred.host, pred.ion)
            if len(m):
                vals = ", ".join(f"{v:.2f} V" for v in m["average_voltage"].head(3))
                st.info(f"📚 In the Materials Project data for {pred.ion}: DFT average voltage {vals} "
                        "(this host may have been in the training set).")
            for n in pred.notes:
                st.caption("• " + n)
        except InputError as e:
            st.error(str(e))
    show(map_figure(pred, ion))


# ---------------------------------------------------------------------------
# Tab 2: Compare ions
# ---------------------------------------------------------------------------
with tab_cmp:
    st.write("Which working ion suits this host best? Ions are compared at the same number of electrons per formula unit.")
    c1, c2, c3 = st.columns([3, 1.3, 2])
    c_host = c1.text_input("Host formula", "FePO4", key="cmp_host",
                           help="Examples: " + ", ".join(EXAMPLE_HOSTS))
    n_e = c2.number_input("Electrons per formula unit", min_value=0.05, value=1.0, step=0.5)
    extra = c3.checkbox("Include Rb, Cs, Al (not in training data)")
    try:
        core.parse_host(c_host)
        present = {str(e) for e in core.Composition(c_host).elements}
        ions = [i for i in TRAINED_IONS + (["Rb", "Cs", "Al"] if extra else []) if i not in present]
        preds = core.predict_many([(c_host.strip(), i, float(n_e) / core.ION_CHARGE[i]) for i in ions])
        fig = go.Figure(go.Bar(
            x=[p.ion for p in preds], y=[p.voltage for p in preds],
            marker_color=[ION_COLORS[p.ion] for p in preds],
            marker_pattern_shape=["" if p.label == "trained" else "/" for p in preds],
            error_y=dict(type="data", array=[p.typical_error or 0 for p in preds], color="#555"),
            text=[f"{p.voltage:.2f} V" for p in preds], textposition="outside", showlegend=False,
            hovertemplate="<b>%{x}</b>: %{y:.2f} V<extra></extra>"))
        m = core.reference_matches(c_host)
        if len(m):
            m = m[m["working_ion"].isin(ions)]
            fig.add_trace(go.Scatter(x=m["working_ion"], y=m["average_voltage"], mode="markers",
                                     name="Materials Project DFT",
                                     marker=dict(symbol="diamond", size=12, color="white", line=dict(color=INK, width=2))))
        fig.update_yaxes(title="Average voltage vs ion metal (V)", rangemode="tozero")
        fig.update_xaxes(title="Working ion  (hatched = not in training data)")
        best = max(preds, key=lambda p: p.energy)
        st.info(f"Highest voltage: **{max(preds, key=lambda p: p.voltage).ion}** · highest specific energy: "
                f"**{best.ion}** ({best.energy:.0f} Wh/kg). Voltages are measured against each ion's own metal, "
                "so compare full cells using the E° column.")
        show(layout(fig, f"{preds[0].host}: predicted voltage for each working ion"))
        df = pd.DataFrame([p.as_row() for p in preds])
        df.insert(4, "E° (V vs SHE)", [E0[p.ion] for p in preds])
        st.dataframe(df, hide_index=True, width="stretch")
    except InputError as e:
        st.error(str(e))
    except Exception as e:
        st.error(f"Features could not be computed ({type(e).__name__}).")


# ---------------------------------------------------------------------------
# Tab 3: Insertion explorer
# ---------------------------------------------------------------------------
with tab_ins:
    st.write("How do average voltage and specific energy change as more ions are inserted?")
    c1, c2, c3 = st.columns([3, 2, 2])
    i_host = c1.text_input("Host formula", "V2O5", key="ins_host")
    i_ion = c2.selectbox("Working ion", ALL_IONS, format_func=ION_LABELS.get, key="ins_ion")
    x_max = c3.slider("Maximum x", 0.5, 6.0, 3.0, 0.5)
    try:
        core.parse_host(i_host, i_ion)
        xs = np.round(np.linspace(0.1, x_max, 30), 3)
        preds = core.predict_many([(i_host.strip(), i_ion, float(v)) for v in xs])
        v = np.array([p.voltage for p in preds]); c = np.array([p.capacity for p in preds])
        en = np.array([p.energy for p in preds]); err = preds[0].typical_error or 0
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(go.Scatter(x=np.r_[xs, xs[::-1]], y=np.r_[v + err, (v - err)[::-1]], fill="toself",
                                 fillcolor="rgba(31,111,139,.15)", line=dict(width=0), hoverinfo="skip",
                                 name="± typical error"), secondary_y=False)
        fig.add_trace(go.Scatter(x=xs, y=v, mode="lines+markers", line=dict(color=ACCENT, width=3), name="Average voltage",
                                 customdata=c, hovertemplate="x = %{x:.2f}<br>%{y:.2f} V<br>%{customdata:.0f} mAh/g<extra></extra>"),
                      secondary_y=False)
        fig.add_trace(go.Scatter(x=xs, y=en, mode="lines", line=dict(color=WARM, width=2, dash="dash"),
                                 name="Specific energy", hovertemplate="x = %{x:.2f}<br>%{y:.0f} Wh/kg<extra></extra>"),
                      secondary_y=True)
        fig.update_xaxes(title=f"{i_ion} inserted per formula unit of {preds[0].host} (x)")
        fig.update_yaxes(title="Average voltage (V)", secondary_y=False, rangemode="tozero")
        fig.update_yaxes(title="Specific energy (Wh/kg)", secondary_y=True, rangemode="tozero", showgrid=False)
        k = int(np.argmax(en))
        st.info(f"Maximum theoretical specific energy at **x ≈ {xs[k]:.2f}**: {en[k]:.0f} Wh/kg "
                f"({v[k]:.2f} V, {c[k]:.0f} mAh/g). This is the average voltage from x = 0 to the chosen x, "
                "not a full voltage profile, and large x may not be structurally possible.")
        show(layout(fig, f"{preds[0].host} + x {i_ion}", 460))
    except InputError as e:
        st.error(str(e))


# ---------------------------------------------------------------------------
# Tab 4: Batch screening
# ---------------------------------------------------------------------------
with tab_batch:
    st.write("Screen many candidate hosts at once. Paste formulas (one per line) and/or upload a CSV with a `host` "
             "column. To give a host its own x, write it after the formula (e.g. `Fe2(SO4)3 2`) or add an `x` "
             "column to the CSV. Up to 500 formulas.")
    c1, c2 = st.columns([3, 2])
    with c1:
        text = st.text_area("Host formulas", height=210,
                            value="FePO4\nMnPO4\nCoPO4\nVOPO4\nFe2(SO4)3 2\nMnO2\nNiO2\nV2O5\nFeF3\nTiS2")
        upload = st.file_uploader("…or upload a CSV", type=["csv"])
    with c2:
        b_ions = st.multiselect("Working ions", ALL_IONS, default=["Li", "Na"], format_func=ION_LABELS.get)
        b_x = st.number_input("Default ions per formula unit (x)", min_value=0.01, value=1.0, step=0.5)
        b_min = st.slider("Show only voltages above (V)", 0.0, 5.0, 0.0, 0.1)
        run = st.button("Run screening", type="primary", width="stretch")
    if run:
        hosts = core.parse_batch(text)
        if upload is not None:
            try:
                up = pd.read_csv(upload)
                col = next((c for c in up.columns if c.lower() in ("host", "formula", "framework", "framework_formula")),
                           up.columns[0])
                xcol = next((c for c in up.columns if c.lower() in ("x", "ions", "x_per_fu")), None)
                for _, r in up.dropna(subset=[col]).iterrows():
                    xv = pd.to_numeric(r[xcol], errors="coerce") if xcol else np.nan
                    hosts.append((str(r[col]).strip(), float(xv) if np.isfinite(xv) and xv > 0 else None))
            except Exception as e:
                st.error(f"Could not read the CSV file ({type(e).__name__}).")
        hosts = list(dict.fromkeys(hosts))[:500]
        if not hosts or not b_ions:
            st.warning("Add at least one formula and choose at least one working ion.")
        else:
            rows, skipped = [], []
            for h, hx in hosts:
                for ion_ in b_ions:
                    try:
                        p = core.predict_one(h, ion_, hx if hx is not None else b_x)
                        r = p.as_row(); r["Notes"] = "; ".join(p.notes)
                        rows.append(r)
                    except InputError as e:
                        skipped.append(f"{h} ({ion_}): {e}")
            df = pd.DataFrame(rows)
            if len(df):
                df = df[df["Voltage (V)"] >= b_min].sort_values("Energy (Wh/kg)", ascending=False)
            st.success(f"Screened {len(hosts)} formula(s) × {len(b_ions)} ion(s): {len(df)} result(s), "
                       "sorted by specific energy.")
            if skipped:
                with st.expander(f"Skipped {len(skipped)} entries"):
                    st.write("\n".join(f"- {s}" for s in skipped))
            if len(df):
                fig = go.Figure()
                for ion_ in b_ions:
                    d = df[df["Ion"] == ion_]
                    if len(d):
                        fig.add_trace(go.Scatter(x=d["Capacity (mAh/g)"], y=d["Voltage (V)"], mode="markers", name=ion_,
                                                 marker=dict(size=11, color=ION_COLORS[ion_], line=dict(color="white", width=1)),
                                                 text=d["Host"], hovertemplate="<b>%{text}</b><br>%{y:.2f} V, %{x:.0f} mAh/g<extra></extra>"))
                fig.update_xaxes(title="Theoretical capacity (mAh/g)"); fig.update_yaxes(title="Average voltage (V)")
                show(layout(fig, "Screening results", 420))
                st.dataframe(df, hide_index=True, width="stretch")
                st.download_button("Download results (CSV)", df.to_csv(index=False).encode(),
                                   "voltage_screening_results.csv", "text/csv")


# ---------------------------------------------------------------------------
# Tab 5: Explore the data
# ---------------------------------------------------------------------------
with tab_data:
    if REF is None:
        st.warning("Reference data not found.")
    else:
        st.write("Browse the Materials Project electrodes the model learned from. Hover over points for details; "
                 "select a table row to predict it.")
        c1, c2, c3 = st.columns([3, 2, 1.3])
        e_ions = c1.multiselect("Working ions", TRAINED_IONS, default=["Li", "Na", "Mg"])
        e_el = c2.text_input("Must contain element(s)", placeholder="e.g. Fe  or  Mn P")
        e_poly = c3.checkbox("Polyanion hosts only")
        lo, hi = st.slider("Voltage range (V)", 0.0, 6.0, (0.0, 6.0), 0.1)
        d = REF[REF["working_ion"].isin(e_ions)]
        d = d[(d["average_voltage"] >= lo) & (d["average_voltage"] <= hi)]
        if e_el.strip():
            wanted = [w.strip().capitalize() for w in e_el.replace(",", " ").split()]
            d = d[d["elements"].apply(lambda s: all(w in s.split() for w in wanted))]
        if e_poly:
            d = d[d["elements"].apply(lambda s: "O" in s.split() and any(p in s.split() for p in ("P", "S", "Si", "B")))]
        st.caption(f"**{len(d)}** electrodes match.")
        fig = go.Figure()
        for ion_ in e_ions:
            di = d[d["working_ion"] == ion_]
            if len(di):
                fig.add_trace(go.Scatter(
                    x=di["capacity_mAh_g"], y=di["average_voltage"], mode="markers", name=f"{ion_} ({len(di)})",
                    marker=dict(size=7, color=ION_COLORS[ion_], opacity=0.7),
                    customdata=np.stack([di["framework_formula"], di["formula_discharge"], di["energy_Wh_kg"]], axis=1),
                    hovertemplate="<b>%{customdata[0]}</b> → %{customdata[1]}<br>%{y:.2f} V · %{x:.0f} mAh/g"
                                  " · %{customdata[2]:.0f} Wh/kg<extra></extra>"))
        fig.update_xaxes(title="Capacity (mAh/g)", range=[0, 700]); fig.update_yaxes(title="Average voltage (V)")
        show(layout(fig, "Materials Project insertion electrodes", 480))
        table = (d[["framework_formula", "working_ion", "average_voltage", "capacity_mAh_g", "energy_Wh_kg",
                    "formula_discharge", "x_per_fu"]]
                 .rename(columns={"framework_formula": "Host", "working_ion": "Ion", "average_voltage": "Voltage (V)",
                                  "capacity_mAh_g": "Capacity (mAh/g)", "energy_Wh_kg": "Energy (Wh/kg)",
                                  "formula_discharge": "Discharged", "x_per_fu": "x"})
                 .sort_values("Voltage (V)", ascending=False).round(2).head(300).reset_index(drop=True))
        event = st.dataframe(table.drop(columns="x"), hide_index=True, width="stretch", height=380,
                             on_select="rerun", selection_mode="single-row", key="explore_table")
        rows = event.selection.rows if event and hasattr(event, "selection") else []
        if rows:
            r = table.iloc[rows[0]]
            xr = float(r["x"]) if np.isfinite(r["x"]) else 1.0
            try:
                p = cached_predict(str(r["Host"]), str(r["Ion"]), xr)
                st.markdown(f"#### Model prediction for {r['Host']} + {xr:g} {r['Ion']}")
                cc1, cc2 = st.columns([3, 2])
                cc1.markdown(result_card(p), unsafe_allow_html=True)
                cc2.metric("Materials Project DFT voltage", f"{r['Voltage (V)']:.2f} V",
                           delta=f"{p.voltage - r['Voltage (V)']:+.2f} V (model − DFT)", delta_color="off")
            except InputError as e:
                st.error(str(e))


# ---------------------------------------------------------------------------
# Tab 6: About
# ---------------------------------------------------------------------------
with tab_about:
    zs = META.get("mae_zero_shot_per_ion", {})
    st.markdown(f"""
### How the model works
**Data.** {META['n_training']:,} DFT-computed insertion electrodes for Li, Na, K, Mg, Ca and Zn from the
[Materials Project](https://materialsproject.org) (CC BY 4.0), cleaned to 0–6 V and near-stable discharged phases.

**Features (60).** Composition statistics of the host (electronegativity, radius, atomic number, mass, row, group,
Mendeleev number, ionisation energy), anion and polyanion fractions, working-ion properties, and the average
redox-metal oxidation state in the charged and discharged host from charge balance.

**Physics-informed target.** A voltage is measured against the metal of its own working ion, so every ion has a
different zero point. The model learns the host potential on the common hydrogen-electrode scale and converts back:""")
    st.latex(r"E_\text{host} = V + E^\circ(\text{ion}), \qquad V = E_\text{host} - E^\circ(\text{ion})")
    st.markdown(f"""
This roughly halves the error for ions the model has never seen, at no cost for known ions.

**Model.** XGBoost, {META['xgb_params']['n_estimators']} trees of depth {META['xgb_params']['max_depth']}.

### Accuracy
* Unseen host materials, all six ions: **MAE {META['mae_in_distribution']:.2f} ± {META.get('mae_in_distribution_std', 0):.2f} V**
  (10 framework-grouped splits).
* Ion left out of training entirely (zero-shot, known hosts): **MAE {META['mae_zero_shot_known_hosts']:.2f} V** on average.
""")
    st.dataframe(pd.DataFrame({"Ion": TRAINED_IONS, "E° (V vs SHE)": [E0[i] for i in TRAINED_IONS],
                               "Zero-shot MAE (V)": [round(zs.get(i, np.nan), 2) for i in TRAINED_IONS]}),
                 hide_index=True)
    st.markdown("""
### Limitations
* Predictions reproduce **DFT (GGA/GGA+U) average voltages**, not experimental values or voltage profiles.
* Anode-type hosts (graphite, Si, Sb), voltages above 5 V and rare chemistries are less reliable.
* Rb and Cs are zero-shot predictions; trivalent Al is outside the training data.
* Capacity and energy are theoretical values for the chosen x and ignore structural limits.

### Code and report
[github.com/shaistasaddique/electrode-voltage-ml](https://github.com/shaistasaddique/electrode-voltage-ml)

*Shaista Saddique · Data: Materials Project (CC BY 4.0)*
""")
