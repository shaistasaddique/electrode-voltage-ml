---
title: Electrode Voltage Predictor
emoji: 🔋
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 6.31.0
app_file: app.py
pinned: false
license: mit
short_description: Predict battery electrode voltages for Li, Na, K, Mg, Ca, Zn
---

# 🔋 Electrode Voltage Predictor

Predict the average voltage, capacity and specific energy of a battery insertion-electrode host for
Li, Na, K, Mg, Ca or Zn from its chemical formula.

* XGBoost model trained on 2,634 DFT-computed electrodes from the Materials Project (CC BY 4.0)
* Physics-informed target E = V + E°(ion), which transfers to ions not seen in training
* Typical error ± 0.39 V on unseen host materials

Tabs: Predict · Compare ions · Insertion explorer · Batch screening · Explore the data · About

Author: Shaista Saddique. Full code, notebooks and report: see the GitHub repository.
