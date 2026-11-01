# Publishing guide

## 1. Put the project on GitHub

### Option A: in the browser (no git needed)

1. Sign in at [github.com](https://github.com) → **+** (top right) → **New repository**.
2. Repository name: `electrode-voltage-ml`. Description: *Physics-informed ML of battery electrode voltages
   (Li, Na, K, Mg, Ca, Zn) with an interactive app*. Choose **Public**. Do **not** tick "Add a README".
3. Click **Create repository**, then **uploading an existing file**.
4. Unzip `electrode-voltage-ml.zip` on your computer, open the `electrode-voltage-ml` folder, select
   **everything inside it** (including the `.github` folder; on a Mac press `Cmd + Shift + .` to show hidden
   folders) and drag it into the upload area.
5. Commit message: `Initial release`. Click **Commit changes**.
6. In `README.md`, replace `shaistasaddique` with your username (pencil icon → edit → commit).

### Option B: with git

```bash
cd electrode-voltage-ml
git init
git add .
git commit -m "Initial release"
git branch -M main
git remote add origin https://github.com/shaistasaddique/electrode-voltage-ml.git
git push -u origin main
```

### Make the repository look finished

* **About** (gear icon on the repository page): add the description, the website (your Hugging Face Space link)
  and topics: `materials-science`, `battery`, `machine-learning`, `materials-project`, `xgboost`, `dft`,
  `gradio`, `electrochemistry`.
* **Releases** → *Create a new release* → tag `v1.0.0` → attach `report/Electrode_Voltage_ML_Book.pdf`.
* Optional: connect the repository to [Zenodo](https://zenodo.org/account/settings/github/) before making the
  release to get a citable DOI for your CV.
* The **Actions** tab will show a green tick once the automated tests pass.

## 2. Host the app permanently on Hugging Face Spaces (free)

1. Create an account at [huggingface.co](https://huggingface.co) → **New Space**.
2. Space name `electrode-voltage-predictor`, SDK **Gradio**, hardware **CPU basic (free)**, Public.
3. **Files** → **Add file** → **Upload files** → upload everything inside the `app/` folder
   (`app.py`, `core.py`, `featurize.py`, `voltage_model_e0.json`, `app_metadata.json`,
   `reference_voltages.csv`, `requirements.txt`, `README.md`).
4. The Space builds in 2–4 minutes and gives a permanent link such as
   `https://huggingface.co/spaces/YOUR_HF_USERNAME/electrode-voltage-predictor`.
5. Add that link to the GitHub README (top badge) and to the repository's **About** box.

If the build fails because of the Gradio version, change `sdk_version` in `app/README.md` to the version shown
in the error message.

## 3. Run it on Google Colab for a quick demo

```python
!git clone https://github.com/shaistasaddique/electrode-voltage-ml.git
%cd electrode-voltage-ml/app
!pip install -q -r requirements.txt
import os; os.environ["GRADIO_SHARE"] = "1"
!python app.py
```

Colab prints a temporary public link (valid for 72 hours).
