# 🌾 Crop Yield Prediction System

A simple Streamlit web app that predicts crop yield (quintal/hectare) based on agricultural and environmental inputs, using a K-Nearest Neighbors regression model trained on historical crop data.

## Features

- Interactive UI built with Streamlit
- Predicts yield using crop type, state, season, area, production, rainfall, fertilizer, and pesticide usage
- Model and encoders are cached (`@st.cache_resource`) so training runs only once per session
- Dropdowns are auto-populated from the dataset (crops, states, seasons)

## Dataset

**File:** `crop_yield.csv` (19,689 records)

| Column | Description |
|---|---|
| `Crop` | Name of the crop (e.g., Rice, Wheat, Arecanut) |
| `Crop_Year` | Year of record (dropped before training) |
| `Season` | Cropping season (e.g., Kharif, Rabi, Whole Year) |
| `State` | Indian state where the crop was grown |
| `Area` | Cultivated area (hectares) |
| `Production` | Total production (tonnes) |
| `Annual_Rainfall` | Annual rainfall (mm) |
| `Fertilizer` | Total fertilizer used (kg) |
| `Pesticide` | Total pesticide used (kg) |
| `Yield` | Target variable — yield (quintal/hectare) |

The dataset must be in the same directory as the script, named exactly `crop_yield.csv`.

## How It Works

1. On startup, `load_and_train()` loads the CSV and drops the `Crop_Year` column.
2. Categorical columns (`Crop`, `State`, `Season`) are label-encoded using `LabelEncoder`.
3. The data is split into training and test sets (70/30 split, `random_state=42`).
4. A `KNeighborsRegressor` (scikit-learn default settings) is trained on the encoded features.
5. User inputs from the sidebar/form are encoded with the same fitted encoders and passed to the model to generate a yield prediction.

## Requirements

- Python 3.8+
- streamlit
- pandas
- numpy
- scikit-learn

## Installation

```bash
pip install streamlit pandas numpy scikit-learn
```

## Usage

Make sure `crop_yield.py` and `crop_yield.csv` are in the same folder, then run:

```bash
streamlit run crop_yield.py
```

The app will open in your browser (usually at `http://localhost:8501`).

### Steps in the app

1. Select **Crop**, **State**, and **Season** from the dropdowns.
2. Enter **Area**, **Production**, **Annual Rainfall**, **Fertilizer**, and **Pesticide** values.
3. Click **Predict Crop Yield** to see the estimated yield in quintal/hectare.

## Project Structure

```
.
├── crop_yield.py       # Streamlit app + model training logic
├── crop_yield.csv      # Dataset used for training
└── README.md           # Project documentation
```

## Notes / Limitations

- The model is trained fresh each time the app cache is cleared — there is no pre-saved model file.
- KNN performance depends heavily on feature scaling; inputs here are not normalized/standardized before training, which may affect prediction accuracy.
- Unseen categorical values (crops/states/seasons not present in the training data) will raise an error, since `LabelEncoder.transform` only recognizes classes it was fit on.
- Predictions are only as reliable as the historical data — extreme or unrealistic input combinations may produce inaccurate yield estimates.

## License

Add your preferred license here (e.g., MIT).
