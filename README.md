# Indian Flag Validator 🇮🇳
Smart validator for Indian Flag images – aspect ratio, stripe proportions &amp; Chakra validation.

## 📌 Overview
This project validates digital images of the Indian National Flag according to official specifications:
- **Aspect Ratio**: 3:2 (±1%)
- **Colors**: Saffron, White, Green, Navy Blue (±5% tolerance)
- **Stripe Proportions**: Each ~1/3 of the height
- **Ashoka Chakra**:
  - Centered in the white band (±1% tolerance)
  - Diameter = 3/4 of the white band height (±5%)
  - Exactly 24 spokes

Validation is automated using **Python + Colab + PIL + NumPy**.

---

## 📂 Files
- `Indian_Flag_Validator_Submission.ipynb` → Main Jupyter/Colab notebook  
- `validator.py` → Core validation logic  
- `sample_images.zip` → Example flag images for testing  

---

## ▶️ Run in Google Colab
Click below to open the project directly in Google Colab:

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/slavebotallen/IndianFlagValidator/blob/main/Indian_Flag_Validator_Submission.ipynb)

---

## 🚀 Usage
Inside Colab:

```python
from validator import validate

# Example: validate one of the sample images
report = validate("sample_images/flag1.png")
print(report)


The output is a JSON Report:
{
  "aspect_ratio": {"status": "pass", "actual": 1.5},
  "colors": {"saffron": {...}, "white": {...}, "green": {...}, "chakra_blue": {...}},
  "stripe_proportion": {...},
  "chakra_position": {...},
  "chakra_spokes": {...}
}
