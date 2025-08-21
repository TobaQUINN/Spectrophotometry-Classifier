# 🌈 Spectrophotometry Classifier

Machine Learning (Random Forest) applied to **Spectrophotometry** — a core concept in Biochemistry.

---

## 📌 Project Overview  

Spectrophotometry is used in the lab to determine macromolecules (Carbohydrates, Proteins, Lipids, etc.) by analyzing **absorbance values at specific wavelengths**. Each macromolecule absorbs light differently depending on its structure, creating unique absorption patterns.

**Examples:**
- Proteins → strong absorbance near **280 nm** (due to aromatic amino acids).  
- Carbohydrates → absorb mainly in the **190–210 nm** range.  
- Lipids → show distinct spectral patterns across **200–300 nm**.  

Traditionally, we identify macromolecules by comparing absorbance at these wavelengths.  
Here, the idea is: **Can Machine Learning automate this classification?**

- **Input** → Absorbance readings (200–700 nm, every 10 nm).  
- **Output** → Predicted macromolecule class (**Carbohydrate / Lipid / Protein**).  

---

## ⚙️ Approach  

### 🔹 Data Preparation
- Dataset → Synthetic absorbance dataset with **51 wavelengths (200–700 nm, 10 nm steps)**.  
- Target classes mapped:  
  - Carbohydrate → `0`  
  - Lipid → `1`  
  - Protein → `2`  
- Dropped `Sample_ID` (not a predictive feature).  

### 🔹 Model
- Algorithm → **Random Forest Classifier**  
- Reason → Handles high-dimensional data well, interpretable via feature importance, and robust to noise.  
- Training → Absorbance spectra used as features to classify macromolecule type.  

### 🔹 Feature Importance
- Random Forest highlights **which wavelengths matter most**.  
- Helps identify spectral regions similar to how biochemists interpret absorbance manually.  

### 🔹 Visualization
- Feature Importance Plot → Key wavelengths.  
- Confusion Matrix → Model performance across classes.  
- Train/Test Accuracy → Model learning curves.  

---

## 💡 Key Insights  

✅ Machine Learning can automate spectrophotometric classification.  
✅ Provides wavelength importance analysis.  
✅ Potential to reduce manual lab interpretation.  

**Future Work:**
- Expand dataset with more replicates.  
- Add noise → mimic real experimental errors.  
- Try deep learning for spectral pattern recognition.  
- Test classification using only **3–5 wavelengths**.  

---

## 🛠️ Tech Stack  

- **Language:** Python 3  
- **Libraries:**  
  - `pandas`, `numpy` → Data preprocessing  
  - `scikit-learn` → Random Forest & evaluation  
  - `matplotlib`, `seaborn` → Visualizations  

---

## 🚀 Usage  

###  Clone Repository  
```bash
git clone https://github.com/yourusername/Spectrophotometry-Classifier.git
cd Spectrophotometry-Classifier

