import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler

def preprocess_data(df):

    print("Starting data Preprocessing...")

    # List of Features
    print("Columns in the dataset:", df.columns.tolist())
    
    # Remove duplicates
    if df.duplicated().any():
        print("Warning: Duplicate rows found. Removing duplicates.")
        df.drop_duplicates(inplace=True)
    else:
        print("No duplicate rows found.")

    # Handle missing values
    if df.isnull().values.any():
        print("Missing values found. Filling missing values with median.") 
        df.fillna(df.median(numeric_only=True), inplace=True)
    else:
        print("No missing values found.")
    
    # Clean and encode 'Class' column
    print("Encoding 'Class' column with Label Encoding.")
    df['Class'] = df['Class'].str.strip().str.capitalize()
    print("Encoded classes")

    # create mapping for later visualization
    class_mapping = {cat: code for code, cat in enumerate(df['Class'].unique())}
    print("Class mapping:", class_mapping)

    df['Class'] = df['Class'].map(class_mapping)
    
    # Scale nm columns
    nm_columns = [col for col in df.columns if col.endswith('nm')]
    if nm_columns:
        print("Encoding nm columns with Min-Max Scaling.")
        scaler = MinMaxScaler()
        df[nm_columns] = scaler.fit_transform(df[nm_columns])
    print("Scaled nm columns")

    # Visualization: average spectra per class
    wavelengths = [int(col.replace("nm", "")) for col in nm_columns]
    for label_name, label_code in class_mapping.items():
        subset = df[df['Class'] == label_code][nm_columns]
        plt.plot(wavelengths, subset.mean(axis=0), label=label_name)
        plt.xlabel("Wavelength (nm)")
        plt.ylabel("Mean Absorbance")
        plt.title("Average Spectra by Macromolecule Class")
        plt.legend()
        plt.savefig(f"reports/average_spectra_class_{label_name}.png")
        plt.show()

    # Correlation heatmap
    corr = df[nm_columns].corr()
    plt.figure(figsize=(10,8))
    sns.heatmap(corr, cmap="coolwarm", center=0)
    plt.title("Correlation Between Wavelengths")
    plt.savefig("reports/correlation_heatmap.png")
    plt.show()

    print("Data Preprocessing Completed.")
    return df

df= pd.read_csv(r'C:\Users\DARA2\Documents\GitHub\Spectrophotometry-Classifier\data\spectrophotometer_readings_raw.csv') 
preprocess_data(df)