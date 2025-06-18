
# Smart Crop Suitability Prediction

This project utilizes Earth Engine and various datasets (MODIS, CHIRPS, SoilGrids) to collect geospatial data and predict the suitability of certain crops across different regions and seasons in India. The data is collected, processed, and combined into a single dataset for potential use in machine learning models.

## Project Structure

*   **Notebook:** `smart_crop_suitability.ipynb`
*   **Data:**
    *   Raw data (collected from Earth Engine)
    *   Processed and combined data: `SMART_CROP_SUITABILITY_DATA.csv`
    *   Final data with suitability labels: `C.S_multi_lable_5_class.csv`
*   **README.md:** This file.

## Setup and Prerequisites

1.  **Google Colab:** This project is designed to be run in Google Colab.
2.  **Earth Engine Account:** You need an Earth Engine account and project.
3.  **Google Drive:** The project saves data to your Google Drive. Ensure you have sufficient space and the correct folder structure.

## Steps to Reproduce

Follow the steps below in your Google Colab notebook:

### 1. Install Dependencies

Install the necessary Python libraries:

### 2. Import Libraries and Authenticate

Import the required libraries and authenticate with your Earth Engine account

## Data Description

The final dataset (`C.S_multi_lable_5_class.csv`) includes the following columns:

*   `Sample_ID`: Unique identifier for each sampled point.
*   `State`: State where the point is located.
*   `Season`: Season of the year.
*   `Year`: Year of data collection.
*   `NDVI`: Normalized Difference Vegetation Index (0-1 range).
*   `Seasonal_Rainfall`: Total rainfall for the specific season.
*   `Annual_Rainfall`: Total rainfall for the entire year.
*   `Temperature`: Average temperature during the season (Celsius).
*   `pH_0_5`: Soil pH at 0-5 cm depth (0-14 scale).
*   `Carbon`: Soil organic carbon at 0-5 cm depth (g/kg).
*   `Texture`: Soil texture class.
*   `geometry`: Geospatial coordinates of the sampled point.
*   `Paddy`: Binary label indicating suitability for Paddy (True/False).
*   `Wheat`: Binary label indicating suitability for Wheat (True/False).
*   `Apple`: Binary label indicating suitability for Apple (True/False).
*   `Tea`: Binary label indicating suitability for Tea (True/False).
*   `Coconut`: Binary label indicating suitability for Coconut (True/False).

## Further Steps

This project provides the foundation for predicting crop suitability. The generated dataset can be used for:

*   **Exploratory Data Analysis:** Visualize the relationships between environmental factors and crop suitability.
*   **Machine Learning Models:** Train classification models to predict crop suitability based on the collected features.
*   **Suitability Mapping:** Visualize the predicted suitability across different regions.

## Credits

*   **Earth Engine:** For providing access to a wide range of geospatial datasets and computational capabilities.
*   **Google Colab:** For providing a free and accessible platform for running the code.
