<<<<<<< HEAD
# Video Game Sales Analysis & Machine Learning Project

A comprehensive machine learning pipeline for analyzing video game sales data using clustering, classification, and neural networks. The project includes dimensionality reduction, automated clustering, traditional ML classifiers, and PyTorch-based deep learning models with full MLflow experiment tracking.

## 📊 Project Overview

This project analyzes the VGSales dataset to:
- Identify patterns and clusters in video game sales across different regions
- Predict game genres based on sales metrics and other features
- Compare traditional ML approaches with deep learning models
- Track experiments and visualizations using MLflow

## 🎯 Features

### Data Analysis & Visualization
- Sales distribution analysis by region (NA, EU, JP, Other, Global)
- Sales trends over time
- Platform and genre performance analysis
- Top games per cluster identification

### Machine Learning Pipeline

#### 1. **Dimensionality Reduction**
- **PCA (2D & 3D)**: Principal Component Analysis for feature reduction
- **t-SNE (2D)**: Non-linear dimensionality reduction for visualization

#### 2. **Clustering**
- **Automatic KMeans**: Elbow method with automatic optimal k detection
- Cluster analysis with heatmaps and distribution plots
- Top 10 games per cluster visualization

#### 3. **Classification Models**
- **Random Forest** with GridSearchCV hyperparameter tuning
- **SGDClassifier** with GridSearchCV optimization
- **Shallow Neural Network** (PyTorch): 3-layer architecture with dropout
- **Deep Neural Network** (PyTorch): 5-layer architecture with batch normalization

### Experiment Tracking
- Full MLflow integration for logging:
  - Model parameters and hyperparameters
  - Performance metrics (accuracy scores)
  - Trained models (scikit-learn and PyTorch)
  - Visualizations and plots

## 📁 Project Structure

```
ipnyb/
├── videogames_sales.ipynb    # Main Jupyter notebook
├── vgsales.csv               # Video game sales dataset
├── deep_nn_pytorch.pth       # Saved deep neural network model
├── shallow_nn_pytorch.pth    # Saved shallow neural network model
├── mlruns/                   # MLflow tracking data
│   ├── 0/                    # Default experiment
│   └── 761854136110779325/   # VideoGameSales_Project experiment
└── README.md                 # This file
```

## 🛠️ Technologies Used

### Core Libraries
- **pandas** & **numpy**: Data manipulation and analysis
- **matplotlib** & **seaborn**: Data visualization
- **scikit-learn**: Traditional ML algorithms and preprocessing
  - PCA, t-SNE for dimensionality reduction
  - KMeans for clustering
  - RandomForest, SGDClassifier for classification
  - GridSearchCV for hyperparameter tuning

### Deep Learning
- **PyTorch**: Neural network implementation
  - Custom shallow and deep NN architectures
  - GPU/CPU support
  - Early stopping and dropout regularization

### Experiment Tracking
- **MLflow**: Experiment tracking, model logging, and artifact management

## 🚀 Getting Started

### Prerequisites

```bash
pip install pandas numpy matplotlib seaborn scikit-learn torch mlflow
```

### Running the Project

1. **Launch MLflow UI** (optional but recommended):
   ```bash
   mlflow ui
   ```
   Navigate to `http://localhost:5000` to view experiments

2. **Run the Jupyter Notebook**:
   ```bash
   jupyter notebook videogames_sales.ipynb
   ```

3. Execute cells sequentially to:
   - Load and preprocess data
   - Perform clustering analysis
   - Train classification models
   - Build and evaluate neural networks
   - Generate visualizations

## 📈 Model Performance

The project implements multiple models for genre classification:

- **Random Forest**: Ensemble learning with GridSearchCV optimization
- **SGDClassifier**: Stochastic gradient descent with regularization
- **Shallow Neural Network**: 3-layer PyTorch model with dropout
- **Deep Neural Network**: 5-layer PyTorch model with batch normalization

All models are evaluated on the same test set and metrics are logged to MLflow for comparison.

## 🔍 Key Insights

### Clustering Analysis
- Automatic detection of optimal number of clusters using elbow method
- Cluster characterization based on regional sales patterns
- Top games identified for each cluster

### Feature Importance
- Random Forest feature importance visualization
- Identification of key predictors for genre classification

### Sales Patterns
- Regional sales distribution and trends
- Platform and genre performance over time
- Global sales patterns across different dimensions

## 📊 Visualizations

The project generates comprehensive visualizations including:
- Sales distribution histograms by region
- Sales trends over time (line plots)
- Platform and genre performance (bar charts)
- Cluster heatmaps and distributions
- PCA 2D/3D scatter plots
- Feature importance charts
- Neural network training history (loss and accuracy curves)
- Model comparison bar charts
- Box plots and violin plots by genre

All visualizations are automatically saved and logged to MLflow.

## 🔬 Experiment Tracking with MLflow

The project uses MLflow to track:
- **Parameters**: Model configurations, optimal k for clustering
- **Metrics**: Accuracy scores for all models
- **Artifacts**: 
  - Trained models (scikit-learn and PyTorch)
  - Visualizations (PNG files)
  - Training histories
- **Nested Runs**: Organized hierarchy for visualizations and neural networks

## 💾 Model Persistence

Trained models are saved in multiple formats:
- **scikit-learn models**: Logged via MLflow (RandomForest, SGDClassifier)
- **PyTorch models**: Saved as `.pth` files (shallow_nn_pytorch.pth, deep_nn_pytorch.pth)

## 🎓 Use Cases

This project demonstrates:
- End-to-end ML pipeline development
- Integration of traditional ML and deep learning
- Experiment tracking and reproducibility
- Data visualization best practices
- MLOps fundamentals with MLflow

## 📝 Dataset

The `vgsales.csv` dataset contains video game sales information with the following features:
- **Name**: Game title
- **Platform**: Gaming platform
- **Year**: Release year
- **Genre**: Game genre
- **Publisher**: Publishing company
- **NA_Sales**: North America sales (millions)
- **EU_Sales**: Europe sales (millions)
- **JP_Sales**: Japan sales (millions)
- **Other_Sales**: Other regions sales (millions)
- **Global_Sales**: Total worldwide sales (millions)

## 🤝 Contributing

This is an educational project demonstrating ML best practices. Feel free to fork and extend with:
- Additional ML algorithms
- Advanced feature engineering
- Hyperparameter optimization techniques
- Model ensemble methods
- Deployment pipelines

## 📄 License

This project is available for educational and research purposes.

## 🙏 Acknowledgments

- VGSales dataset for providing comprehensive video game sales data
- PyTorch and scikit-learn communities for excellent documentation
- MLflow for making experiment tracking seamless

---


**Last Updated**: February 2026
=======
# Video Game Sales Analysis & Machine Learning Project

A comprehensive machine learning pipeline for analyzing video game sales data using clustering, classification, and neural networks. The project includes dimensionality reduction, automated clustering, traditional ML classifiers, and PyTorch-based deep learning models with full MLflow experiment tracking.

## 📊 Project Overview

This project analyzes the VGSales dataset to:
- Identify patterns and clusters in video game sales across different regions
- Predict game genres based on sales metrics and other features
- Compare traditional ML approaches with deep learning models
- Track experiments and visualizations using MLflow

## 🎯 Features

### Data Analysis & Visualization
- Sales distribution analysis by region (NA, EU, JP, Other, Global)
- Sales trends over time
- Platform and genre performance analysis
- Top games per cluster identification

### Machine Learning Pipeline

#### 1. **Dimensionality Reduction**
- **PCA (2D & 3D)**: Principal Component Analysis for feature reduction
- **t-SNE (2D)**: Non-linear dimensionality reduction for visualization

#### 2. **Clustering**
- **Automatic KMeans**: Elbow method with automatic optimal k detection
- Cluster analysis with heatmaps and distribution plots
- Top 10 games per cluster visualization

#### 3. **Classification Models**
- **Random Forest** with GridSearchCV hyperparameter tuning
- **SGDClassifier** with GridSearchCV optimization
- **Shallow Neural Network** (PyTorch): 3-layer architecture with dropout
- **Deep Neural Network** (PyTorch): 5-layer architecture with batch normalization

### Experiment Tracking
- Full MLflow integration for logging:
  - Model parameters and hyperparameters
  - Performance metrics (accuracy scores)
  - Trained models (scikit-learn and PyTorch)
  - Visualizations and plots

## 📁 Project Structure

```
ipnyb/
├── videogames_sales.ipynb    # Main Jupyter notebook
├── vgsales.csv               # Video game sales dataset
├── deep_nn_pytorch.pth       # Saved deep neural network model
├── shallow_nn_pytorch.pth    # Saved shallow neural network model
├── mlruns/                   # MLflow tracking data
│   ├── 0/                    # Default experiment
│   └── 761854136110779325/   # VideoGameSales_Project experiment
└── README.md                 # This file
```

## 🛠️ Technologies Used

### Core Libraries
- **pandas** & **numpy**: Data manipulation and analysis
- **matplotlib** & **seaborn**: Data visualization
- **scikit-learn**: Traditional ML algorithms and preprocessing
  - PCA, t-SNE for dimensionality reduction
  - KMeans for clustering
  - RandomForest, SGDClassifier for classification
  - GridSearchCV for hyperparameter tuning

### Deep Learning
- **PyTorch**: Neural network implementation
  - Custom shallow and deep NN architectures
  - GPU/CPU support
  - Early stopping and dropout regularization

### Experiment Tracking
- **MLflow**: Experiment tracking, model logging, and artifact management

## 🚀 Getting Started

### Prerequisites

```bash
pip install pandas numpy matplotlib seaborn scikit-learn torch mlflow
```

### Running the Project

1. **Launch MLflow UI** (optional but recommended):
   ```bash
   mlflow ui
   ```
   Navigate to `http://localhost:5000` to view experiments

2. **Run the Jupyter Notebook**:
   ```bash
   jupyter notebook videogames_sales.ipynb
   ```

3. Execute cells sequentially to:
   - Load and preprocess data
   - Perform clustering analysis
   - Train classification models
   - Build and evaluate neural networks
   - Generate visualizations

## 📈 Model Performance

The project implements multiple models for genre classification:

- **Random Forest**: Ensemble learning with GridSearchCV optimization
- **SGDClassifier**: Stochastic gradient descent with regularization
- **Shallow Neural Network**: 3-layer PyTorch model with dropout
- **Deep Neural Network**: 5-layer PyTorch model with batch normalization

All models are evaluated on the same test set and metrics are logged to MLflow for comparison.

## 🔍 Key Insights

### Clustering Analysis
- Automatic detection of optimal number of clusters using elbow method
- Cluster characterization based on regional sales patterns
- Top games identified for each cluster

### Feature Importance
- Random Forest feature importance visualization
- Identification of key predictors for genre classification

### Sales Patterns
- Regional sales distribution and trends
- Platform and genre performance over time
- Global sales patterns across different dimensions

## 📊 Visualizations

The project generates comprehensive visualizations including:
- Sales distribution histograms by region
- Sales trends over time (line plots)
- Platform and genre performance (bar charts)
- Cluster heatmaps and distributions
- PCA 2D/3D scatter plots
- Feature importance charts
- Neural network training history (loss and accuracy curves)
- Model comparison bar charts
- Box plots and violin plots by genre

All visualizations are automatically saved and logged to MLflow.

## 🔬 Experiment Tracking with MLflow

The project uses MLflow to track:
- **Parameters**: Model configurations, optimal k for clustering
- **Metrics**: Accuracy scores for all models
- **Artifacts**: 
  - Trained models (scikit-learn and PyTorch)
  - Visualizations (PNG files)
  - Training histories
- **Nested Runs**: Organized hierarchy for visualizations and neural networks

## 💾 Model Persistence

Trained models are saved in multiple formats:
- **scikit-learn models**: Logged via MLflow (RandomForest, SGDClassifier)
- **PyTorch models**: Saved as `.pth` files (shallow_nn_pytorch.pth, deep_nn_pytorch.pth)

## 🎓 Use Cases

This project demonstrates:
- End-to-end ML pipeline development
- Integration of traditional ML and deep learning
- Experiment tracking and reproducibility
- Data visualization best practices
- MLOps fundamentals with MLflow

## 📝 Dataset

The `vgsales.csv` dataset contains video game sales information with the following features:
- **Name**: Game title
- **Platform**: Gaming platform
- **Year**: Release year
- **Genre**: Game genre
- **Publisher**: Publishing company
- **NA_Sales**: North America sales (millions)
- **EU_Sales**: Europe sales (millions)
- **JP_Sales**: Japan sales (millions)
- **Other_Sales**: Other regions sales (millions)
- **Global_Sales**: Total worldwide sales (millions)

## 🤝 Contributing

This is an educational project demonstrating ML best practices. Feel free to fork and extend with:
- Additional ML algorithms
- Advanced feature engineering
- Hyperparameter optimization techniques
- Model ensemble methods
- Deployment pipelines

## 📄 License

This project is available for educational and research purposes.

## 🙏 Acknowledgments

- VGSales dataset for providing comprehensive video game sales data
- PyTorch and scikit-learn communities for excellent documentation
- MLflow for making experiment tracking seamless

---
**Last Updated**: February 2026

>>>>>>> 9c99c11efc5a354aeddbb8b67fe486fccd38f5fd
