# 📚 Transformers in NLP: RoBERTa and XLNet

## 🎯 Business Objective

In Part 1 of our Transformer series - Multi-Class Text Classification with Deep Learning using BERT, we explored the evolution of NLP models, from simpler models like Bag of Words (BOW) and TF-IDF to advanced Transformer architectures like BERT. 

In Part 2, we dive into two novel architectures that enhance BERT's performance through innovative training and optimization techniques:
- **RoBERTa**: A Robustly Optimized BERT Pretraining Approach
- **XLNet**: Generalized Autoregressive Pretraining for Language Understanding

We'll analyze these models, explore their training methods, and use them to classify human emotions from text data.

## 📄 Data Description

We use the **Emotion** dataset from the Hugging Face library, which consists of English Twitter messages labeled with six basic emotions: anger, fear, joy, love, sadness, and surprise. 

### Dataset Breakdown:
- **Train**: 16,000 rows
- **Validation**: 2,000 rows
- **Test**: 2,000 rows

### Labels:
- 0: sadness
- 1: joy
- 2: love
- 3: anger
- 4: fear
- 5: surprise

## 🛠️ Tech Stack

- **Language**: Python
- **Libraries**: `datasets`, `numpy`, `pandas`, `matplotlib`, `seaborn`, `ktrain`, `transformers`, `tensorflow`, `sklearn`
- **Environment**: Jupyter Notebook, Google Colab Pro (Recommended)

## 🚀 Approach

1. **Install Libraries**: Ensure all necessary libraries are installed.
2. **Load Dataset**: Load and explore the Emotion dataset.
3. **Data Preprocessing**: Convert datasets to DataFrame and create additional features.
4. **Data Visualization**: Use histograms to visualize data distribution.
5. **Model Training**:
    - **RoBERTa**:
        - Create and configure the model.
        - Preprocess data, compile the model, and find optimal learning rates.
        - Fine-tune the model and evaluate its performance.
        - Save and test the model.
    - **XLNet**:
        - Similar steps as RoBERTa with additional understanding of Autoregressive and Autoencoder models.
6. **Performance Evaluation**: Evaluate both models on test data and compare their metrics.

## 📂 Project Structure

### Modular Code

- **src**: Contains modularized code for the entire project.
  - **Engine.py**: Main script to run the project.
  - **ML_Pipeline**: Folder with functions for data processing and model training.
- **output**: Contains trained models for easy loading and reuse.
- **lib**: Contains Jupyter notebooks and reference materials.

## 📝 Project Takeaways

1. Understand business problems in NLP.
2. Explore Transformer architectures and self-attention mechanisms.
3. Gain insights into RoBERTa and XLNet models.
4. Learn data preprocessing and visualization techniques.
5. Develop and fine-tune Transformer models.
6. Compare and evaluate model performances.

## 📦 Setup Instructions

### Prerequisites

- Ensure `git` is installed on your machine.

### Installation

1. **Clone the repo**
   ```sh
   git clone https://github.com/Vidhi1290/Text-Classification-with-Transformers-RoBERTa-and-XLNet-Model.git
   ```
2. **Navigate to the project directory**
   ```sh
   cd Text-Classification-with-Transformers-RoBERTa-and-XLNet-Model
   ```
3. **Install dependencies**
   ```sh
   pip install -r modular_code/requirements.txt
   ```

4. **Run the project**
   ```sh
   python modular_code/src/Engine.py
   ```

## Follow me on:

Follow me on:
- **[LinkedIn](https://www.linkedin.com/in/vidhi-waghela-434663198/)**
- **[Kaggle](https://www.kaggle.com/vidhikishorwaghela)**
- **[Medium](https://medium.com/@datasciencemeetscybersecurity)**
- **[GitHub](https://github.com/Vidhi1290)**

Feel free to reach out for any questions or collaboration opportunities!
