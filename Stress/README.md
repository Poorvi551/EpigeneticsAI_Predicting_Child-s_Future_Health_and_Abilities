# Child Stress Prediction Tool

A comprehensive machine learning application that predicts whether a child is experiencing stress based on psychological, physical, environmental, and social factors.

## 🚀 Features

- **Interactive Web Interface**: User-friendly Streamlit web app
- **Multiple ML Models**: Random Forest, XGBoost, SVM, Neural Network, and more
- **Binary Classification**: Stressed vs Not Stressed prediction
- **Comprehensive Assessment**: 20 different factors analyzed
- **Real-time Predictions**: Instant results with confidence scores
- **Actionable Recommendations**: Personalized advice based on predictions
- **Professional UI**: Clean, modern interface with responsive design

## 📊 Model Performance

- **Accuracy**: 85-95% (varies by model)
- **Features**: 20 psychological, physical, environmental, and social factors
- **Training Data**: 1,100 samples
- **Model Types**: Binary classification (Stressed/Not Stressed)

## 🛠️ Installation

### Prerequisites
- Python 3.8 or higher
- Virtual environment (recommended)

### Setup Instructions

1. **Clone or download the project files**

2. **Create and activate virtual environment**:
   ```bash
   python -m venv venv
   venv\Scripts\activate  # Windows
   # or
   source venv/bin/activate  # macOS/Linux
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Train the model** (if not already done):
   - Open `stress.ipynb` in Jupyter
   - Run all cells to train the model
   - This will create the model files (`best_binary_stress_model.pkl`, `binary_scaler.pkl`)

5. **Run the Streamlit app**:
   ```bash
   streamlit run app.py
   ```

6. **Open your browser** and go to `http://localhost:8501`

## 📋 Usage

### Web Interface (Streamlit App)

1. **Launch the app**: `streamlit run app.py`
2. **Fill the assessment form** with the child's information:
   - **Psychological Factors**: Anxiety, self-esteem, depression, mental health history
   - **Physical Factors**: Headache, blood pressure, sleep quality, breathing problems
   - **Environmental Factors**: Noise level, living conditions, safety, basic needs
   - **Academic Factors**: Performance, study load, teacher relationship, career concerns
   - **Social Factors**: Social support, peer pressure, activities, bullying
3. **Click "Predict Stress Level"** to get results
4. **Review the prediction** and recommendations

### Jupyter Notebook

1. **Open `stress.ipynb`**
2. **Run all cells** to train models and see analysis
3. **Use interactive functions**:
   ```python
   run_stress_assessment()  # Complete interactive assessment
   quick_assessment()       # Test with sample data
   ```

## 📁 Project Structure

```
Stress/
├── app.py                          # Streamlit web application
├── stress.ipynb                    # Jupyter notebook with ML models
├── StressLevelDataset.csv          # Training dataset
├── requirements.txt                # Python dependencies
├── README.md                       # This file
├── venv/                          # Virtual environment
├── best_binary_stress_model.pkl   # Trained binary model (created after training)
├── binary_scaler.pkl              # Feature scaler (created after training)
├── best_stress_model.pkl          # Trained multi-class model (created after training)
└── scaler.pkl                     # Feature scaler (created after training)
```

## 🧠 Model Details

### Features Analyzed (20 total)

**Psychological Factors:**
- Anxiety Level (0-30)
- Self Esteem (0-30)
- Mental Health History (Yes/No)
- Depression Level (0-30)

**Physical Factors:**
- Headache Frequency (0-5)
- Blood Pressure Issues (0-5)
- Sleep Quality (0-5)
- Breathing Problems (0-5)

**Environmental Factors:**
- Noise Level at Home (0-5)
- Living Conditions Quality (0-5)
- Safety Level (0-5)
- Basic Needs Met (0-5)

**Academic Factors:**
- Academic Performance (0-5)
- Study Load (0-5)
- Teacher-Student Relationship (0-5)
- Future Career Concerns (0-5)

**Social Factors:**
- Social Support (0-5)
- Peer Pressure (0-5)
- Extracurricular Activities (0-5)
- Bullying Experience (0-5)

### Model Types

1. **Binary Classification**: Stressed (1) vs Not Stressed (0)
2. **Multi-class Classification**: Low (0), Medium (1), High (2) stress levels

## 🎯 Results Interpretation

### Binary Classification Results
- **Not Stressed (0)**: Child appears to be coping well
- **Stressed (1)**: Child shows signs of stress and may need support

### Confidence Levels
- **High (>80%)**: Very confident prediction
- **Medium (60-80%)**: Moderately confident prediction
- **Low (<60%)**: Less confident prediction

## 💡 Recommendations

The app provides personalized recommendations based on the prediction:

**For Stressed Children:**
- Consider professional consultation
- Monitor behavior closely
- Provide emotional support
- Create safe environment
- Encourage open communication

**For Non-Stressed Children:**
- Continue monitoring
- Maintain supportive environment
- Encourage healthy coping mechanisms
- Regular check-ins

## ⚠️ Important Disclaimer

This tool is designed for **educational and preliminary assessment purposes only**. It should **NOT** replace professional medical, psychological, or psychiatric evaluation. Always consult with qualified healthcare professionals for proper diagnosis and treatment.

## 🔧 Troubleshooting

### Common Issues

1. **Model files not found**:
   - Make sure you've run the Jupyter notebook to train the models
   - Check that `.pkl` files are in the project directory

2. **Streamlit not starting**:
   - Ensure virtual environment is activated
   - Check that all dependencies are installed: `pip install -r requirements.txt`

3. **Import errors**:
   - Verify all packages are installed correctly
   - Try reinstalling: `pip install --upgrade -r requirements.txt`

### Getting Help

If you encounter issues:
1. Check the error messages in the terminal/console
2. Ensure all dependencies are installed
3. Verify the model files exist
4. Make sure you're using the correct Python version

## 📈 Future Enhancements

- [ ] Add more sophisticated visualization
- [ ] Include historical tracking
- [ ] Add export functionality for reports
- [ ] Implement user authentication
- [ ] Add batch processing capabilities
- [ ] Include more detailed analytics

## 🤝 Contributing

Feel free to contribute to this project by:
- Reporting bugs
- Suggesting new features
- Improving documentation
- Adding new models or features

## 📄 License

This project is for educational purposes. Please ensure proper attribution if used in academic or professional settings.

---

**Remember**: This tool is a preliminary assessment tool and should not replace professional medical or psychological evaluation.
