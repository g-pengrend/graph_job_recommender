# Job Recommendation System

A graph-based job recommendation system using machine learning and network analysis.

## Prerequisites

Before you begin, ensure you have Python 3.8+ installed on your system.

## Installation

### 1. Clone the Repository 

### 2. Set Up Virtual Environment

#### Windows
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
.\venv\Scripts\activate
```

#### macOS/Linux
```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

## System-Specific Setup

### Windows
1. Install [Visual C++ Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/)
2. Install Microsoft Visual C++ Redistributable

### macOS
1. Install Xcode Command Line Tools:
```bash
xcode-select --install
```

#### Apple Silicon (M1/M2) Users
- Some packages may require Rosetta 2
- Use ARM-optimized versions of packages when available

### Linux
```bash
sudo apt-get update
sudo apt-get install python3-dev build-essential
```

## GPU Support (Optional)

If you have a CUDA-capable GPU:

1. Replace `faiss-cpu` with `faiss-gpu` in requirements.txt
2. Install the CUDA version of PyTorch:
   - Visit [PyTorch's website](https://pytorch.org/get-started/locally/)
   - Select your specific configuration
   - Use the provided installation command

## Troubleshooting

### Common Issues

#### FAISS Installation Issues
- **Windows**: Ensure Visual C++ Build Tools are installed
- **macOS**: Install Xcode Command Line Tools
- **Linux**: Install build-essential package

#### Package Conflicts
1. Clear existing packages:
```bash
pip uninstall -r requirements.txt -y
```
2. Reinstall requirements:
```bash
pip install -r requirements.txt
```

## Usage

1. Activate your virtual environment (if not already activated)
2. Run the recommendation system:
```python
python recommendation_system.py ### not done this way yet ###
```

## Dependencies

Key packages and their versions:

- pandas >= 1.5.0
- numpy >= 1.21.0
- scipy >= 1.7.0
- scikit-learn >= 1.0.0
- torch >= 2.0.0
- transformers >= 4.30.0
- faiss-cpu >= 1.7.0
- annoy >= 1.17.0
- networkx >= 2.8.0
- geopy >= 2.3.0

For a complete list of dependencies, see `requirements.txt`

## License

[Your chosen license]

## Contributing

[Your contribution guidelines]

## Authors

[Your name/team]

## Acknowledgments

[Any acknowledgments]
```

This README provides a comprehensive guide for users across different operating systems. You should customize the sections marked with brackets `[]` according to your specific project details, and add any additional information that might be relevant to your specific implementation.