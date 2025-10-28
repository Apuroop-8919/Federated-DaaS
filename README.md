# Federated-DaaS
Privacy Preserving Federated Learning DaaS platform for demo purpose

# Privacy-Preserving Federated Learning DaaS

## What is this?
Collaborative AI platform prototype for SmartIDEAthon 2025, demonstrating privacy-preserving federated learning. Institutions (e.g., hospitals, banks, colleges) train models locally—only the weights (not the data) are shared and aggregated.

## Features
- **No raw data leaves the institution** (data privacy by design)
- **Model weights aggregated** using Federated Averaging (FedAvg)
- **Improves accuracy across multiple nodes**
- Works for healthcare, finance, education, sports datasets

## Demo Results
- Model accuracy improves from ~62% → ~85% over 5 federated rounds (assumption) 
- Output chart: `federated_learning_results.png`

## How to Run
1. Clone repo:
    ```
    git clone https://github.com/Apuroop-8919/Federated-DaaS.git
    cd Federated-DaaS
    ```
2. (Optional) Create Python environment, install dependencies:
    ```
    python3 -m venv env
    source env/bin/activate
    pip install -r requirements.txt  # or: pip install flwr torch scikit-learn numpy matplotlib
    ```
3. Run the demo:
    ```
    python federated_learning_demo.py
    ```

## Technologies Used
- PyTorch
- Federated Learning (FedAvg)
- Scikit-learn
- Numpy
- Matplotlib


## License
See [LICENSE](LICENSE).
