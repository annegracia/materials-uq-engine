# =====================================================================
# STEP 1: INSTALL AND IMPORT MODERN DEPENDENCIES
# =====================================================================
!pip install torch numpy pandas plotly scikit-learn

import os
import time
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from dataclasses import dataclass
from typing import List, Tuple
import plotly.graph_objects as go
from google.colab import drive

# =====================================================================
# STEP 2: DEFINE THE QRNN DEEP LEARNING ARCHITECTURE
# =====================================================================
class QuantileRegressionNN(nn.Module):
    """
    A Neural Network designed with 3 output nodes to simultaneously 
    calculate the Floor (2.5%), Middle (50%), and Ceiling (97.5%) 
    boundaries of your dataset.
    """
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 3) # Outputs: [Lower Bound, Median, Upper Bound]
        )
        
    def forward(self, x):
        return self.net(x)

def pinball_loss(preds: torch.Tensor, targets: torch.Tensor, quantiles: List[float] = [0.025, 0.50, 0.975]) -> torch.Tensor:
    """
    The mathematical penalty game. Punishes the network severely if it 
    steps outside its designated percentage boundaries.
    """
    losses = []
    for i, q in enumerate(quantiles):
        error = targets - preds[:, i:i+1]
        # The core asymmetric penalty rule
        loss = torch.max((q - 1) * error, q * error)
        losses.append(loss.mean())
    return sum(losses)

# =====================================================================
# STEP 3: CLEAN DATA PROCESSOR DEFINITIONS
# =====================================================================
@dataclass
class QRNNResult:
    column_name: str
    total_records: int
    predicted_median: float
    lower_bound_95: float
    upper_bound_95: float
    spread: float # Added spread field
    training_time_ms: float

class UQDataProcessor:
    def __init__(self, file_path: str):
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Could not locate the file at: {file_path}")
        self.df = pd.read_csv(file_path)
        
    def discover_numerical_columns(self) -> List[str]:
        """Scans the CSV file to find columns that contain valid numbers."""
        return self.df.select_dtypes(include=[np.number]).columns.tolist()

    def extract_target_data(self, column_name: str) -> np.ndarray:
        """Extracts clean, non-null values from the selected column."""
        return self.df[column_name].dropna().values

# =====================================================================
# STEP 4: THE MODEL TRAINING PIPELINE ENGINE
# =====================================================================
def run_qrnn_uncertainty_quantification(data: np.ndarray, column_name: str) -> QRNNResult:
    t0 = time.perf_counter()
    
    # Calculate empirical baseline stats
    mean_val = np.mean(data)
    std_val = np.std(data) if np.std(data) > 0 else 0.1
    
    # Generate an index grid acting as our synthetic X input for the AI spatial network
    X_train = np.linspace(0, 1, len(data)).reshape(-1, 1)
    Y_train = data.reshape(-1, 1)
    
    # Convert data into PyTorch Tensors
    X_tensor = torch.tensor(X_train, dtype=torch.float32)
    Y_tensor = torch.tensor(Y_train, dtype=torch.float32)
    
    # Instantiate Model & Optimizer
    model = QuantileRegressionNN()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    
    # Train the AI across 250 epochs (penalty training cycles)
    model.train()
    for epoch in range(250):
        optimizer.zero_grad()
        predictions = model(X_tensor)
        loss = pinball_loss(predictions, Y_tensor)
        loss.backward()
        optimizer.step()
        
    # Switch to Evaluation mode to extract our boundaries
    model.eval()
    with torch.no_grad():
        final_preds = model(X_tensor).numpy()
        
    # Compile results using the mean center anchor point
    mid_index = len(data) // 2
    dt = (time.perf_counter() - t0) * 1000
    
    # Calculate spread
    lower_bound = float(np.mean(final_preds[:, 0]))
    upper_bound = float(np.mean(final_preds[:, 2]))
    spread_value = upper_bound - lower_bound

    return QRNNResult(
        column_name=column_name,
        total_records=len(data),
        predicted_median=float(np.mean(final_preds[:, 1])),
        lower_bound_95=lower_bound,
        upper_bound_95=upper_bound,
        spread=spread_value, # Assign spread value
        training_time_ms=dt
    ), final_preds, Y_train.flatten()

# =====================================================================
# STEP 5: INTERACTIVE PLOTLY GRAPH GENERATOR
# =====================================================================
def generate_interactive_chart(raw_data: np.ndarray, predictions: np.ndarray, result: QRNNResult):
    """Generates an advanced, high-fidelity interactive visualization."""
    x_axis = np.arange(len(raw_data))
    
    fig = go.Figure()
    
    # 1. Add Raw CSV Data Points
    fig.add_trace(go.Scatter(
        x=x_axis, y=raw_data,
        mode='markers',
        name='Raw Measurements',
        marker=dict(color='#34495E', size=7, opacity=0.7)
    ))
    
    # 2. Add AI Predicted Median (Middle Line)
    fig.add_trace(go.Scatter(
        x=x_axis, y=predictions[:, 1],
        mode='lines',
        name='AI Predicted Median (50%)',
        line=dict(color='#2ECC71', width=3)
    ))
    
    # 3. Add AI Predicted Upper Bound (Ceiling Line)
    fig.add_trace(go.Scatter(
        x=x_axis, y=predictions[:, 2],
        mode='lines',
        name='AI Upper Bound (97.5%)',
        line=dict(color='#E74C3C', width=2, dash='dash')
    ))
    
    # 4. Add AI Predicted Lower Bound (Floor Line)
    fig.add_trace(go.Scatter(
        x=x_axis, y=predictions[:, 0],
        mode='lines',
        name='AI Lower Bound (2.5%)',
        line=dict(color='#3498DB', width=2, dash='dash')
    ))
    
    # Layout and Customization
    fig.update_layout(
        title=dict(
            text=f"Deep Learning QRNN Uncertainty Profile for column: <b>{result.column_name}</b>",
            font=dict(size=16)
        ),
        xaxis_title="Measurement Record Index",
        yaxis_title="Data Scale Range Value",
        template="plotly_white",
        hovermode="x unified",
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
    )
    
    fig.show()

# =====================================================================
# STEP 6: TEST RUNNER EXECUTION ENVIRONMENT
# =====================================================================

# A. Mount your Google Drive
print("-> Please authorize Google Drive mounting to locate your CSV file:")
drive.mount('/content/drive')

# B. INPUT YOUR FILE PATH HERE
# Example template path: '/content/drive/MyDrive/Folder/your_file.csv'
CSV_FILE_PATH = '/content/drive/MyDrive/FSS_PCE_test/csv_test.txt' 

if __name__ == "__main__":
    # Check if user updated the default path template
    if not os.path.exists(CSV_FILE_PATH):
        print(f"\n[!] Error: File not found at '{CSV_FILE_PATH}'.")
        print("Please check your file name or upload your file directly to Colab and change the path string.")
    else:
        # Load up data processor
        processor = UQDataProcessor(CSV_FILE_PATH)
        numerical_cols = processor.discover_numerical_columns()
        
        if not numerical_cols:
            print("[!] Critical Error: No numerical data found inside this file.")
        else:
            # DEFAULT LOGIC: Pick the last numerical column automatically
            default_target_column = numerical_cols[-1]
            
            print("\n" + "="*60)
            print("                CSV DATA DISCOVERY MANAGER")
            print("="*60)
            print(f"Available numerical features found: {numerical_cols}")
            print(f"-> System automatically defaulted to last column: '{default_target_column}'")
            print("="*60)
            
            # --- INTERACTIVE USER CHANGE OPTION ---
            # To inspect a different column instead of the default last one, 
            # simply uncomment the line below and type the column name:
            # default_target_column = "Your_Other_Column_Name"
            
            # Extract clean arrays
            target_data = processor.extract_target_data(default_target_column)
            
            print(f"\nInitiating Deep Learning Quantile Optimization for '{default_target_column}'...")
            print("-> Training neural network pinball gradients across hidden layers...")
            
            # Run calculations
            uq_summary, model_bounds, raw_vector = run_qrnn_uncertainty_quantification(target_data, default_target_column)
            
            # Output Clean Summary Card
            print("\n" + "═"*60)
            print("             DEEP LEARNING QRNN METRICS SUMMARY CARD")
            print("═"*60)
            print(f" Target Feature Column Analyzed : {uq_summary.column_name}")
            print(f" Dataset Population Evaluated   : {uq_summary.total_records} rows")
            print(f" AI Optimized Symmetrical Median: {uq_summary.predicted_median:.5f} \u00B1 {uq_summary.spread/2:.5f}") # Changed format
            print(f" Deep Learning Floor (2.5%)     : {uq_summary.lower_bound_95:.5f}")
            print(f" Deep Learning Ceiling (97.5%)  : {uq_summary.upper_bound_95:.5f}")
            print(f" Deep Learning Spread           : {uq_summary.spread:.5f}") # Added spread print
            print(f" Complete 95% Safety Envelope   : [{uq_summary.lower_bound_95:.4f}, {uq_summary.upper_bound_95:.4f}]")
            print(f" Execution Compute Latency Time : {uq_summary.training_time_ms:.2f} ms")
            print("═"*60)
            
            # Display interactive charts
            generate_interactive_chart(raw_vector, model_bounds, uq_summary)
