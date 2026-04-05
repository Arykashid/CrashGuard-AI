"""
Main Execution Script
Orchestrates the complete pipeline: data generation, preprocessing, 
training, evaluation, and real-time simulation.
"""

import os
import pandas as pd
import numpy as np
from data_generator import generate_cpu_data, add_missing_values
from preprocessing import prepare_data
from lstm_model import build_lstm_model, train_model
from evaluate import evaluate_model, plot_training_history
from realtime_simulation import simulate_realtime_prediction, plot_realtime_simulation, calculate_realtime_metrics


def main():
    """
    Main execution function.
    """
    print("="*70)
    print("Real-Time System Performance Forecasting using LSTM")
    print("="*70)
    
    # Create necessary directories
    os.makedirs('data', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    
    # Step 1: Generate or load data
    print("\n[Step 1] Generating synthetic CPU usage data...")
    data_file = 'data/cpu_usage_data.csv'
    
    if os.path.exists(data_file):
        print(f"Loading existing data from {data_file}")
        df = pd.read_csv(data_file)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
    else:
        print("Generating new synthetic data...")
        df = generate_cpu_data(
            num_samples=10000,
            start_date='2024-01-01',
            freq='1min',
            noise_level=0.1,
            trend=True
        )
        # Add missing values to simulate real-world data
        df = add_missing_values(df, missing_ratio=0.03)
        df.to_csv(data_file, index=False)
        print(f"Data saved to {data_file}")
    
    print(f"Data shape: {df.shape}")
    print(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    print(f"Missing values: {df['cpu_usage'].isna().sum()}")
    
    # Step 2: Preprocess data
    print("\n[Step 2] Preprocessing data...")
    window_size = 60  # Use 60 time steps (1 hour if 1-min intervals) as input
    forecast_horizon = 1  # Predict 1 step ahead
    
    processed_data = prepare_data(
        df,
        window_size=window_size,
        forecast_horizon=forecast_horizon,
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15
    )
    
    print(f"Training samples: {processed_data['X_train'].shape[0]}")
    print(f"Validation samples: {processed_data['X_val'].shape[0]}")
    print(f"Test samples: {processed_data['X_test'].shape[0]}")
    
    # Step 3: Build LSTM model
    print("\n[Step 3] Building LSTM model...")
    model = build_lstm_model(
        window_size=window_size,
        forecast_horizon=forecast_horizon,
        lstm_units=[64, 64, 32],  # Three LSTM layers
        dropout_rate=0.2
    )
    
    print("\nModel Architecture:")
    model.summary()
    
    # Step 4: Train model
    print("\n[Step 4] Training LSTM model...")
    history = train_model(
        model,
        processed_data['X_train'],
        processed_data['y_train'],
        processed_data['X_val'],
        processed_data['y_val'],
        epochs=50,
        batch_size=32,
        verbose=1,
        model_save_path='models/lstm_model.h5'
    )
    
    # Plot training history
    plot_training_history(history, save_path='results/training_history.png')
    
    # Step 5: Evaluate model
    print("\n[Step 5] Evaluating model on test set...")
    test_metrics = evaluate_model(
        model,
        processed_data['X_test'],
        processed_data['y_test'],
        processed_data['scaler'],
        plot=True,
        save_dir='results'
    )
    
    # Step 6: Real-time simulation
    print("\n[Step 6] Simulating real-time prediction...")
    
    # Use a portion of the test data for simulation
    simulation_start = len(processed_data['X_train']) + len(processed_data['X_val'])
    simulation_results = simulate_realtime_prediction(
        model=model,
        data=processed_data['original_data'],
        scaler=processed_data['scaler'],
        window_size=window_size,
        start_idx=simulation_start,
        num_predictions=200,
        update_frequency=1
    )
    
    # Calculate metrics for real-time simulation
    realtime_metrics = calculate_realtime_metrics(simulation_results)
    print(f"\nReal-Time Simulation Metrics:")
    print(f"RMSE: {realtime_metrics['rmse']:.4f}")
    print(f"MAE:  {realtime_metrics['mae']:.4f}")
    
    # Plot real-time simulation
    plot_realtime_simulation(simulation_results, save_path='results/realtime_simulation.png')
    
    # Final summary
    print("\n" + "="*70)
    print("PIPELINE COMPLETE")
    print("="*70)
    print("\nSummary:")
    print(f"  - Model saved to: models/lstm_model.h5")
    print(f"  - Test RMSE: {test_metrics['rmse']:.4f}")
    print(f"  - Test MAE:  {test_metrics['mae']:.4f}")
    print(f"  - Real-time RMSE: {realtime_metrics['rmse']:.4f}")
    print(f"  - Real-time MAE:  {realtime_metrics['mae']:.4f}")
    print(f"\nResults saved to 'results/' directory")
    print("="*70)


if __name__ == "__main__":
    main()
