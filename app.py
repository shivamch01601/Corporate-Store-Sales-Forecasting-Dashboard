import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import pickle
import os
import matplotlib.pyplot as plt

# Streamlit application layout definition
st.set_page_config(page_title="Data Scientist Assignment - LSTM Dashboard", layout="wide")

st.title("Corporate Store Sales Forecasting Dashboard")
st.markdown("### Production Evaluation System | Deep Learning Architecture Designed by: Shivam Chaudhary")

# --- 1. DEEP FORECASTER FRAMEWORK CLASS STRUCT ---
class MultiOutputLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=2):
        super(MultiOutputLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        return self.fc(out[:, -1, :])

# --- 2. LOGICAL FEATURE INFERENCE PIPELINE ---
def dynamic_feature_generator(df):
    df = df.copy()
    df['DayOfWeek'] = df['Date'].dt.dayofweek
    df['IsWeekend'] = df['DayOfWeek'].isin([5, 6]).astype(int)
    df['Month'] = df['Date'].dt.month
    
    df['Sin_Month'] = np.sin(2 * np.pi * df['Month'] / 12)
    df['Cos_Month'] = np.cos(2 * np.pi * df['Month'] / 12)
    
    df['IsHoliday'] = df['Date'].apply(
        lambda x: 1 if (x.month == 1 and x.day == 26) or 
                       (x.month == 8 and x.day == 15) or 
                       (x.month == 10 and x.day == 2) or 
                       (x.month == 12 and x.day == 25) else 0
    )
    
    delhi_temp_profile = {1: 15, 2: 19, 3: 25, 4: 31, 5: 34, 6: 35, 7: 31, 8: 30, 9: 30, 10: 28, 11: 20, 12: 15}
    df['Delhi_Temp'] = df['Month'].map(delhi_temp_profile)
    return df

# Validate asset file paths
required_files = ['lstm_store_model.pth', 'scaler.pkl', 'metadata.pkl', 'Store Sales Hoi Assignment Data Scientist.xlsx']
missing_files = [f for f in required_files if not os.path.exists(f)]

if missing_files:
    st.error(f"Missing mandatory pipeline resources: {missing_files}. Please execute your Jupyter pipeline notebook first.")
else:
    # Unpickle model metadata and weights
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    with open('metadata.pkl', 'rb') as f:
        meta = pickle.load(f)
        
    store_columns = meta['store_columns']
    feature_columns = meta['feature_columns']
    window_size = meta['window_size']
    num_stores = len(store_columns)
    
    # Load model configuration weights
    model = MultiOutputLSTM(input_size=len(feature_columns), hidden_size=64, output_size=num_stores)
    model.load_state_dict(torch.load('lstm_store_model.pth'))
    model.eval()

    # Read data securely with accurate column typings
    @st.cache_data
    def fetch_processed_dataframe():
        df = pd.read_excel('Store Sales Hoi Assignment Data Scientist.xlsx', sheet_name='Sheet1')
        df.columns = df.columns.astype(str)
        df.iloc[:, 1:] = df.iloc[:, 1:].ffill().bfill().fillna(0)
        return dynamic_feature_generator(df)

    dataset = fetch_processed_dataframe()
    st.success("Optimized Model Weights and Context Configurations Mounted Successfully!")
    
    # Action Deployment Triggers
    if st.button("Evaluate Target Window and Run Forecast Projection Loops"):
        
        # --- PHASE A: ACCURACY EVALUATION VALIDATION ENGINE (APRIL 01 - APRIL 12) ---
        validation_mask = (dataset['Date'] >= '2026-04-01') & (dataset['Date'] <= '2026-04-12')
        val_indices = dataset[validation_mask].index
        
        scaled_full_data = scaler.transform(dataset[feature_columns])
        val_predictions_list = []
        
        for idx in val_indices:
            current_window = scaled_full_data[idx - window_size : idx, :]
            tensor_window = torch.tensor(current_window, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                pred_out = model(tensor_window).numpy()[0]
            val_predictions_list.append(pred_out)
            
        dummy_inverter = np.zeros((len(val_predictions_list), len(feature_columns)))
        dummy_inverter[:, :num_stores] = val_predictions_list
        rescaled_preds = scaler.inverse_transform(dummy_inverter)[:, :num_stores]
        
        pred_df = pd.DataFrame(rescaled_preds, columns=store_columns, index=dataset[validation_mask]['Date'])
        actual_df = dataset[validation_mask][store_columns].set_index(dataset[validation_mask]['Date'])
        
        # Calculate system-wide percentage metric
        mape_values = np.mean(np.abs((actual_df - pred_df) / (actual_df + 1e-5))) * 100
        
        st.header("Model Validation Window Performance Metrics")
        st.metric(label="Overall Corporate Network Evaluation MAPE (Apr 1 - Apr 12)", value=f"{mape_values.mean():.2f}%")
        
        # Interactive plotting section
        st.subheader("Store-Level Validation Target Comparison")
        picked_store = st.selectbox("Select specific store location profile to plot:", store_columns)
        
        fig, ax = plt.subplots(figsize=(11, 4))
        ax.plot(actual_df.index, actual_df[picked_store], label="Actual Realized Sales", marker='o', color='#1f77b4')
        ax.plot(pred_df.index, pred_df[picked_store], label="LSTM Model Prediction", linestyle='--', marker='x', color='#ff7f0e')
        ax.set_title(f"Validation Tracking Comparison Profile - Store Location ID: {picked_store}")
        ax.set_ylabel("Sales Value Scale")
        ax.legend()
        st.pyplot(fig)
        
        # --- PHASE B: EXTENDED MULTI-STEP FORECAST THROUGH MAY 31 ---
        st.header("Projected Sales Horizon (Through May 31st, 2026)")
        
        future_timeline = pd.date_range(start="2026-05-01", end="2026-05-31", freq='D')
        future_dataframe = pd.DataFrame({'Date': future_timeline})
        
        for store in store_columns:
            future_dataframe[store] = 0.0
            
        future_dataframe = dynamic_feature_generator(future_dataframe)
        appended_master = pd.concat([dataset, future_dataframe], ignore_index=True)
        
        for step in range(len(dataset), len(appended_master)):
            inference_window = appended_master.iloc[step - window_size : step]
            scaled_inference_window = scaler.transform(inference_window[feature_columns])
            tensor_input = torch.tensor(scaled_inference_window, dtype=torch.float32).unsqueeze(0)
            
            with torch.no_grad():
                raw_forecast = model(tensor_input).numpy()[0]
                
            inversion_row = np.zeros((1, len(feature_columns)))
            inversion_row[0, :num_stores] = raw_forecast
            actualized_forecast_row = scaler.inverse_transform(inversion_row)[0, :num_stores]
            
            appended_master.loc[step, store_columns] = actualized_forecast_row
            
        final_output_forecast = appended_master[appended_master['Date'] >= '2026-05-01'][['Date'] + store_columns]
        st.dataframe(final_output_forecast.style.format(precision=2))
        
        # Download handler block
        csv_bytes = final_output_forecast.to_csv(index=False).encode('utf-8')
        st.download_button(label="Export Final May 2026 Predictions Sheet (CSV)", data=csv_bytes, file_name="May_2026_Predictions_Data_Scientist.csv", mime="text/csv")