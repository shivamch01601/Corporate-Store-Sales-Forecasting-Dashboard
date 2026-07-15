import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import pickle
import os
import matplotlib.pyplot as plt

st.set_page_config(page_title="Corporate Store Sales Forecasting Dashboard", layout="wide")

st.title("Corporate Store Sales Forecasting Dashboard")
st.markdown("### Production Evaluation Engine | Responsive Deep Learning Architecture | Author: Shivam")

class RobustGRUForecaster(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RobustGRUForecaster, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers=1, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        out, _ = self.gru(x)
        return self.fc(out[:, -1, :])

def dynamic_feature_generator(df):
    data = df.copy()
    data['DayOfWeek'] = data['Date'].dt.dayofweek
    data['IsWeekend'] = data['DayOfWeek'].isin([5, 6]).astype(int)
    data['Month'] = data['Date'].dt.month
    data['Sin_Month'] = np.sin(2 * np.pi * data['Month'] / 12)
    data['Cos_Month'] = np.cos(2 * np.pi * data['Month'] / 12)
    data['IsHoliday'] = data['Date'].apply(lambda x: 1 if (x.month==1 and x.day==26) or (x.month==8 and x.day==15) or (x.month==10 and x.day==2) or (x.month==12 and x.day==25) else 0)
    delhi_temp_profile = {1: 15, 2: 19, 3: 25, 4: 31, 5: 34, 6: 35, 7: 31, 8: 30, 9: 30, 10: 28, 11: 20, 12: 15}
    data['Delhi_Temp'] = data['Month'].map(delhi_temp_profile)
    return data

required_files = ['lstm_store_model.pth', 'sales_scaler.pkl', 'feature_scaler.pkl', 'metadata.pkl']
if any(not os.path.exists(f) for f in required_files):
    st.error("Error: Missing tracking model artifacts. Please execute your Jupyter Pipeline Engine first.")
else:
    with open('sales_scaler.pkl', 'rb') as f: sales_scaler = pickle.load(f)
    with open('feature_scaler.pkl', 'rb') as f: feature_scaler = pickle.load(f)
    with open('metadata.pkl', 'rb') as f: meta = pickle.load(f)
        
    store_columns, exo_columns, window_size = meta['store_columns'], meta['exo_columns'], meta['window_size']
    num_stores = len(store_columns)
    
    model = RobustGRUForecaster(input_size=num_stores + len(exo_columns), hidden_size=16, output_size=num_stores)
    model.load_state_dict(torch.load('lstm_store_model.pth'))
    model.eval()

    @st.cache_data
    def fetch_processed_dataframe():
        df = pd.read_excel('Store Sales Hoi Assignment Data Scientist.xlsx', sheet_name='Sheet1')
        df.columns = df.columns.astype(str)
        df.iloc[:, 1:] = df.iloc[:, 1:].ffill().bfill().fillna(0)
        df[store_columns] = df[store_columns].clip(lower=0)
        return dynamic_feature_generator(df)

    dataset = fetch_processed_dataframe()
    st.success("Outlier-Resilient Bias-Corrected Scaler Mounted Successfully.")
    
    if st.button("Execute Comprehensive Tracking Metrics and Forecast Projections"):
        # --- PHASE A: VALIDATION METRICS ---
        validation_mask = (dataset['Date'] >= '2026-04-01') & (dataset['Date'] <= '2026-04-12')
        val_df = dataset[validation_mask]
        
        scaled_full_sales = sales_scaler.transform(dataset[store_columns])
        scaled_full_exo = feature_scaler.transform(dataset[exo_columns])
        scaled_full_matrix = np.hstack([scaled_full_sales, scaled_full_exo])
        
        val_predictions_list = []
        for idx in val_df.index:
            current_window = scaled_full_matrix[idx - window_size : idx, :]
            tensor_window = torch.tensor(current_window, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                pred_out = model(tensor_window).numpy()[0]
            val_predictions_list.append(pred_out)
            
        rescaled_preds = sales_scaler.inverse_transform(np.array(val_predictions_list))
        pred_df = pd.DataFrame(rescaled_preds, columns=store_columns, index=val_df['Date'])
        actual_df = val_df[store_columns].set_index(val_df['Date'])
        
        # Isolate active stores (average sales >= 5000) to ensure division by tiny near-zero values in closed stores doesn't skew network accuracy statistics
        active_stores = [col for col in store_columns if actual_df[col].mean() >= 5000]
        
        active_mape_values = np.mean(np.abs((actual_df[active_stores] - pred_df[active_stores]) / actual_df[active_stores])) * 100
        overall_val_accuracy = max(0, 100 - active_mape_values.mean())
        
        st.header("Production Evaluation Performance Summary")
        col1, col2 = st.columns(2)
        with col1:
            st.metric(label="System Test/Validation Accuracy Score", value=f"{overall_val_accuracy:.2f}%")
        with col2:
            st.metric(label="Network Mean Error Limit (MAPE)", value=f"{active_mape_values.mean():.2f}%")
        
        st.subheader("Store Location Tracking Profile Plotter")
        picked_store = st.selectbox("Select specific store ID profile:", store_columns)
        
        fig, ax = plt.subplots(figsize=(11, 4))
        ax.plot(actual_df.index, actual_df[picked_store], label="Actual Realized Sales", marker='o', color='#1f77b4', linewidth=2)
        ax.plot(pred_df.index, pred_df[picked_store], label="LSTM Model Prediction", linestyle='--', marker='x', color='#ff7f0e', linewidth=2)
        ax.set_title(f"Validation Tracking Profile Comparison - Store ID: {picked_store}")
        ax.set_ylabel("Sales Metric Scale")
        ax.legend()
        st.pyplot(fig)
        
        # --- PHASE B: AUTOREGRESSIVE TIME FORECAST TO MAY 31 ---
        st.header("Projected Sales Horizon (Through May 31st, 2026)")
        
        future_timeline = pd.date_range(start="2026-05-01", end="2026-05-31", freq='D')
        future_dataframe = pd.DataFrame({'Date': future_timeline})
        for store in store_columns: future_dataframe[store] = 0.0
        future_dataframe = dynamic_feature_generator(future_dataframe)
        appended_master = pd.concat([dataset, future_dataframe], ignore_index=True)
        
        for step in range(len(dataset), len(appended_master)):
            inference_window = appended_master.iloc[step - window_size : step]
            s_sales = sales_scaler.transform(inference_window[store_columns])
            s_exo = feature_scaler.transform(inference_window[exo_columns])
            combined_window = np.hstack([s_sales, s_exo])
            
            tensor_input = torch.tensor(combined_window, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                raw_forecast = model(tensor_input).numpy()[0]
                
            actualized_forecast_row = sales_scaler.inverse_transform(raw_forecast.reshape(1, -1))[0]
            appended_master.loc[step, store_columns] = actualized_forecast_row
            
        final_output_forecast = appended_master[appended_master['Date'] >= '2026-05-01'][['Date'] + store_columns]
        st.dataframe(final_output_forecast.style.format(precision=2))
        
        csv_bytes = final_output_forecast.to_csv(index=False).encode('utf-8')
        st.download_button(label="Download Final May 2026 Predictions (CSV)", data=csv_bytes, file_name="May_2026_Predictions.csv", mime="text/csv")