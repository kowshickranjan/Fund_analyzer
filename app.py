import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, StackingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import xgboost as xgb
from sklearn.metrics import mean_squared_error
from scipy.optimize import minimize
import traceback
import warnings
import os
from matplotlib.backends.backend_pdf import PdfPages
import io

warnings.filterwarnings('ignore')
plt.style.use('default')

class AdvancedLumpSumAnalyzer:
    def __init__(self, nav_csv="NAV_DATA.csv"):
        self.nav_csv = nav_csv
        self.risk_free_rate = 0.06
        self.forecast_weeks = 52
        self.fund_mapping = {
            "F001": "ABSL large cap regular",
            "F002": "Axis large cap regular",
            "F003": "DSP large cap regular",
            "F004": "HDFC large cap regular",
            "F005": "HSBC large cap regular",
            "F006": "ICICI large cap regular",
            "F007": "Kotak large cap regular",
            "F008": "UTI large cap regular",
            "F009": "ABSL mid cap regular",
            "F010": "Axis mid cap regular",
            "F011": "DSP mid cap regular",
            "F012": "HDFC mid cap regular",
            "F013": "HSBC mid cap regular",
            "F014": "Kotak mid cap regular",
            "F015": "UTI mid cap regular",
            "F016": "ABSL Small cap regular",
            "F017": "Axis Small cap regular",
            "F018": "DSP Small cap regular",
            "F019": "HDFC Small cap regular",
            "F020": "HSBC Small cap regular",
            "F021": "ICICI Small cap regular",
            "F022": "Kotak Small cap regular",
            "F023": "ABSL large cap and mid cap regular",
            "F024": "Axis large cap and mid cap regular",
            "F025": "DSP large cap and mid cap regular",
            "F026": "HDFC large cap and mid cap regular",
            "F027": "HSBC large cap and mid cap regular",
            "F028": "ICICI large cap and mid cap regular",
            "F029": "Kotak large cap and mid cap regular",
            "F030": "UTI large cap and mid cap regular",
            "F031": "ABSL flexi cap regular",
            "F032": "Axis flexi cap regular",
            "F033": "DSP flexi cap regular",
            "F034": "HDFC flexi cap regular",
            "F035": "HSBC flexi cap regular",
            "F036": "Kotak flexi cap regular",
            "F037": "UTI flexi cap regular"
        }
        self.nav_data = None
        self.selected_funds = None
        self.optimal_weights = None
        self.forecasts = None
        self.cagr_forecasts = None
        self.risk_levels = None
        self.user_lumpsum_amount = None
        self.lumpsum_results = None


    def load_data(self):
        """Load NAV data from local CSV file - AUTO LOAD"""
        if not os.path.exists(self.nav_csv):
            raise FileNotFoundError(f"NAV_DATA.csv not found in {os.getcwd()}. Please place it in the same folder as this script.")
        
        df = pd.read_csv(self.nav_csv)
        df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
        
        # Find date column
        date_col = 'Nav_Date' if 'Nav_Date' in df.columns else 'Nav Date'
        
        if date_col not in df.columns:
            raise ValueError("Date column 'Nav_Date' or 'Nav Date' not found in CSV.")
        
        # Force convert to datetime
        df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
        
        # Check if conversion failed (all NaT)
        if df[date_col].isna().all():
            raise ValueError(f"Failed to parse any dates in column '{date_col}'. Check date format (should be YYYY-MM-DD, DD/MM/YYYY, etc.)")
        
        # Drop rows where date conversion failed
        original_len = len(df)
        df = df.dropna(subset=[date_col])
        dropped = original_len - len(df)
        if dropped > 0:
            st.warning(f"⚠️ Dropped {dropped} rows with invalid dates.")
        
        # Set as index
        df.set_index(date_col, inplace=True)
        df.sort_index(inplace=True)
        
        # Final validation
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index, errors='coerce')
            df = df.dropna()
            if not isinstance(df.index, pd.DatetimeIndex):
                raise ValueError("Index is not DatetimeIndex even after conversion. Check your data.")
        
        self.nav_data = df
        return df


    def simple_markowitz_optimization(self):
        returns = self.nav_data[self.selected_funds].pct_change().dropna()
        five_year_data = returns.tail(1260)
        avg_returns = five_year_data.mean() * 252
        covariance = five_year_data.cov() * 252
        
        optimal_weights = self._optimize_portfolio(avg_returns, covariance)
        self.optimal_weights = optimal_weights
        return optimal_weights


    def _optimize_portfolio(self, returns, covariance):
        n = len(returns)
        def sharpe_ratio(weights):
            p_return = sum(weights[i] * returns.iloc[i] for i in range(n))
            p_risk = np.sqrt(sum(sum(weights[i] * weights[j] * covariance.iloc[i,j] 
                                   for j in range(n)) for i in range(n)))
            return -(p_return - self.risk_free_rate) / p_risk
        constraints = ({'type': 'eq', 'fun': lambda w: sum(w) - 1})
        bounds = [(0.10, 0.50) for _ in range(n)]
        initial = [1/n] * n
        result = minimize(sharpe_ratio, initial, method='SLSQP', bounds=bounds, constraints=constraints)
        return {fund: result.x[i] for i, fund in enumerate(returns.index)}


    def phase2_stacked_regression_forecasting(self):
        all_forecasts = {}
        cagr_forecasts = {}
        for fund in self.selected_funds:
            forecast_df, historical_cagr, conf_interval = self.create_cagr_forecast_model(self.nav_data[fund])
            all_forecasts[fund] = forecast_df
            predicted_cagr_pct = forecast_df['predicted_cagr'].iloc[-1] * 100
            cagr_forecasts[fund] = predicted_cagr_pct
            self.plot_forecast_with_labels(fund, forecast_df, historical_cagr, conf_interval)
        self.forecasts = all_forecasts
        self.cagr_forecasts = cagr_forecasts
        return all_forecasts, cagr_forecasts


    def calculate_drawdown(self, nav_series):
        rolling_max = nav_series.expanding().max()
        drawdown = (nav_series - rolling_max) / rolling_max
        return drawdown


    def create_cagr_forecast_model(self, nav_series):
        y = (nav_series / nav_series.shift(252)) - 1
        y = y.dropna()
        daily_returns = nav_series.pct_change().fillna(0)
        features = pd.DataFrame(index=y.index)
        
        features['return_21d'] = daily_returns.rolling(21).mean().shift(252) * 252
        features['return_63d'] = daily_returns.rolling(63).mean().shift(252) * 252
        features['return_252d'] = daily_returns.rolling(252).mean().shift(252) * 252
        features['vol_21d'] = daily_returns.rolling(21).std().shift(252) * np.sqrt(252)
        features['vol_63d'] = daily_returns.rolling(63).std().shift(252) * np.sqrt(252)
        features['vol_252d'] = daily_returns.rolling(252).std().shift(252) * np.sqrt(252)
        features['momentum_21d'] = (nav_series / nav_series.shift(21) - 1).shift(252)
        features['momentum_63d'] = (nav_series / nav_series.shift(63) - 1).shift(252)
        features['momentum_252d'] = (nav_series / nav_series.shift(252) - 1).shift(252)
        features['sharpe_21d'] = (features['return_21d'] / features['vol_21d']).replace([np.inf, -np.inf], 0)
        features['sharpe_63d'] = (features['return_63d'] / features['vol_63d']).replace([np.inf, -np.inf], 0)
        features['sharpe_252d'] = (features['return_252d'] / features['vol_252d']).replace([np.inf, -np.inf], 0)
        features['drawdown'] = self.calculate_drawdown(nav_series).shift(252)
        features['rsi_14'] = nav_series.rolling(14).apply(
            lambda x: 100 - (100 / (1 + (x.diff().clip(lower=0).mean() / x.diff().clip(upper=0).abs().mean())))
        ).shift(252)
        
        features = features.dropna()
        y_aligned = y.loc[features.index]
        
        if len(features) < 100:
            raise ValueError("Not enough data after feature engineering.")
        
        split_idx = len(features) - self.forecast_weeks
        if split_idx < 50:
            split_idx = len(features) // 2
        
        X_train, X_test = features.iloc[:split_idx], features.iloc[split_idx:]
        y_train, y_test = y_aligned.iloc[:split_idx], y_aligned.iloc[split_idx:]
        
        base_models = [
            ('ridge', Ridge(alpha=1.0)),
            ('forest', RandomForestRegressor(n_estimators=150, max_depth=8, random_state=42)),
            ('xgboost', xgb.XGBRegressor(n_estimators=200, learning_rate=0.08, max_depth=5, random_state=42, verbosity=0))
        ]
        meta_model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
        
        model = Pipeline([
            ('scaler', StandardScaler()),
            ('stacker', StackingRegressor(
                estimators=base_models,
                final_estimator=meta_model,
                cv=5,
                n_jobs=-1
            ))
        ])
        
        model.fit(X_train, y_train)
        
        if len(y_test) > 0:
            test_predictions = model.predict(X_test)
            residuals = y_test - test_predictions
            confidence_interval = 1.96 * np.std(residuals)
        else:
            confidence_interval = 0.15
        
        future_dates = pd.date_range(X_test.index[-1], periods=self.forecast_weeks+1, freq='W')[1:]
        future_features = X_test.iloc[-self.forecast_weeks:] if len(X_test) >= self.forecast_weeks else X_test
        future_predictions = model.predict(future_features)
        
        all_dates = list(X_test.index) + list(future_dates)
        all_predictions = list(test_predictions) + list(future_predictions)
        
        forecast_df = pd.DataFrame({
            'predicted_cagr': all_predictions,
            'best_case_cagr': [p + confidence_interval for p in all_predictions],
            'worst_case_cagr': [p - confidence_interval for p in all_predictions]
        }, index=all_dates)
        
        return forecast_df, y_aligned, confidence_interval


    def plot_forecast_with_labels(self, fund_id, forecast_df, historical_cagr, confidence_interval):
        fig, ax = plt.subplots(figsize=(14, 7))
        
        hist_data = historical_cagr.tail(252).copy()
        hist_data = hist_data.dropna()
        
        # Smooth historical line using interpolation for better visuals
        if len(hist_data) > 1:
            daily_index = pd.date_range(start=hist_data.index[0], end=hist_data.index[-1], freq='D')
            hist_data_reindexed = hist_data.reindex(hist_data.index.union(daily_index)).interpolate(method='time')
            hist_data_smooth = hist_data_reindexed.loc[daily_index]
            ax.plot(hist_data_smooth.index, hist_data_smooth.values * 100,
                    label="Historical 1Y CAGR (%)", color="#2E86C1", linewidth=2.5, alpha=0.85, linestyle='-')
        else:
            ax.plot(hist_data.index, hist_data.values * 100,
                    label="Historical 1Y CAGR (%)", color="#2E86C1", linewidth=2.5, alpha=0.85)

        # Smooth predicted line using interpolation
        if len(forecast_df) > 1:
            pred_smooth = forecast_df['predicted_cagr'].rolling(window=3, center=True, min_periods=1).mean()
            ax.plot(forecast_df.index, pred_smooth * 100,
                    'o-', label="AI Predicted CAGR (%)", color="#28B463", linewidth=2.5, markersize=4, alpha=0.9)
        else:
            ax.plot(forecast_df.index, forecast_df['predicted_cagr'] * 100,
                    'o--', label="AI Predicted CAGR (%)", color="#28B463", linewidth=2.5, markersize=6)

        ax.fill_between(forecast_df.index,
                        forecast_df['worst_case_cagr'] * 100,
                        forecast_df['best_case_cagr'] * 100,
                        color="#58D68D", alpha=0.25, label="Confidence Range (95%)")

        if len(hist_data) > 0:
            current_cagr = hist_data.iloc[-1] * 100
            ax.annotate(f'Current\n{current_cagr:+.1f}%',
                       xy=(hist_data.index[-1], current_cagr),
                       xytext=(15, 25), textcoords='offset points',
                       bbox=dict(boxstyle='round,pad=0.5', facecolor='#2E86C1', alpha=0.9),
                       fontsize=10, color='white', fontweight='bold', ha='center')

        if len(forecast_df) > 0:
            final_pred = forecast_df['predicted_cagr'].iloc[-1] * 100
            final_best = forecast_df['best_case_cagr'].iloc[-1] * 100
            final_worst = forecast_df['worst_case_cagr'].iloc[-1] * 100

            ax.annotate(f'Predicted\n{final_pred:+.1f}%',
                       xy=(forecast_df.index[-1], final_pred),
                       xytext=(-70, 30), textcoords='offset points',
                       bbox=dict(boxstyle='round,pad=0.5', facecolor='#28B463', alpha=0.9),
                       fontsize=10, color='white', fontweight='bold', ha='center')

            ax.annotate(f'Best\n{final_best:+.1f}%',
                       xy=(forecast_df.index[-1], final_best),
                       xytext=(-50, -20), textcoords='offset points',
                       bbox=dict(boxstyle='round,pad=0.4', facecolor='#239B56', alpha=0.8),
                       fontsize=9, color='white', fontweight='bold', ha='center')

            ax.annotate(f'Worst\n{final_worst:+.1f}%',
                       xy=(forecast_df.index[-1], final_worst),
                       xytext=(-50, 20), textcoords='offset points',
                       bbox=dict(boxstyle='round,pad=0.4', facecolor='#E74C3C', alpha=0.8),
                       fontsize=9, color='white', fontweight='bold', ha='center')

        fund_name = self.fund_mapping[fund_id]
        ax.set_title(f"{fund_name} ({fund_id}) - 1-Year CAGR Forecast", fontsize=14, fontweight='bold')
        ax.set_xlabel("Date", fontsize=12, fontweight='bold')
        ax.set_ylabel("Annualized Return (%)", fontsize=12, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(alpha=0.4, linestyle='--')
        ax.axhline(y=self.risk_free_rate * 100, color='red', linestyle='--', label=f'Risk-Free Rate ({self.risk_free_rate:.1%})')
        ax.legend()
        plt.tight_layout()
        st.pyplot(fig)


    def _calculate_risk_levels(self):
        returns = self.nav_data[self.selected_funds].pct_change().dropna().tail(1260)
        annual_volatility = returns.std() * np.sqrt(252)
        risk_levels = {}
        for fund in self.selected_funds:
            vol = annual_volatility[fund]
            if vol < 0.15:
                risk_levels[fund] = "Low"
            elif vol <= 0.25:
                risk_levels[fund] = "Medium"
            else:
                risk_levels[fund] = "High"
        self.risk_levels = risk_levels
        return risk_levels


    def calculate_lumpsum_returns(self):
        if self.risk_levels is None:
            self._calculate_risk_levels()
        amount = self.user_lumpsum_amount
        eq_growth = sum((1/len(self.selected_funds)) * (self.cagr_forecasts[fund]/100) for fund in self.selected_funds)
        eq_projected = amount * (1 + eq_growth)
        opt_growth = sum(self.optimal_weights[fund] * (self.cagr_forecasts[fund]/100) for fund in self.selected_funds)
        opt_projected = amount * (1 + opt_growth)
        improvement = ((opt_projected - eq_projected) / eq_projected) * 100
        self.lumpsum_results = {
            'equal': {'projected': eq_projected, 'growth': eq_growth},
            'optimal': {'projected': opt_projected, 'growth': opt_growth}
        }
        return self.lumpsum_results


    def simple_investment_summary(self):
        amount = self.user_lumpsum_amount
        eq_projected = self.lumpsum_results['equal']['projected']
        opt_projected = self.lumpsum_results['optimal']['projected']
        
        fig = plt.figure(figsize=(18, 12))
        
        # Pie Chart
        ax1 = plt.subplot(2, 2, 1)
        fund_labels = []
        weights = []
        for fund, weight in self.optimal_weights.items():
            fund_name = self.fund_mapping[fund]
            risk = self.risk_levels[fund]
            display_name = f"{fund_name[:15]}..." if len(fund_name) > 18 else fund_name
            label = f"{display_name}\n({weight:.1%}) [{risk}]"
            fund_labels.append(label)
            weights.append(weight)
        colors = plt.cm.Set3.colors[:len(weights)]
        wedges, texts = ax1.pie(weights, labels=fund_labels, colors=colors, startangle=90)
        ax1.set_title('Optimal Lump Sum Allocation (with Risk)', fontsize=14, fontweight='bold', pad=20)
        
        # Bar Chart
        ax2 = plt.subplot(2, 2, 2)
        strategies = ['Equal Allocation', 'Optimal Allocation']
        values = [eq_projected, opt_projected]
        bars = ax2.bar(strategies, values, color=['#87CEEB', '#F4A460'], alpha=0.8)
        for bar in bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + height*0.01, f'₹{height:,.0f}', ha='center', va='bottom', fontweight='bold')
        ax2.set_title(f'Projected Value After 1 Year (Investment: ₹{amount:,.0f})', fontsize=14, fontweight='bold', pad=20)
        ax2.set_ylabel('Projected Value (₹)', fontsize=12)
        ax2.grid(alpha=0.3)
        ax2.set_ylim(0, max(values) * 1.05)
        
        # Allocation Table
        ax3 = plt.subplot(2, 1, 2)
        ax3.axis('tight')
        ax3.axis('off')
        table_data = []
        headers = ['Fund', 'Allocation', 'Expected CAGR', 'Risk Level', 'Amount', 'Projected']
        for fund in self.selected_funds:
            weight = self.optimal_weights[fund]
            cagr = self.cagr_forecasts[fund] / 100
            risk = self.risk_levels[fund]
            allocated_amt = amount * weight
            projected = allocated_amt * (1 + cagr)
            fund_name = self.fund_mapping[fund][:20] + "..." if len(self.fund_mapping[fund]) > 20 else self.fund_mapping[fund]
            table_data.append([
                fund_name,
                f"{weight:.1%}",
                f"{cagr:+.1%}",
                risk,
                f"₹{allocated_amt:,.0f}",
                f"₹{projected:,.0f}"
            ])
        table = ax3.table(cellText=table_data, colLabels=headers, cellLoc='center', loc='center', bbox=[0, 0.2, 1, 0.6])
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)
        risk_color_map = {"Low": "#D5F5E3", "Medium": "#FEF9E7", "High": "#FADBD8"}
        for (i, j), cell in table.get_celld().items():
            if i == 0:
                cell.set_text_props(weight='bold', color='white')
                cell.set_facecolor('#2C3E50')
            else:
                if j == 3:
                    risk_level = cell.get_text().get_text()
                    cell.set_facecolor(risk_color_map.get(risk_level, "white"))
                else:
                    cell.set_facecolor('white')
                cell.set_edgecolor('gray')
                cell.set_linewidth(1)
        
        plt.tight_layout()
        st.pyplot(fig)

        # Display summary table
        st.subheader("📋 Allocation Breakdown")
        summary_data = []
        for fund in self.selected_funds:
            weight = self.optimal_weights[fund]
            cagr = self.cagr_forecasts[fund] / 100
            risk = self.risk_levels[fund]
            allocated_amt = amount * weight
            projected = allocated_amt * (1 + cagr)
            fund_name = self.fund_mapping[fund]
            summary_data.append({
                "Fund": fund_name,
                "Allocation": f"{weight:.1%}",
                "Expected CAGR": f"{cagr:+.1%}",
                "Risk Level": risk,
                "Amount": f"₹{allocated_amt:,.0f}",
                "Projected Value": f"₹{projected:,.0f}"
            })
        
        summary_df = pd.DataFrame(summary_data)
        st.dataframe(summary_df, use_container_width=True)

        return fig  # Return figure for PDF export


# ===== STREAMLIT APP =====
def main():
    st.set_page_config(
        page_title="Advanced Lump Sum Analyzer",
        page_icon="💰",
        layout="wide"
    )
    
    st.title("💰 Advanced Lump Sum Investment Analyzer")
    st.markdown("""
    ### 🎯 Forecast 1-Year CAGR • Optimize Allocation • Beat Risk-Free Rate (6%)
    Automatically loads NAV_DATA.csv from the same folder. No upload needed.
    """)
    
    # Initialize analyzer
    if 'analyzer' not in st.session_state:
        st.session_state.analyzer = AdvancedLumpSumAnalyzer()
    
    analyzer = st.session_state.analyzer
    
    # Auto-load data
    try:
        analyzer.load_data()
        st.success(f"✅ Loaded {len(analyzer.nav_data)} records")
        st.info(f"📅 Date Range: {analyzer.nav_data.index.min().strftime('%Y-%m-%d')} to {analyzer.nav_data.index.max().strftime('%Y-%m-%d')}")
    except Exception as e:
        st.error(f"❌ Failed to load NAV_DATA.csv: {str(e)}")
        st.code(traceback.format_exc())
        st.stop()  # Stop execution if data fails to load

    st.markdown("---")
    st.header("1️⃣ Select Funds for Analysis")
    fund_display = [f"{fid} - {fname}" for fid, fname in analyzer.fund_mapping.items()]
    
    selected_display = st.multiselect(
        "Choose 3-8 funds (recommended)",
        options=fund_display,
        default=fund_display[:5],
        help="Select funds you want to include in your portfolio"
    )
    
    if len(selected_display) < 3:
        st.warning("⚠️ Please select at least 3 funds")
        return
    if len(selected_display) > 8:
        st.warning("⚠️ Maximum 8 funds allowed")
        return
    
    selected_funds = [item.split(" - ")[0] for item in selected_display]
    analyzer.selected_funds = selected_funds
    
    st.write("✅ Selected Funds:")
    for item in selected_display:
        st.write(f"- {item}")
    
    st.markdown("---")
    st.header("2️⃣ Portfolio Optimization")
    if st.button("🚀 Run Markowitz Optimization", type="primary"):
        with st.spinner("Optimizing portfolio allocation..."):
            try:
                weights = analyzer.simple_markowitz_optimization()
                st.success("✅ Optimization Complete!")
                
                st.subheader("📊 Optimization Results")
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**Optimal Allocation:**")
                    for fund, weight in weights.items():
                        st.write(f"{fund} ({analyzer.fund_mapping[fund]}): {weight:.1%}")
                
                returns = analyzer.nav_data[analyzer.selected_funds].pct_change().dropna().tail(1260)
                avg_returns = returns.mean() * 252
                covariance = returns.cov() * 252
                
                equal_weights = [1/len(analyzer.selected_funds)] * len(analyzer.selected_funds)
                eq_return = sum(w * avg_returns[fund] for w, fund in zip(equal_weights, analyzer.selected_funds))
                opt_return = sum(weights[fund] * avg_returns[fund] for fund in analyzer.selected_funds)
                
                st.markdown("**Expected Annual Returns:**")
                st.write(f"Equal Weight: {eq_return:.1%}")
                st.write(f"Optimal Weight: {opt_return:.1%}")
                improvement = ((opt_return - eq_return) / abs(eq_return)) * 100 if eq_return != 0 else 0
                st.write(f"Improvement: {improvement:+.1f}%")
                
            except Exception as e:
                st.error(f"Optimization failed: {str(e)}")
                st.code(traceback.format_exc())
    
    st.markdown("---")
    st.header("3️⃣ AI Forecasting (1-Year CAGR)")
    if analyzer.optimal_weights is not None:
        if st.button("🔮 Generate AI Forecasts", type="primary"):
            with st.spinner("Generating AI forecasts... This may take a few minutes."):
                try:
                    all_forecasts, cagr_forecasts = analyzer.phase2_stacked_regression_forecasting()
                    st.success("✅ Forecasts Generated!")
                    
                    st.subheader("📈 Forecast Summary")
                    forecast_data = []
                    for fund in analyzer.selected_funds:
                        forecast_data.append({
                            "Fund": f"{fund} - {analyzer.fund_mapping[fund]}",
                            "Predicted 1-Year CAGR": f"{cagr_forecasts[fund]:+.1f}%"
                        })
                    
                    forecast_df = pd.DataFrame(forecast_data)
                    st.dataframe(forecast_df, use_container_width=True)
                    
                except Exception as e:
                    st.error(f"Forecasting failed: {str(e)}")
                    st.code(traceback.format_exc())
    else:
        st.info("👈 Run Portfolio Optimization first to enable forecasting")
    
    st.markdown("---")
    st.header("4️⃣ Investment Analysis")
    analyzer.user_lumpsum_amount = st.number_input(
        "Enter your lump sum investment amount (₹)",
        min_value=10000,
        max_value=100000000,
        value=100000,
        step=50000,
        format="%d"
    )
    
    if analyzer.cagr_forecasts is not None:
        if st.button("🧮 Calculate Projected Returns", type="primary"):
            with st.spinner("Calculating projected returns..."):
                try:
                    analyzer.calculate_lumpsum_returns()
                    st.success("✅ Returns Calculated!")
                    
                    st.subheader("🎯 Projected Returns After 1 Year")
                    eq_proj = analyzer.lumpsum_results['equal']['projected']
                    opt_proj = analyzer.lumpsum_results['optimal']['projected']
                    advantage = opt_proj - eq_proj
                    advantage_pct = (advantage / eq_proj) * 100 if eq_proj != 0 else 0
                    
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Equal Allocation", f"₹{eq_proj:,.0f}")
                    col2.metric("Optimal Allocation", f"₹{opt_proj:,.0f}", f"{advantage_pct:+.1f}%")
                    col3.metric("Advantage", f"₹{advantage:,.0f}")
                    
                    st.info(f"""
                    💡 **Recommendation**: With your investment of **₹{analyzer.user_lumpsum_amount:,.0f}**, 
                    the optimal allocation is projected to grow to **₹{opt_proj:,.0f}** in one year.
                    """)
                    
                except Exception as e:
                    st.error(f"Calculation failed: {str(e)}")
                    st.code(traceback.format_exc())
        
        if hasattr(analyzer, 'lumpsum_results') and st.button("📊 Show Complete Investment Summary", type="primary"):
            with st.spinner("Generating investment summary..."):
                try:
                    summary_fig = analyzer.simple_investment_summary()
                    
                    st.subheader("📌 Key Insights")
                    eq_proj = analyzer.lumpsum_results['equal']['projected']
                    opt_proj = analyzer.lumpsum_results['optimal']['projected']
                    advantage = opt_proj - eq_proj
                    advantage_pct = (advantage / eq_proj) * 100 if eq_proj != 0 else 0
                    
                    st.success(f"""
                    ✅ **Optimization Advantage**: By using our AI-recommended allocation instead of equal weighting, 
                    you could potentially gain an additional **₹{advantage:,.0f} ({advantage_pct:+.1f}%)** on your investment.
                    """)
                    
                    high_risk_funds = [fund for fund in analyzer.selected_funds if analyzer.risk_levels[fund] == "High"]
                    if len(high_risk_funds) > 0:
                        st.warning(f"""
                        ⚠️ **Risk Notice**: Your portfolio includes {len(high_risk_funds)} high-risk fund(s): 
                        {', '.join([f'{fund} ({analyzer.fund_mapping[fund]})' for fund in high_risk_funds])}
                        """)

                    # PDF Export Button
                    st.markdown("---")
                    st.subheader("📥 Download Report as PDF")
                    pdf_buffer = io.BytesIO()
                    with PdfPages(pdf_buffer) as pdf:
                        pdf.savefig(summary_fig)
                        # Add forecast plots if available
                        if analyzer.forecasts:
                            for fund in analyzer.selected_funds:
                                fig, ax = plt.subplots(figsize=(14, 7))
                                forecast_df = analyzer.forecasts[fund]
                                hist_data = (analyzer.nav_data[fund] / analyzer.nav_data[fund].shift(252)) - 1
                                hist_data = hist_data.dropna().tail(252)
                                if len(hist_data) > 1:
                                    daily_index = pd.date_range(start=hist_data.index[0], end=hist_data.index[-1], freq='D')
                                    hist_data_reindexed = hist_data.reindex(hist_data.index.union(daily_index)).interpolate(method='time')
                                    hist_data_smooth = hist_data_reindexed.loc[daily_index]
                                    ax.plot(hist_data_smooth.index, hist_data_smooth.values * 100,
                                            label="Historical 1Y CAGR (%)", color="#2E86C1", linewidth=2.5, alpha=0.85)
                                ax.plot(forecast_df.index, forecast_df['predicted_cagr'] * 100,
                                        'o-', label="AI Predicted CAGR (%)", color="#28B463", linewidth=2.5, markersize=4)
                                ax.fill_between(forecast_df.index,
                                                forecast_df['worst_case_cagr'] * 100,
                                                forecast_df['best_case_cagr'] * 100,
                                                color="#58D68D", alpha=0.25, label="Confidence Range (95%)")
                                ax.set_title(f"{analyzer.fund_mapping[fund]} ({fund}) - Forecast", fontsize=14, fontweight='bold')
                                ax.legend()
                                ax.grid(alpha=0.3)
                                plt.tight_layout()
                                pdf.savefig(fig)
                                plt.close(fig)

                    pdf_buffer.seek(0)
                    st.download_button(
                        label="📄 Download Full Report as PDF",
                        data=pdf_buffer,
                        file_name="LumpSum_Investment_Report.pdf",
                        mime="application/pdf"
                    )

                except Exception as e:
                    st.error(f"Summary generation failed: {str(e)}")
                    st.code(traceback.format_exc())
    else:
        st.info("👈 Generate AI Forecasts first to enable return calculations")

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #888; font-size: 0.9em;'>
        Engineered by <strong>Kowshick Ranjan</strong> | Advanced AI-Driven Portfolio Optimization
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()