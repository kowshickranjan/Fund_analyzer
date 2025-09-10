import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import xgboost as xgb
from sklearn.metrics import mean_squared_error
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')

import io
from matplotlib.backends.backend_pdf import PdfPages

# ========== ANALYZER CLASS ==========
class AdvancedLumpSumAnalyzer:
    def __init__(self, nav_data):
        self.nav_data = nav_data
        self.forecast_weeks = 52
        self.fund_mapping = {
            "F001": "ABSL Large Cap", "F002": "Axis Large Cap", "F003": "DSP Large Cap",
            "F004": "HDFC Large Cap", "F005": "HSBC Large Cap", "F006": "ICICI Large Cap",
            "F007": "Kotak Large Cap", "F008": "UTI Large Cap", "F009": "ABSL Mid Cap",
            "F010": "Axis Mid Cap", "F011": "DSP Mid Cap", "F012": "HDFC Mid Cap",
            "F013": "HSBC Mid Cap", "F014": "Kotak Mid Cap", "F015": "UTI Mid Cap",
            "F016": "ABSL Small Cap", "F017": "Axis Small Cap", "F018": "DSP Small Cap",
            "F019": "HDFC Small Cap", "F020": "HSBC Small Cap", "F021": "ICICI Small Cap",
            "F022": "Kotak Small Cap", "F023": "ABSL Lg+Mid", "F024": "Axis Lg+Mid",
            "F025": "DSP Lg+Mid", "F026": "HDFC Lg+Mid", "F027": "HSBC Lg+Mid",
            "F028": "ICICI Lg+Mid", "F029": "Kotak Lg+Mid", "F030": "UTI Lg+Mid",
            "F031": "ABSL Flexi Cap", "F032": "Axis Flexi Cap", "F033": "DSP Flexi Cap",
            "F034": "HDFC Flexi Cap", "F035": "HSBC Flexi Cap", "F036": "Kotak Flexi Cap",
            "F037": "UTI Flexi Cap"
        }
        self.benchmark_mapping = {
            "F001": "Nifty_100", "F002": "Nifty_100", "F003": "Nifty_100",
            "F004": "Nifty_100", "F005": "Nifty_100", "F006": "Nifty_100",
            "F007": "Nifty_100", "F008": "Nifty_100",
            "F009": "Nifty_Midcap_150", "F010": "Nifty_Midcap_150", "F011": "Nifty_Midcap_150",
            "F012": "Nifty_Midcap_150", "F013": "Nifty_Midcap_150", "F014": "Nifty_Midcap_150",
            "F015": "Nifty_Midcap_150",
            "F016": "Nifty_Smallcap_250", "F017": "Nifty_Smallcap_250", "F018": "Nifty_Smallcap_250",
            "F019": "Nifty_Smallcap_250", "F020": "Nifty_Smallcap_250", "F021": "Nifty_Smallcap_250",
            "F022": "Nifty_Smallcap_250",
            "F023": "Nifty_500", "F024": "Nifty_500", "F025": "Nifty_500",
            "F026": "Nifty_500", "F027": "Nifty_500", "F028": "Nifty_500",
            "F029": "Nifty_500", "F030": "Nifty_500", "F031": "Nifty_500",
            "F032": "Nifty_500", "F033": "Nifty_500", "F034": "Nifty_500",
            "F035": "Nifty_500", "F036": "Nifty_500", "F037": "Nifty_500"
        }
        self.all_benchmarks = ["Nifty_100", "Nifty_Midcap_150", "Nifty_Smallcap_250", "Nifty_500"]

        self.selected_funds = None
        self.optimal_weights = None
        self.cagr_forecasts = None
        self.benchmark_cagr_forecasts = {}
        self.benchmark_forecast_details = {}
        self.risk_levels = None
        self.user_lumpsum_amount = None
        self.model_selection_info = {}
        self.fund_characteristics = {}
        self.lumpsum_results = None


    def load_and_validate_data(self):
        df = self.nav_data.copy()
        df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
        date_col = 'Nav_Date' if 'Nav_Date' in df.columns else 'Nav Date'
        df[date_col] = pd.to_datetime(df[date_col])
        df.set_index(date_col, inplace=True)
        df.sort_index(inplace=True)

        numeric_cols = list(self.fund_mapping.keys()) + self.all_benchmarks
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        self.nav_data = df
        return df


    def get_funds_by_category(self, category):
        mapping = {
            "Large Cap": ["F001", "F002", "F003", "F004", "F005", "F006", "F007", "F008"],
            "Mid Cap": ["F009", "F010", "F011", "F012", "F013", "F014", "F015"],
            "Small Cap": ["F016", "F017", "F018", "F019", "F020", "F021", "F022"],
            "Flexi/Large+Mid Cap": ["F023", "F024", "F025", "F026", "F027", "F028", "F029", "F030", "F031", "F032", "F033", "F034", "F035", "F036", "F037"]
        }
        return mapping.get(category, [])


    def simple_markowitz_optimization(self):
        returns = self.nav_data[self.selected_funds].pct_change().dropna()
        five_year_data = returns.tail(1260)
        avg_returns = five_year_data.mean() * 252
        covariance = five_year_data.cov() * 252

        benchmark_returns = {}
        for fund in self.selected_funds:
            bench = self.benchmark_mapping[fund]
            bench_cagr = self.benchmark_cagr_forecasts.get(bench, 0.0) / 100
            benchmark_returns[fund] = bench_cagr

        optimal_weights = self._optimize_portfolio(avg_returns, covariance, benchmark_returns)
        self.optimal_weights = optimal_weights
        return optimal_weights


    def _optimize_portfolio(self, returns, covariance, benchmark_returns):
        n = len(returns)
        alphas = returns - pd.Series(benchmark_returns)

        def alpha_risk_ratio(weights):
            p_alpha = sum(weights[i] * alphas.iloc[i] for i in range(n))
            p_risk = np.sqrt(sum(sum(weights[i] * weights[j] * covariance.iloc[i,j] for j in range(n)) for i in range(n)))
            return -p_alpha / p_risk if p_risk > 1e-6 else 0

        constraints = ({'type': 'eq', 'fun': lambda w: sum(w) - 1})
        bounds = [(0.10, 0.50) for _ in range(n)]
        initial = [1/n] * n
        result = minimize(alpha_risk_ratio, initial, method='SLSQP', bounds=bounds, constraints=constraints)
        return {fund: result.x[i] for i, fund in enumerate(returns.index)}


    def phase1_forecast_benchmarks(self):
        self.benchmark_cagr_forecasts = {}
        self.benchmark_forecast_details = {}

        for bench in self.all_benchmarks:
            try:
                nav_series = self.nav_data[bench].dropna()
                if len(nav_series) < 252:
                    self.benchmark_cagr_forecasts[bench] = 0.0
                    continue

                char = self.analyze_fund_characteristics(nav_series)
                model_results = self.evaluate_simple_models(nav_series, char)

                best_name = max(model_results, key=lambda x: model_results[x]['composite_score'])
                best = model_results[best_name]

                predicted_cagr = best['forecast_df']['predicted_cagr'].iloc[-1] * 100
                self.benchmark_cagr_forecasts[bench] = predicted_cagr
                self.benchmark_forecast_details[bench] = {
                    'model': best_name,
                    'score': best['composite_score'],
                    'forecast_df': best['forecast_df']
                }

            except:
                self.benchmark_cagr_forecasts[bench] = 0.0


    def phase2_forecast_funds(self):
        self.model_selection_info = {}
        self.fund_characteristics = {}
        self.cagr_forecasts = {}

        for fund in self.selected_funds:
            try:
                nav_series = self.nav_data[fund].dropna()
                if len(nav_series) < 252:
                    self.cagr_forecasts[fund] = 0.0
                    continue

                char = self.analyze_fund_characteristics(nav_series)
                model_results = self.evaluate_simple_models(nav_series, char)

                best_name = max(model_results, key=lambda x: model_results[x]['composite_score'])
                best = model_results[best_name]

                self.model_selection_info[fund] = {
                    'selected_model': best_name,
                    'composite_score': best['composite_score'],
                    'rmse': best['rmse'],
                    'direction_accuracy': best['direction_accuracy'],
                    'characteristics': char
                }

                predicted_cagr = best['forecast_df']['predicted_cagr'].iloc[-1] * 100
                self.cagr_forecasts[fund] = predicted_cagr

            except:
                self.cagr_forecasts[fund] = 0.0


    def analyze_fund_characteristics(self, nav_series):
        dr = nav_series.pct_change().dropna()
        ann_vol = dr.std() * np.sqrt(252)
        vol_cat = "Low" if ann_vol < 0.15 else "Medium" if ann_vol <= 0.25 else "High"

        trend_persistence = 0
        if len(dr) >= 60:
            ac1, ac5 = dr.autocorr(lag=1), dr.autocorr(lag=5)
            trend_persistence = (ac1 + ac5) / 2

        behavior = "Mixed" if abs(trend_persistence) < 0.05 else "Trending" if trend_persistence > 0.05 else "Mean-Reverting"

        return {'annual_volatility': ann_vol, 'volatility_category': vol_cat, 'trend_persistence': trend_persistence, 'behavior_pattern': behavior}


    def evaluate_simple_models(self, nav_series, characteristics):
        y = (nav_series / nav_series.shift(756)) ** (1/3) - 1
        y = y.dropna()
        dr = nav_series.pct_change().fillna(0)

        features = pd.DataFrame(index=y.index)

        for w in [21, 63, 252]:
            features[f'return_{w}d'] = dr.rolling(w).mean().shift(756) * 252
            features[f'vol_{w}d'] = dr.rolling(w).std().shift(756) * np.sqrt(252)
            features[f'momentum_{w}d'] = (nav_series / nav_series.shift(w) - 1).shift(756)

        features = features.dropna()
        y = y.loc[features.index]
        if len(features) < 100:
            raise ValueError("Insufficient data")

        models = {
            'Ridge': Pipeline([('scaler', StandardScaler()), ('model', Ridge(alpha=1.0))]),
            'XGBoost': Pipeline([('scaler', StandardScaler()), ('model', xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42, verbosity=0))])
        }

        tscv = TimeSeriesSplit(n_splits=5)
        results = {}

        for name, model in models.items():
            rmses, dir_accs, hit_rates, biases = [], [], [], []

            for train_idx, test_idx in tscv.split(features):
                Xt, Xv = features.iloc[train_idx], features.iloc[test_idx]
                yt, yv = y.iloc[train_idx], y.iloc[test_idx]

                model.fit(Xt, yt)
                yp = model.predict(Xv)

                rmses.append(np.sqrt(mean_squared_error(yv, yp)))
                dir_accs.append(np.mean(np.sign(yv) == np.sign(yp)))
                hit_rates.append(np.mean(np.abs((yp - yv) / yv) < 0.2) if len(yv) > 0 else 0)
                biases.append(np.mean(yp - yv))

            avg_rmse = np.mean(rmses)
            comp_score = (1/(1+avg_rmse))*0.4 + np.mean(dir_accs)*0.3 + np.mean(hit_rates)*0.2 + (1-min(abs(np.mean(biases)),1))*0.1

            model.fit(features, y)
            future_features = features.iloc[-self.forecast_weeks:] if len(features) >= self.forecast_weeks else features
            future_preds = model.predict(future_features)
            future_dates = pd.date_range(features.index[-1], periods=self.forecast_weeks+1, freq='W')[1:]

            forecast_df = pd.DataFrame({'predicted_cagr': future_preds}, index=future_dates)

            results[name] = {
                'rmse': avg_rmse,
                'direction_accuracy': np.mean(dir_accs),
                'hit_rate': np.mean(hit_rates),
                'bias': np.mean(biases),
                'composite_score': comp_score,
                'forecast_df': forecast_df,
                'historical_cagr': y
            }

        return results


    def plot_fund_vs_benchmark_forecast(self, fund_id):
        if fund_id not in self.cagr_forecasts:
            return None

        try:
            nav_series = self.nav_data[fund_id].dropna()
            y = (nav_series / nav_series.shift(756)) ** (1/3) - 1
            hist_cagr = y.dropna()

            best = self.model_selection_info[fund_id]
            forecast_df = best['forecast_df']
            selected_model = best['selected_model']
            char = best['characteristics']

            bench = self.benchmark_mapping[fund_id]
            bench_forecast = self.benchmark_forecast_details.get(bench, {}).get('forecast_df', None)
            bench_cagr_forecast = self.benchmark_cagr_forecasts.get(bench, 0.0)

            fig, ax = plt.subplots(figsize=(14, 8))
            hist_data = hist_cagr.tail(756).dropna()

            if len(hist_data) > 1:
                daily_idx = pd.date_range(hist_data.index[0], hist_data.index[-1], freq='D')
                hist_smooth = hist_data.reindex(hist_data.index.union(daily_idx)).interpolate(method='time').loc[daily_idx]
                ax.plot(hist_smooth.index, hist_smooth.values * 100,
                        label=f"{self.fund_mapping[fund_id]} Historical 3Y CAGR (%)", color="#2E86C1", linewidth=3, alpha=0.85)
            else:
                ax.plot(hist_data.index, hist_data.values * 100,
                        label="Historical 3Y CAGR (%)", color="#2E86C1", linewidth=3, alpha=0.85)

            ax.plot(forecast_df.index, forecast_df['predicted_cagr'] * 100,
                    'o--', label="AI Predicted Fund 3Y CAGR (%)", color="#28B463", linewidth=3, markersize=8)

            if bench_forecast is not None and len(bench_forecast) > 0:
                ax.plot(bench_forecast.index, bench_forecast['predicted_cagr'] * 100,
                        '--', label=f"{bench} AI Forecasted 3Y CAGR", color="#E74C3C", linewidth=2.5, alpha=0.9)
            else:
                ax.axhline(y=bench_cagr_forecast, color='#E74C3C', linestyle='--', linewidth=2.5,
                           label=f"{bench} Forecasted CAGR ({bench_cagr_forecast:+.1f}%)")

            if len(hist_data) > 0:
                current_cagr = hist_data.iloc[-1] * 100
                ax.annotate(f'Fund Current\n{current_cagr:+.1f}%',
                            xy=(hist_data.index[-1], current_cagr),
                            xytext=(20, 30), textcoords='offset points',
                            bbox=dict(boxstyle='round,pad=0.6', facecolor='#2E86C1', alpha=0.95),
                            fontsize=11, fontweight='bold', color='white', ha='center')

            if len(forecast_df) > 0:
                final_pred = forecast_df['predicted_cagr'].iloc[-1] * 100
                ax.annotate(f'Fund Predicted\n{final_pred:+.1f}%',
                            xy=(forecast_df.index[-1], final_pred),
                            xytext=(-80, 40), textcoords='offset points',
                            bbox=dict(boxstyle='round,pad=0.6', facecolor='#28B463', alpha=0.95),
                            fontsize=11, fontweight='bold', color='white', ha='center')

            fund_name = self.fund_mapping[fund_id]
            vol, beh = char['volatility_category'], char['behavior_pattern']
            title = f"{fund_name} ({fund_id}) vs {bench}\nModel: {selected_model} | Volatility: {vol} | Behavior: {beh}"
            ax.set_title(title, fontsize=14, fontweight='bold')
            ax.set_xlabel("Date", fontsize=12)
            ax.set_ylabel("Annualized Return (%)", fontsize=12)
            ax.legend(fontsize=11)
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            return fig
        except Exception as e:
            st.error(f"Error generating plot for {fund_id}: {str(e)}")
            return None


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


    def calculate_lumpsum_returns(self, amount):
        self.user_lumpsum_amount = amount

        eq_growth = sum((1/len(self.selected_funds)) * (self.cagr_forecasts[fund]/100) for fund in self.selected_funds)
        eq_projected = amount * (1 + eq_growth)

        opt_growth = sum(self.optimal_weights[fund] * (self.cagr_forecasts[fund]/100) for fund in self.selected_funds)
        opt_projected = amount * (1 + opt_growth)

        self.lumpsum_results = {
            'equal': {'projected': eq_projected, 'growth': eq_growth},
            'optimal': {'projected': opt_projected, 'growth': opt_growth}
        }
        return self.lumpsum_results


    def generate_summary_table(self):
        rows = []
        for fund in self.selected_funds:
            weight = self.optimal_weights[fund]
            cagr = self.cagr_forecasts[fund] / 100
            bench = self.benchmark_mapping[fund]
            bench_cagr = self.benchmark_cagr_forecasts.get(bench, 0.0) / 100
            alpha = cagr - bench_cagr
            allocated_amt = self.user_lumpsum_amount * weight
            projected = allocated_amt * (1 + cagr)
            rows.append({
                'Fund Name': self.fund_mapping[fund],
                'Allocation': f"{weight:.1%}",
                '3Y CAGR': f"{cagr:+.1%}",
                'Benchmark CAGR': f"{bench_cagr:+.1%}",
                'Alpha': f"{alpha:+.1%}",
                'Risk Level': self.risk_levels[fund],
                'Selected Model': self.model_selection_info[fund]['selected_model'],
                'Projected Value': f"₹{projected:,.0f}"
            })
        return pd.DataFrame(rows)


    def create_pie_chart(self):
        fig, ax = plt.subplots(figsize=(8, 8))
        weights = [self.optimal_weights[fund] for fund in self.selected_funds]
        labels = [f"{self.fund_mapping[fund]}\n{w:.0%} [{self.risk_levels[fund]}]" for fund, w in zip(self.selected_funds, weights)]
        colors = plt.cm.Set3.colors[:len(weights)]
        wedges, texts = ax.pie(weights, labels=labels, colors=colors, startangle=90, textprops={'fontsize': 10})
        ax.set_title('Optimal Allocation by Fund & Risk Profile', fontsize=14, fontweight='bold')
        return fig


    def create_bar_chart(self):
        fig, ax = plt.subplots(figsize=(8, 6))
        strategies = ['Equal Allocation', 'Optimal Allocation']
        values = [self.lumpsum_results['equal']['projected'], self.lumpsum_results['optimal']['projected']]
        bars = ax.bar(strategies, values, color=['#87CEEB', '#F4A460'], alpha=0.85, edgecolor='black', linewidth=1.5)

        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, height * 1.02,
                    f'₹{height:,.0f}', ha='center', va='bottom',
                    fontweight='bold', fontsize=12)

        ax.set_title(f'Projected Portfolio Value\n(₹{self.user_lumpsum_amount:,.0f} over 3 Years)', fontsize=14, fontweight='bold')
        ax.set_ylabel('Amount (₹)', fontsize=12)
        ax.grid(True, alpha=0.3)
        return fig


    def export_to_pdf(self):
        buf = io.BytesIO()
        with PdfPages(buf) as pdf:
            # Title Page
            fig = plt.figure(figsize=(10, 8))
            fig.text(0.5, 0.5, 'Lump Sum Investment Analysis Report\nAdvanced AI Forecasting Engine', 
                    ha='center', va='center', fontsize=16, fontweight='bold')
            pdf.savefig(fig)
            plt.close(fig)

            # Summary Table
            summary_df = self.generate_summary_table()
            fig, ax = plt.subplots(figsize=(12, len(summary_df)*0.5 + 2))
            ax.axis('tight')
            ax.axis('off')
            table = ax.table(cellText=summary_df.values, colLabels=summary_df.columns, cellLoc='center', loc='center')
            table.auto_set_font_size(False)
            table.set_fontsize(10)
            table.scale(1.2, 2)
            ax.set_title('Investment Allocation Summary', fontsize=14, fontweight='bold', pad=20)
            pdf.savefig(fig)
            plt.close(fig)

            # Pie Chart
            fig = self.create_pie_chart()
            pdf.savefig(fig)
            plt.close(fig)

            # Bar Chart
            fig = self.create_bar_chart()
            pdf.savefig(fig)
            plt.close(fig)

            # Individual Fund Forecasts
            for fund in self.selected_funds:
                fig = self.plot_fund_vs_benchmark_forecast(fund)
                if fig:
                    pdf.savefig(fig)
                    plt.close(fig)

        buf.seek(0)
        return buf


# ========== STREAMLIT APP ==========
st.set_page_config(page_title="Advanced Lump Sum Analyzer", layout="wide")
st.title("🎯 Advanced Lump Sum Investment Analyzer")
st.markdown("### AI-Powered 3-Year CAGR Forecasting vs Benchmarks")

# File uploader
uploaded_file = st.file_uploader("Upload NAV_DATA.csv", type="csv")
if uploaded_file is None:
    st.warning("📥 Please upload your NAV_DATA.csv file to begin.")
    st.stop()

# Load data
@st.cache_data
def load_data(file):
    return pd.read_csv(file)

df = load_data(uploaded_file)
analyzer = AdvancedLumpSumAnalyzer(df)
analyzer.load_and_validate_data()

# Sidebar for inputs
st.sidebar.header("📊 Investment Configuration")

# Step 1: Number of funds
num_funds = st.sidebar.number_input("Select number of funds (3-8)", min_value=3, max_value=8, value=3)

# Step 2: Select by category
st.sidebar.subheader("Step 1: Select Funds by Category")
categories = ["Large Cap", "Mid Cap", "Small Cap", "Flexi/Large+Mid Cap"]
selected_funds = []

for i in range(num_funds):
    st.sidebar.markdown(f"**Fund #{i+1}**")
    category = st.sidebar.selectbox(f"Category for Fund #{i+1}", categories, key=f"cat_{i}")
    available_funds = analyzer.get_funds_by_category(category)
    fund_names = {fid: analyzer.fund_mapping[fid] for fid in available_funds if fid in analyzer.nav_data.columns}
    
    if fund_names:
        selected_fund = st.sidebar.selectbox(
            f"Select Fund", 
            options=list(fund_names.keys()), 
            format_func=lambda x: fund_names[x],
            key=f"fund_{i}"
        )
        if selected_fund and selected_fund not in selected_funds:
            selected_funds.append(selected_fund)
    else:
        st.sidebar.warning(f"No funds available in {category}")

if len(selected_funds) < num_funds:
    st.warning("⚠️ Please select all funds before proceeding.")
    st.stop()

analyzer.selected_funds = selected_funds

# Step 3: Investment amount
investment_amount = st.sidebar.number_input("Investment Amount (₹)", min_value=1000, value=100000, step=1000)

# Run analysis
if st.sidebar.button("🚀 Run Analysis"):
    with st.spinner("⏳ Phase 1: Forecasting Benchmarks..."):
        analyzer.phase1_forecast_benchmarks()
    
    with st.spinner("⏳ Phase 2: Optimizing Portfolio..."):
        analyzer.simple_markowitz_optimization()
        analyzer._calculate_risk_levels()
    
    with st.spinner("⏳ Phase 3: Forecasting Selected Funds..."):
        analyzer.phase2_forecast_funds()
        analyzer.calculate_lumpsum_returns(investment_amount)

    st.success("✅ Analysis Complete!")

    # Display Results
    st.header("📈 Investment Analysis Summary")

    # Summary Table
    st.subheader("📋 Allocation & Performance Summary")
    summary_df = analyzer.generate_summary_table()
    st.dataframe(summary_df, use_container_width=True)

    # Charts Row
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("🥧 Optimal Allocation")
        pie_fig = analyzer.create_pie_chart()
        st.pyplot(pie_fig)

    with col2:
        st.subheader("💰 Projected Growth")
        bar_fig = analyzer.create_bar_chart()
        st.pyplot(bar_fig)

    # Forecast Plots in Summary (Expandable)
    st.subheader("📈 Forecast Details (Fund vs Benchmark)")
    st.markdown("> 💡 _Click each expander to view 3-year CAGR forecast vs benchmark._")
    
    for fund in selected_funds:
        with st.expander(f"📊 {analyzer.fund_mapping[fund]} ({fund})"):
            fig = analyzer.plot_fund_vs_benchmark_forecast(fund)
            if fig:
                st.pyplot(fig)
            else:
                st.error("⚠️ Forecast plot could not be generated for this fund.")

    # Download PDF
    st.subheader("📥 Download Full Report")
    pdf_buffer = analyzer.export_to_pdf()
    st.download_button(
        label="📄 Download PDF Report (Includes All Charts & Tables)",
        data=pdf_buffer,
        file_name="Lumpsum_Investment_Analysis.pdf",
        mime="application/pdf",
        help="✅ Includes summary table, pie chart, bar chart, and all forecast plots."
    )

st.sidebar.markdown("---")
st.sidebar.info("💡 Pro Tip: Select funds from different categories for better diversification and risk management!")