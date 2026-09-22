from core.data import load_prices
from core.returns import log_returns
from models.calibration import calibrate_gbm, calibrate_heston

prices = load_prices("AAPL", "2015-01-01", "2024-01-01")
rets = log_returns(prices)

mu, sigma = calibrate_gbm(rets)
print("Calibrated GBM parameters for AAPL (annualised):")
print(f"  drift mu:       {mu:.4f}  ({mu:.1%})")
print(f"  volatility s:   {sigma:.4f}  ({sigma:.1%})")

params = calibrate_heston(rets)
print("\nCalibrated Heston parameters for AAPL:")
for name, value in params.items():
    print(f"  {name:6s}: {value:.4f}")
