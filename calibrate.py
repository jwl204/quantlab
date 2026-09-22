from core.data import load_prices
from core.returns import log_returns
from models.calibration import calibrate_gbm
from models.calibration import calibrate_heston

prices = load_prices("AAPL", "2015-01-01", "2024-01-01")
rets = log_returns(prices)

mu, sigma = calibrate_gbm(rets)
print("Calibrated GBM parameters for AAPL (annualised):")
print("  drift mu:       {:.4f}  ({:.1%})".format(mu, mu))
print("  volatility s:   {:.4f}  ({:.1%})".format(sigma, sigma))

params = calibrate_heston(rets)
print("\nCalibrated Heston parameters for AAPL:")
for name, value in params.items():
    print("  {:6s}: {:.4f}".format(name, value))
