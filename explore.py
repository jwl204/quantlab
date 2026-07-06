import matplotlib.pyplot as plt

from core.data import load_prices
from core.returns import log_returns

prices = load_prices("AAPL", "2018-01-01", "2024-01-01")
rets = log_returns(prices)

print(rets.describe())
print("annualised vol:", rets.std() * (252 ** 0.5))

rets.plot(title="AAPL daily log returns")
plt.tight_layout()
plt.show()