"""
Provides functions to extract list of index constituents.
"""
import finnhub
import pandas as pd

# finnhub_client = finnhub.Client(api_key="cdsc9hiad3i727soq8v0cdsc9hiad3i727soq8vg")
# data = finnhub_client.indices_const(symbol="^FTSE")
# data = data['constituents']
# print(data)
# pd.DataFrame(data).to_csv(r"C:\Users\iainm\OneDrive\Documents\Fomorian Capital\model.2.0\data\FTSE.csv")
# print(data)

finnhub_client = finnhub.Client(api_key="cdsc9hiad3i727soq8v0cdsc9hiad3i727soq8vg")
data = finnhub_client.stock_symbols('US')
all_data = pd.DataFrame()
for i in data:
    for j in i:
        all_data.at[i['symbol'], j] = i[j]
all_data.to_csv(r"C:\Users\iainm\OneDrive\Documents\Fomorian Capital\model.2.0\data\sp500_details.csv")
