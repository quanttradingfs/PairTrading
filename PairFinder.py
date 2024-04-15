import pandas as pd
from alpaca.data import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import GetAssetsRequest, GetCalendarRequest
from alpaca.trading.enums import AssetClass
from datetime import datetime
import numpy as np
import statsmodels.api as sm
import statsmodels.tsa.stattools as ts
from scipy.stats import t
from sklearn.decomposition import PCA


class PairFinder:

    def __init__(self, keys, lookback_window=90, p_threshold=0.05):
        self.__stock_client = StockHistoricalDataClient(keys[0], keys[1])
        self.__trading_client = TradingClient(keys[0], keys[1])
        self.__lookback_window = lookback_window
        self.__p_threshold = p_threshold
        self.__potential_pairs_data = None
        self.num_pairs = 0

    def __get_market_data_EQ_Alpaca(self, ticker_sector, start_date=None, end_date=None, timeframe=TimeFrame.Day):
        """
        Method that gets available historical data from Alpaca for tradable US equities that are shortable and no ETFs for
        last year
        """
        # get tickers of tradable and shortable US EQ that are not ETFs if symbols=None
        search_params = GetAssetsRequest(asset_class=AssetClass.US_EQUITY)
        investable_universe = self.__trading_client.get_all_assets(search_params)
        tickers = sorted([stock.symbol for stock in investable_universe if stock.tradable and stock.shortable and "ETF" not in stock.name
                          and stock.fractionable and stock.symbol in ticker_sector])

        # return empty df with there are no tradable tickers for sector
        if len(tickers) == 0:
            return pd.DataFrame()

        if start_date is None or end_date is None:
            # get data for last year
            end_date = pd.Timestamp.today()
            start_date = end_date - pd.Timedelta(days=self.__lookback_window)
            start_date = datetime.strptime(str(start_date)[:10], '%Y-%m-%d')
            end_date = datetime.strptime(str(end_date)[:10], '%Y-%m-%d')

        request_params = StockBarsRequest(symbol_or_symbols=tickers, timeframe=timeframe, start=start_date, end=end_date)

        bars = self.__stock_client.get_stock_bars(request_params)
        data_df = None

        # check if bars is non-empty
        if bars.data:
            # format timestamp column (cutoff time)
            df = bars.df.reset_index()
            df["timestamp"] = df["timestamp"].astype(str).str.split().str[0]
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            df.set_index("timestamp", inplace=True)

            for ticker in df["symbol"].unique():
                data_ticker = df[df["symbol"] == ticker].copy()
                data_ticker.drop(columns="symbol", inplace=True)
                # rename columns such that columns can be assigned to ticker
                data_ticker = data_ticker.add_suffix(f"_{ticker}", axis=1)

                # merge trading data of ticker with parent df
                if data_df is None:
                    data_df = data_ticker
                else:
                    data_df = data_df.merge(data_ticker, how="outer", suffixes=("", "_right"),
                                            left_index=True, right_index=True)

            # clean columns
            data_df = data_df[data_df.columns.drop(list(data_df.filter(regex='right')))]

        return data_df

    def __get_month_first_trading_days(self):
        """
        Method to filter for first trading day of month for last 24 months
        :return: df with first trading days of last 24 months
        """
        # get all trading days for past 24 month
        calendar_request = GetCalendarRequest()
        end_date_calender = pd.Timestamp(str(pd.Timestamp.today())[:10])
        start_date_calendar = end_date_calender - pd.DateOffset(months=24)
        calendar_request.end = end_date_calender
        calendar_request.start = start_date_calendar
        trading_calendar = App.trading_client.get_calendar(calendar_request)

        # filter for first trading days of month
        trading_days = pd.DataFrame(data={"Year": [trading_day.date.year for trading_day in trading_calendar],
                                          "Month": [trading_day.date.month for trading_day in trading_calendar],
                                          "Day": [trading_day.date.day for trading_day in trading_calendar]})
        trading_days.drop_duplicates(subset=["Year", "Month"], keep="first", inplace=True)
        trading_days.reset_index(drop=True, inplace=True)

        return trading_days

    def __test_pair(self, pair):
        """
        Method that test a single pair for cointegration
        :param pair: series that contains information about pair
        :return: series with new information about pair
        """
        # get stock returns for lookback window
        self.num_pairs += 1
        print(self.num_pairs)
        data_pair = self.__potential_pairs_data[[f"vwap_{pair.iloc[0]}", f"vwap_{pair.iloc[1]}"]].dropna()

        stock_a = data_pair.iloc[-self.__lookback_window:, 0]
        stock_b = data_pair.iloc[-self.__lookback_window:, 1]

        stock_a_ret = stock_a/stock_a.shift(1) - 1
        stock_a_ret = stock_a_ret.iloc[1:]
        stock_b_ret = stock_b/stock_b.shift(1) - 1
        stock_b_ret = stock_b_ret.iloc[1:]

        # determine correlation
        val_correlation = stock_a_ret.corr(stock_b_ret)

        if stock_a.index.min() < stock_b.index.min():
            stock_a = stock_a[stock_a.index >= stock_b.index.min()]
        else:
            stock_b = stock_b[stock_b.index >= stock_a.index.min()]

        # test pair for cointegration
        t_stat, p_value, crit_value_5pct = ts.coint(stock_a, stock_b, trend="c")  # constant trend term

        # check if cointegration is statistically significant
        if p_value <= self.__p_threshold:
            # fit E[a - (ß*b)] + c = 0
            stock_b = sm.add_constant(stock_b) # kaggle approach https://www.kaggle.com/code/yekahaaagayeham/pair-trading-strategy-stock-prediction
            model = sm.OLS(stock_a, stock_b)
            results = model.fit()
            val_beta = results.params[f"vwap_{pair['Stock_b']}"]
            val_beta_p_value = results.pvalues[f"vwap_{pair['Stock_b']}"]

            val_constant = results.params["const"]
            val_constant_p_value = results.pvalues["const"]

            return pd.Series([True, val_correlation, p_value, val_beta, val_beta_p_value, val_constant, val_constant_p_value])

        else:
            return pd.Series([False, val_correlation, np.nan, np.nan, np.nan, np.nan, np.nan])

    def __get_autocorrelation(self, data):
        """
        Method to compute autocorrelation value and its statistical significance for given time series
        :param data: Series with return data on stock
        :return: df with information on autocorrelation of time series
        """
        dgf_total = len(data)
        # test autocorrelation for different lags (in months)
        ac_val = data.autocorr(lag=1)
        t_val = ac_val / np.sqrt((1 - ac_val**2) / (dgf_total - 2))  # t-statistic for autocorrelation value
        p_val = 2*(1 - t.cdf(abs(t_val), dgf_total - 2))  # resulting p-value
        results = pd.Series([ac_val, t_val, p_val])

        return results

    def __get_momentum(self, tickers, start_date_mom, end_date_mom):
        """
        Method to compute momentum for stocks of given sector based on autocorrelation of risk factors gained from PCA
        :param tickers: list with tickers of given sector
        :param start_date_mom:
        :param end_date_mom:
        :return:
        """
        ticker_filtered_data = self.__get_market_data_EQ_Alpaca(tickers, start_date_mom, end_date_mom, TimeFrame.Month)

        risk_factors = []
        exposures = []
        for tn in range(0, 12):
            # get data
            ticker_filtered_data_tn = ticker_filtered_data.iloc[tn: tn + 12, :]
            symbols = [column.split("_")[-1] for column in ticker_filtered_data_tn]
            ticker_filtered_data_tn.columns = symbols

            # calculate monthly returns
            ticker_filtered_data_returns = ticker_filtered_data_tn.apply(lambda column: column.shift(-1).apply(np.log)
                                                                                     - column.apply(np.log))
            ticker_filtered_data_returns.dropna(how="all", inplace=True)
            ticker_filtered_data_norm = (ticker_filtered_data_returns - ticker_filtered_data_returns.mean())\
                                        / ticker_filtered_data_returns.std()
            ticker_filtered_data_norm.dropna(axis=1, inplace=True)

            # perform PCA on returns
            PCA_App = PCA(svd_solver="full")
            returns_array = ticker_filtered_data_norm.to_numpy()

            PCA_App.fit(returns_array)

            risk_factors.append(np.atleast_2d(PCA_App.singular_values_))

            exposures.append(pd.DataFrame(data=PCA_App.components_, columns=ticker_filtered_data_norm.columns).T)

        factors = pd.DataFrame(data=np.concatenate(risk_factors))
        autocorrelations = factors.apply(lambda column: self.__get_autocorrelation(column), axis=0).T
        autocorrelations.rename(columns={0: "Autocorrelation", 1: "t-value", 2: "p-value"}, inplace=True)
        significant_factors = autocorrelations[autocorrelations["p-value"] <= 0.05]

        factor_exposures = pd.concat(exposures)
        factor_exposures_ticker = factor_exposures.groupby(factor_exposures.index).agg("mean")
        factor_exposures_filtered = factor_exposures_ticker.loc[:, significant_factors.index]
        stock_momentum = factor_exposures_filtered.dot(significant_factors["Autocorrelation"]).to_frame()

        return stock_momentum

    def find_pairs_actual(self):
        """
        Method to determine if pre-selected pairs are cointegrated
        :return: df with cointegrated pairs and corresponding ß-values
        """
        # only test pairs that are in same sector
        tickers_sector = pd.read_csv("tickers.csv", delimiter=",")[["Symbol", "Sector"]]
        industries = tickers_sector["Sector"].unique()
        actual_pairs_temp_lst = []

        # get all trading days for past 24 month
        calendar_request = GetCalendarRequest()
        end_date_calender = pd.Timestamp(str(pd.Timestamp.today())[:10])
        start_date_calendar = end_date_calender - pd.DateOffset(months=24)
        calendar_request.end = end_date_calender
        calendar_request.start = start_date_calendar
        trading_calendar = App.__trading_client.get_calendar(calendar_request)

        # select first and last trading day (later used as start and end date)
        start_date = trading_calendar[0].date
        end_date = trading_calendar[-1].date

        for index, sector in enumerate(industries):
            ticker_sector = list(tickers_sector[tickers_sector["Sector"] == sector]["Symbol"])

            self.__potential_pairs_data = self.__get_market_data_EQ_Alpaca(ticker_sector).filter(regex="vwap")
            symbols = sorted([column.split("_")[-1] for column in self.__potential_pairs_data.columns])

            if self.__potential_pairs_data.shape[0] == 0 or self.__potential_pairs_data.shape[1] <2:
                continue

            # create all combination of possible pairs
            pairs = {"Stock_a": [], "Stock_b": []}
            for index_1, symbol in enumerate(symbols):
                for index_2 in range(0, index_1):
                    pairs["Stock_a"].append(symbol)
                    pairs["Stock_b"].append(symbols[index_2])

            potential_pairs = pd.DataFrame(data=pairs)

            potential_pairs["Cointegrated"] = False
            potential_pairs[["Correlation", "P_Value", "Beta", "Beta_P_Value", "Constant", "Constant_P_Value",
                             "Momentum_a", "Momentum_b"]] = np.nan

            # determine if pairs are cointegrated
            potential_pairs[["Cointegrated", "Correlation", "P_Value", "Beta", "Beta_P_Value", "Constant", "Constant_P_Value"]] = \
                            potential_pairs.apply(lambda pair: self.__test_pair(pair), axis=1)

            actual_pairs_sector = potential_pairs[potential_pairs["Cointegrated"]]

            actual_pairs_sector["Sector"] = sector
            # momentum = self.__get_momentum(ticker_sector, start_date, end_date)
            #
            # # update momentum for stock_a
            # momentum.rename(columns={0: "Momentum_a"}, inplace=True)
            # actual_pairs_sector.set_index("Stock_a", inplace=True)
            # actual_pairs_sector.update(momentum)
            # actual_pairs_sector.reset_index(inplace=True)
            #
            # # update momentum for stock_b
            # momentum.rename(columns={"Momentum_a": "Momentum_b"}, inplace=True)
            # actual_pairs_sector.set_index("Stock_b", inplace=True)
            # actual_pairs_sector.update(momentum)
            # actual_pairs_sector.reset_index(inplace=True)
            #
            actual_pairs_temp_lst.append(actual_pairs_sector)

            self.num_pairs = 0

        actual_pairs = pd.concat(actual_pairs_temp_lst)
        current_date = pd.Timestamp.today()
        current_month = current_date.month
        actual_pairs.to_feather(f"actual_pairs_{current_date.year}_M{current_month}.feather")

        return None


if __name__ == "__main__":
    keys = ["PK63EMCGHNEWJTXMVH00",  "71apnYcpN6j5Mc9qGdEjPpnd1csGIz3q0yM1uhIc"]
    App = PairFinder(keys)
    App.find_pairs_actual()

    x = 0
