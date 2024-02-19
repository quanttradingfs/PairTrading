import access as a
import pandas as pd
import numpy as np
import datetime
import math
from alpaca.data import StockHistoricalDataClient
from alpaca.trading.client import TradingClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.trading.requests import OrderRequest, GetCalendarRequest, ClosePositionRequest
#note test 1
# test 2

class PairTrade:

    def __init__(self, keys, z_score_threshold=1, lookback_window=30, timeframe=TimeFrame.Hour):
        self.__stock_client = StockHistoricalDataClient(keys[0], keys[1])
        self.__trading_client = TradingClient(keys[0], keys[1])
        self.__z_score_threshold = z_score_threshold
        self.__lookback_window = lookback_window
        self.__timeframe = timeframe
        self.__current_date = datetime.datetime.strptime(str(pd.Timestamp.today())[:16], '%Y-%m-%d %H:%M')
        self.__actual_pairs = pd.read_feather(f"actual_pairs_2023_Q4.feather").sample(3000)

    def __get_trade_data_pair(self, symbols, start_date=None, end_date=None, timeframe=None):
        """
        Method to get historical stock data for a given pair of stocks over the horizon of self.__lookback_window
        :param symbols: list with symbols of the two stocks
        :return: df with historical data
        """
        if start_date is None or end_date is None or timeframe is None:
            # define start and end date through current date and defined lookback_window (must consider 15 min delay) 
            end_date_trading = self.__current_date - pd.Timedelta(minutes=15)
            end_date = end_date_trading.astimezone(datetime.timezone.utc)
            start_date = end_date_trading - pd.Timedelta(days=1)
            timeframe = self.__timeframe

        # get stock data
        request_params = StockBarsRequest(symbol_or_symbols=symbols, timeframe=timeframe,
                                          start=start_date, end=end_date)

        bars = self.__stock_client.get_stock_bars(request_params)
        data_df = pd.DataFrame()

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
                if data_df.empty():
                    data_df = data_ticker
                else:
                    data_df = data_df.merge(data_ticker, how="outer", suffixes=("", "_right"),
                                            left_index=True, right_index=True)

            # clean columns
            data_df = data_df[data_df.columns.drop(list(data_df.filter(regex='right')))]

        return data_df

    def __get_buy_signal(self, pair):
        """
        Method to determine if a cointegrated pair is worth trading
        :param pair: series containing both ticker symbols of the pair
        :return: series with information on if pair should be traded and how
        """
        data = self.__get_trade_data_pair(list(pair)).filter(regex="vwap")

        data["Return_a"] = data.iloc[:, 0].shift(1) - data.iloc[:, 0]
        data["Return_b"] = data.iloc[:, 1].shift(1) - data.iloc[:, 1]

        # determine spread between stock price series and calculate z-scores of the spread
        data['Spread'] = data["Return_a"] - data["Return_b"]
        data.dropna(inplace=True)

        data['z-scores'] = (data['Spread'] - data['Spread'].mean())/data['Spread'].std()

        # sprd indicates % how often Stock_a trades above Stock_b (sprd>0.5) or vice versa (sprd<0.5)
        sprd = data['Spread'].fillna(0).gt(0).sum()/data['Spread'].count() #TODO: #1 make sure it does not divide by zero

        # when Stock_a trades above Stock_b:‚
        if sprd > 0.75:
          if data['z-scores'].iloc[-1] > self.__z_score_threshold:
              # short 'Stock_a' and buy 'Stock_b'
              return pd.Series([True, "S", "B"])
          elif data['z-scores'].iloc[-1] < (-1 * self.__z_score_threshold):
              # buy 'Stock_a' and short 'Stock_b'
              return pd.Series([True, "B", "S"])
          else:
              return pd.Series([False, "F", "F"])

        # when Stock_b trades above Stock_a:
        elif sprd < 0.25:
          if data['z-scores'].iloc[-1] > self.__z_score_threshold:
            # buy 'Stock_a' and short 'Stock_b'
            return pd.Series([True, "B", "S"])
          elif data['z-scores'].iloc[-1] < (-1 * self.__z_score_threshold):
            # short 'Stock_a' and buy 'Stock_b'
            return pd.Series([True, "S", "B"])
          else:
            return pd.Series([False, "F", "F"])

        else:
            return pd.Series([False, "F", "F"])

    def __get_current_price(self, row):
        """
        Method that fetches most recent prices for a given pair
        :param row: series containing both ticker symbols of the pair
        :return: series containing prices for corresponding symbols and order types
        """
        # get prices for last minute (must consider 15 min delay due to subscription plan)
        end_date_trading = datetime.datetime.strptime(str(pd.Timestamp.today())[:16], '%Y-%m-%d %H:%M')
        end_date_trading = end_date_trading - pd.Timedelta(minutes=15)
        end_date_trading = end_date_trading.astimezone(datetime.timezone.utc)
        start_date_trading = end_date_trading - pd.Timedelta(days=1)  # larger timeframe selected because not all stocks are very liquid

        data = self.__get_trade_data_pair(list(row[["Stock_a", "Stock_b"]]), start_date_trading, end_date_trading,
                                          TimeFrame.Minute).filter(regex="vwap")

        # check if prices for both stocks exist
        if data.shape[1] == 2:
            price_a = data.iloc[-1, 0]
            price_b = data.iloc[-1, 1]

            return pd.Series([price_a, price_b])

        else:
            return pd.Series([np.nan, np.nan])

    def __create_order_list(self):
        """
        Method that creates an order_list that can be used to re-balance a portfolio
        :return: order_list that contains dicts where each dict contains information on a single conintegrated pair
        """
        # get trade signal for pairs
        self.__actual_pairs[["Signal", "Type_a", "Type_b"]] = self.__actual_pairs[["Stock_a", "Stock_b"]].apply(lambda row:
                                                                        self.__get_buy_signal(row), axis=1)

        traded_pairs = self.__actual_pairs[self.__actual_pairs["Signal"]]

        # check if current day is trading day
        start_date_calendar = pd.Timestamp(str(self.__current_date)[:10]) - pd.Timedelta(days=1)
        calendar_request = GetCalendarRequest()
        calendar_request.start = start_date_calendar
        calendar_request.end = pd.Timestamp(str(self.__current_date)[:10])
        trading_calendar = self.__trading_client.get_calendar(calendar_request)
        most_recent_trading_day = trading_calendar[-1].date

        # if current day is not a trading day no pairs are traded
        if str(most_recent_trading_day) == str(self.__current_date)[:10] and not traded_pairs.empty:
            # get price for pairs
            traded_pairs[["Price_a", "Price_b"]] = traded_pairs[["Stock_a", "Stock_b", "Type_a", "Type_b"]].apply(lambda row:
                                                                            self.__get_current_price(row), axis=1)

            # get current account balance
            account_info = self.__trading_client.get_account()
            max_amount_invested = float(account_info.equity) * 0.7

            traded_pairs["Sum_Price"] = abs(traded_pairs["Price_a"]) + abs(traded_pairs["Price_b"])
            traded_pairs.dropna(inplace=True)

            if not traded_pairs.empty:
                pairs_buy = traded_pairs.groupby(traded_pairs["Sum_Price"].cumsum() <= max_amount_invested) \
                                        .Sum_Price.groups[1].values

                traded_pairs = traded_pairs.loc[pairs_buy, :]

                # filter for buy-short pairs
                b_s_pairs = traded_pairs[(traded_pairs["Type_a"] == "B") & (traded_pairs["Type_b"] == "S")].copy()
                s_b_pairs = traded_pairs[(traded_pairs["Type_a"] == "S") & (traded_pairs["Type_b"] == "B")].copy()

                # create order list for buy-short pairs
                order_list_b_s = [{row[0]: 1/row[2], row[1]: -1} for row in b_s_pairs[["Stock_a", "Stock_b", "Beta"]].itertuples(index=False)]
                order_list_s_b = [{row[0]: -1, row[1]: row[2]} for row in s_b_pairs[["Stock_a", "Stock_b", "Beta"]].itertuples(index=False)]

                # b_s_trades_buy_a = {val[0]: 1/val[1] for key, val in b_s_pairs[b_s_pairs["Type_a"] == "B"][["Stock_a", "Beta"]].T.to_dict("list").items()}
                # b_s_trades_buy_b = {val[0]: val[1] for key, val in b_s_pairs[b_s_pairs["Type_b"] == "B"][["Stock_b", "Beta"]].T.to_dict("list").items()}
                # b_s_trades_sell_a = {symbol: -1 for symbol in b_s_pairs[b_s_pairs["Type_a"] == "S"]["Stock_a"]}
                # b_s_trades_sell_b = {symbol: -1 for symbol in b_s_pairs[b_s_pairs["Type_b"] == "S"]["Stock_b"]}

                # b_s_order_list = b_s_trades_buy_a | b_s_trades_buy_b | b_s_trades_sell_a | b_s_trades_sell_b

                # filter for buy-short pairs
                # b_b_pairs = traded_pairs[(traded_pairs["Type_a"] == "B") & (traded_pairs["Type_b"] == "B")].copy()
                #
                # # create order list for buy-buy pairs
                # b_b_trades_buy_a = {symbol: 1 for symbol in b_b_pairs["Stock_a"]}
                # b_b_trades_buy_b = {row["Stock_b"]: row["Beta"] for row in b_b_pairs[["Stock_b", "Beta"]]}
                #
                # b_b_order_list = b_b_trades_buy_a | b_b_trades_buy_b
                #
                # # filter for short-short pairs
                # s_s_pairs = traded_pairs[(traded_pairs["Type_a"] == "S") & (traded_pairs["Type_b"] == "S")].copy()
                #
                # # create order list for buy-buy pairs
                # s_s_trades_buy_a = {symbol: 1 for symbol in s_s_pairs["Stock_a"]}
                # s_s_trades_buy_b = {row["Stock_b"]: row["Beta"] for row in s_s_pairs[["Stock_b", "Beta"]]}
                #
                # s_s_order_list = s_s_trades_buy_a | s_s_trades_buy_b
                #
                # # merge order lists
                # order_list = b_s_order_list | b_b_order_list | s_s_order_list

                order_lists = order_list_b_s + order_list_s_b
            else:
                order_lists = []
        # no trades are executed if exchanges are closed today
        else:
            order_lists = []

        return order_lists

    def __get_positions(self):
        """
        Method that returns current positions of portfolio
        :return: dict containing tickers with corresponding position size
        """
        positions = self.__trading_client.get_all_positions()
        positions_dict = {asset.symbol: int(asset.qty_available) for asset in positions if int(asset.qty_available) != 0}
        return positions_dict

    def __adjust_portfolio(self, new_positions, order_type_buy="market", order_type_sell="market"):
        """
        Method that adjusts portfolio and places corresponding short and long trades
        :param new_positions: dict containing tickers that are adjusted with corresponding position size (final)
        :param order_type_buy: string that determines which order type is used for increasing position (default: market order)
        :param order_type_sell: string that determines which order type is used for decreasing position (default: market order)
        :return: two lists with information on closed and executed trades
        """
        # get current positions
        current_positions = self.__get_positions()

        # close positions that are not in new_positions
        close_positions = [ticker for ticker in current_positions.keys() if ticker not in new_positions.keys()]
        close_order_info = []
        for ticker in close_positions:
            close_order_info.append(self.__trading_client.close_position(ticker))

        # adjust position for each ticker
        adjustment_order_info = []
        for ticker in new_positions.keys():
            # check if already invested and position different
            if ticker in current_positions.keys() and current_positions[ticker] != new_positions[ticker]:
                # determine if current position needs to be increased or decreased
                side_info = ("buy", order_type_buy) if new_positions[ticker] > current_positions[ticker] else ("sell", order_type_sell)

                # check if it is enough to liquidate position
                if math.copysign(1, new_positions[ticker]) != math.copysign(1, current_positions[ticker]):
                    # check if whole position can be liquidated
                    if abs(new_positions[ticker]) >= abs(current_positions[ticker]):
                        adjustment_order_info.append(self.__trading_client.close_position(ticker))
                        quantity = new_positions[ticker] + current_positions[ticker]
                        order_request = OrderRequest(symbol=ticker, qty=quantity, side=side_info[0], type=side_info[1], time_in_force="day")
                        try:
                            adjustment_order_info.append(self.__trading_client.submit_order(order_request))
                        except:
                            # re-buy position if order request failed (can happen if stock not shortable)
                            side_contra = ("buy", order_type_buy) if side_info[0] == "sell" else ("sell", order_type_sell)
                            quantity = adjustment_order_info[-1].qty
                            order_request = OrderRequest(symbol=ticker, qty=quantity, side=side_contra[0], type=side_contra[1], time_in_force="day")
                            adjustment_order_info.append(self.__trading_client.submit_order(order_request))

                    # liquidate only part of position otherwise
                    else:
                        close_request = ClosePositionRequest(qty=str(abs(new_positions[ticker])))
                        adjustment_order_info.append(self.__trading_client.close_position(ticker, close_request))

                # enlarge position otherwise
                else:
                    quantity = abs(new_positions[ticker] - current_positions[ticker])
                    order_request = OrderRequest(symbol=ticker, qty=quantity, side=side_info[0], type=side_info[1], time_in_force="day")
                    try:
                        adjustment_order_info.append(self.__trading_client.submit_order(order_request))
                    except:
                        break

            elif ticker not in current_positions.keys():
                # send order
                side = "buy" if new_positions[ticker] >= 0 else "sell"
                order_request = OrderRequest(symbol=ticker, qty=abs(new_positions[ticker]), side=side, type=order_type_buy, time_in_force="day")
                try:
                    adjustment_order_info.append(self.__trading_client.submit_order(order_request))
                except:
                    break

        return close_order_info, adjustment_order_info

    def trade_pairs(self):
        """
        Method that creates an order_list and trades it
        """
        order_lists = self.__create_order_list()
        for order_list in order_lists:
            _, _ = self.__adjust_portfolio(order_list)

        return None


if __name__ == "__main__":
    keys = a.thore
    App = PairTrade(keys)
    App.trade_pairs()
