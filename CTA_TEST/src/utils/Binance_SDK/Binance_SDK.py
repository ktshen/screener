import time
import requests
import pytz
import hmac
import hashlib
import pandas as pd
import numpy as np
import datetime


tz = pytz.timezone('UTC')


class BinanceSpotClient:
    _EndPoint = 'https://api.binance.com'

    def __init__(self, api_key, api_secret):
        self._api_key = api_key
        self._api_secret = api_secret
        self.header = {'X-MBX-APIKEY': self._api_key}

    def _addSign(self, param, recvWindow=60000):
        timestamp = int((datetime.datetime.now(
            tz) - datetime.datetime.utcfromtimestamp(0).replace(tzinfo=tz)).total_seconds() * 1000)
        param['timestamp'] = timestamp
        param['recvWindow'] = recvWindow
        hashString = ''
        for key in param.keys():
            if param[key]:
                if type(param[key]) == list:
                    for p in param[key]:
                        hashString += key + '=' + str(p) + '&'
                else:
                    hashString += key + '=' + str(param[key]) + '&'
        hashString = hashString[:-1]
        signature = hmac.new(bytes(self._api_secret, 'latin-1'),
                             msg=bytes(hashString, 'latin-1'),
                             digestmod=hashlib.sha256).hexdigest()
        param['signature'] = signature

        return param

    def _process_response(self, response):
        try:
            data = response.json()
        except ValueError:
            response.raise_for_status()
            raise
        if type(data) == dict:
            if 'code' in data.keys():
                raise Exception(str(data))
            else:
                return data
        else:
            return data

    def _get(self, path, params):
        r = requests.get(self._EndPoint+path, headers=self.header,
                         params=self._addSign(params))

        return self._process_response(response=r)

    def _get_2(self, path, params):
        r = requests.get(self._EndPoint+path,
                         headers=self.header, params=(params))
        return self._process_response(response=r)

    def _post(self, path, params):
        r = requests.post(self._EndPoint+path,
                          headers=self.header, params=self._addSign(params))

        return self._process_response(response=r)

    def _delete(self, path, params):
        r = requests.delete(self._EndPoint+path,
                            headers=self.header, params=self._addSign(params))
        return self._process_response(response=r)

    def strToTimestamp(self, dt_str):
        if dt_str:
            dt = datetime.datetime.strptime(dt_str, "%Y-%m-%d %H:%M:%S")
            timestamp = time.mktime(dt.timetuple()) * 1000
            return int(timestamp)
        else:
            return None

    def query_order(self, symbol, order_id):
        return self._get(path='/api/v3/order', params={'symbol': symbol, 'orderId': order_id})

    def place_order(self, symbol, side, _type, quantity, price):
        return self._post(f'/api/v3/order', params={
            'symbol': symbol,
            'side': side,
            'type': _type,
            'timeInForce': 'GTC',
            'quantity': quantity,
            "price": price
        }
        )

    def place_marketOrder(self, symbol, side, quantity):
        return self._post(path='/api/v3/order', params={'symbol': symbol, 'side': side, 'type': 'MARKET', 'quantity': quantity})

    def get_Account(self):
        return self._get(path='/api/v3/account', params={})

    def get_price(self, symbol):
        return self._get_2(path='/api/v3/ticker/bookTicker', params={'symbol': symbol})

    def get_avg_price(self, symbol):
        return self._get_2(path='/api/v3/avgPrice', params={'symbol': symbol})
    
    def get_TradeHistory(self, symbol, startTime=None, endTime=None, limit=1000):
        params = {'symbol': symbol, 'limit': limit}
        if startTime:
            params['startTime'] = self.strToTimestamp(startTime)
        if endTime:
            params['endTime'] = self.strToTimestamp(endTime)
        return self._get(path='/api/v3/myTrades', params=params)
    
    def get_margin_TradeHistory(self, symbol, startTime=None, endTime=None, limit=1000):
        params = {'symbol': symbol, 'limit': limit}
        if startTime:
            params['startTime'] = self.strToTimestamp(startTime)
        if endTime:
            params['endTime'] = self.strToTimestamp(endTime)
        return self._get(path='/sapi/v1/margin/myTrades', params=params)

    def transfer_futuresAccount(self, asset, amount, types):
        return self._post(path='/sapi/v1/futures/transfer', params={'asset': asset, 'amount': amount, 'type': types})

    def transfer_marginAccount(self, asset, amount, types):
        '''
        1: transfer from main account to cross margin account 
        2: transfer from cross margin account to main account
        '''
        return self._post(path='/sapi/v1/margin/transfer', params={'asset': asset, 'amount': amount, 'type': types})

    def place_margin_order(self, symbol, side, price, quantity, _type='LIMIT_MAKER', sideEffectType='MARGIN_BUY', isIsolated=False):
        return self._post(path=f'/sapi/v1/margin/order', params={
            'symbol': symbol,
            'side': side,
            'price': price,
            'type': _type,
            'quantity': quantity,
            'sideEffectType': sideEffectType
        }
        )

    def query_margin_order(self, symbol, order_id):
        return self._get(path='/sapi/v1/margin/order', params={'symbol': symbol, 'orderId': order_id})

    def borrow_futuresAccount(self, coin, collateralCoin, collateralAmount):
        return self._post(path='/sapi/v1/futures/loan/borrow', params={'coin': coin, 'collateralCoin': collateralCoin, 'collateralAmount': collateralAmount})

    def bswap(self):
        return self._get(path='/sapi/v1/bswap/pools', params={})

    def get_margin_assets(self):
        return self._get(path='/sapi/v1/margin/allAssets', params={})

    def get_margin_Account(self):
        return self._get(path='/sapi/v1/margin/account', params={})

    def get_margin_repay(self, asset, amount):
        return self._post(path='/sapi/v1/margin/repay', params={'asset': asset, 'amount': amount})

    def dust_to_bnb(self, asset):
        return self._post(path='/sapi/v1/asset/dust', params={'asset': asset})

    def check_flexible_save(self):
        return self._get(path='/sapi/v1/lending/daily/product/list', params={})

    def place_flexible_save(self, ID, amount):
        return self._post(path='/sapi/v1/lending/daily/purchase', params={'productId': ID, 'amount': amount})

    def redeem_flexible_save(self, ID, amount, typee):
        return self._post(path='/sapi/v1/lending/daily/redeem', params={'productId': ID, 'amount': amount, 'type': typee})

    def get_flexible_product_position(self, asset):
        return self._get(path='/sapi/v1/lending/daily/token/position', params={'asset': asset})

    def get_lending_Account(self):
        return self._get(path='/sapi/v1/lending/union/account', params={})

    def get_klines_data(self, symbol, interval, startTime=None, endTime=None, limit=1000):
        df = pd.DataFrame(self._get_2(f'/api/v3/klines', params={
                          'symbol': symbol, 'interval': interval, 'startTime': startTime, "endTime": endTime, 'limit': limit}))
        df = df.rename(columns=dict(zip(df.columns, ['open time', 'open', 'high', 'low', 'close', 'volume', 'close time',
                       'Quote asset volume', 'Number of trades', 'Taker buy base asset volume', 'Taker buy quote asset volume', 'Unused field'])))
        return df

    def cancel_orders(self, symbol):
        return self._delete(f'/api/v3/openOrders', params={'symbol': symbol})

    def cancel_margin_orders(self, symbol):
        return self._delete(f'/sapi/v1/margin/openOrders', params={'symbol': symbol})
    
    def get_open_orders(self):
        return self._get(f'/sapi/v1/margin/openOrders', params={})

    def get_exchange_information(self):
        return self._get_2(f'/api/v3/exchangeInfo', params={})

    def get_trades(self, symbol, limit=500):
        return self._get_2(f'/api/v3/trades', params={'symbol': symbol, 'limit': limit})

    def get_orderbook(self, symbol, limit=10):
        return self._get_2(path='/api/v3/depth', params={'symbol': symbol, 'limit': limit})

    def get_statistics(self):
        return self._get_2(path='/api/v3/ticker/24hr', params={})

    def get_borrow_rates(self, vip_level=0):
        if vip_level != 0:
            return self._get(path='/sapi/v1/margin/crossMarginData', params={'vipLevel': vip_level})
        else:
            return self._get(path='/sapi/v1/margin/crossMarginData', params={})

    def asset_transfer(self, _type, asset, amount):
        """
            MAIN_UMFUTURE Spot account transfer to USDⓈ-M Futures account
            MAIN_CMFUTURE Spot account transfer to COIN-M Futures account
            MAIN_MARGIN Spot account transfer to Margin（cross）account
            UMFUTURE_MAIN USDⓈ-M Futures account transfer to Spot account
            UMFUTURE_MARGIN USDⓈ-M Futures account transfer to Margin（cross）account
            CMFUTURE_MAIN COIN-M Futures account transfer to Spot account
            CMFUTURE_MARGIN COIN-M Futures account transfer to Margin(cross) account
            MARGIN_MAIN Margin（cross）account transfer to Spot account
            MARGIN_UMFUTURE Margin（cross）account transfer to USDⓈ-M Futures
            MARGIN_CMFUTURE Margin（cross）account transfer to COIN-M Futures
            ISOLATEDMARGIN_MARGIN Isolated margin account transfer to Margin(cross) account
            MARGIN_ISOLATEDMARGIN Margin(cross) account transfer to Isolated margin account
            ISOLATEDMARGIN_ISOLATEDMARGIN Isolated margin account transfer to Isolated margin account
            MAIN_FUNDING Spot account transfer to Funding account
            FUNDING_MAIN Funding account transfer to Spot account
            FUNDING_UMFUTURE Funding account transfer to UMFUTURE account
            UMFUTURE_FUNDING UMFUTURE account transfer to Funding account
            MARGIN_FUNDING MARGIN account transfer to Funding account
            FUNDING_MARGIN Funding account transfer to Margin account
            FUNDING_CMFUTURE Funding account transfer to CMFUTURE account
            CMFUTURE_FUNDING CMFUTURE account transfer to Funding account
        """
        return self._post('/sapi/v1/asset/transfer', {
            'type': _type,
            'asset': asset,
            'amount': amount
        })


class BinanceFuturesClient:
    _EndPoint = 'https://fapi.binance.com'

    def __init__(self, api_key,
                 api_secret):
        self._api_key = api_key
        self._api_secret = api_secret
        self.header = {'X-MBX-APIKEY': self._api_key}

    def _addSign(self, param, recvWindow=5000):
        timestamp = int((datetime.datetime.now(
            tz) - datetime.datetime.utcfromtimestamp(0).replace(tzinfo=tz)).total_seconds() * 1000)
        param['timestamp'] = timestamp
        param['recvWindow'] = recvWindow
        hashString = ''
        for key in param.keys():
            hashString += key + '=' + str(param[key]) + '&'
        hashString = hashString[:-1]
        signature = hmac.new(bytes(self._api_secret, 'latin-1'),
                             msg=bytes(hashString, 'latin-1'),
                             digestmod=hashlib.sha256).hexdigest()
        param['signature'] = signature
        return param

    def _process_response(self, response):
        try:
            data = response.json()
        except ValueError:
            response.raise_for_status()
            raise
        if type(data) == dict:
            if 'code' in data.keys():
                if data['code'] != 200:
                    raise Exception(str(data))
                else:
                    return data
            else:
                return data
        else:
            return data

    def _get(self, path, params):
        r = requests.get(self._EndPoint+path, headers=self.header,
                         params=self._addSign(params))

        return self._process_response(response=r)

    def _post(self, path, params):
        r = requests.post(self._EndPoint+path,
                          headers=self.header, params=self._addSign(params))
        return self._process_response(response=r)

    def _delete(self, path, params):
        r = requests.delete(self._EndPoint+path,
                            headers=self.header, params=self._addSign(params))
        return self._process_response(response=r)

    def get_next_funding_rate(self, symbol):
        resp = self._get(path='/fapi/v1/premiumIndex',
                         params={'symbol': symbol})
        return float(resp['lastFundingRate'])

    def strToTimestamp(self, dt_str):
        dt = datetime.datetime.strptime(dt_str, "%Y-%m-%d %H:%M:%S")
        timestamp = time.mktime(dt.timetuple()) * 1000
        return int(timestamp)

    def place_marketOrder(self, symbol, side, quantity, reduce_only=False):
        return self._post(path='/fapi/v1/order', params={'symbol': symbol, 'side': side, 'type': 'MARKET', 'quantity': quantity, 'reduceOnly': reduce_only})

    def query_order(self, symbol, orderId):
        return self._get(path='/fapi/v1/order', params={'symbol': symbol, 'orderId': orderId})

    def query_orders(self, symbol):
        return self._get(path='/fapi/v1/allOrders', params={'symbol': symbol})

    def get_position(self):
        return self._get(path='/fapi/v1/positionRisk', params={})

    def get_account(self):
        return self._get(path='/fapi/v2/account', params={})

    def get_balances(self):
        return self._get(path='/fapi/v2/balance', params={})

    def get_IncomeHistory(self, symbol='', startTime=0, limit=1000):  # , startTime, endTime,
        if startTime == 0:
            return self._get(path='/fapi/v1/income', params={'limit': limit, 'incomeType': 'FUNDING_FEE'})
        else:
            return self._get(path='/fapi/v1/income', params={'limit': limit, 'startTime': startTime, 'incomeType': 'FUNDING_FEE'})

    def get_TradeHistory(self, symbol='', startTime=None, endTime=None, limit=1000):
        params = {'symbol': symbol, 'limit': limit}
        if startTime:
            params['startTime'] = self.strToTimestamp(startTime)
        if endTime:
            params['endTime'] = self.strToTimestamp(endTime)
        return self._get(path='/fapi/v1/userTrades', params=params)

    def get_exchange_information(self):
        return self._get(path='/fapi/v1/exchangeInfo', params={})

    def change_leverage(self, symbol, leverage):
        return self._post(path='/fapi/v1/leverage', params={'symbol': symbol, 'leverage': int(leverage)})

    def change_margin_type(self, symbol, marginType):
        return self._post(path='/fapi/v1/marginType', params={'symbol': symbol, 'marginType': marginType})

    def change_position_mode(self, dualSidePosition):
        return self._post(path='/fapi/v1/positionSide/dual', params={'dualSidePosition': dualSidePosition})

    def get_fundingrate(self, symbol, limit=1000):
        return self._get(path='/fapi/v1/fundingRate', params={'symbol': symbol, 'limit': limit})

    def get_price(self, symbol):
        return self._get(path='/fapi/v1/ticker/price', params={'symbol': symbol})

    def get_orderbook(self, symbol, limit=10):
        return self._get(path='/fapi/v1/depth', params={'symbol': symbol, 'limit': limit})

    def get_klines_data(self, pair, interval, contractType, startTime=None, endTime=None, limit=1000):
        df = pd.DataFrame(self._get(f'/fapi/v1/continuousKlines', params={
                          'pair': pair, 'contractType': contractType, 'interval': interval, 'startTime': startTime, "endTime": endTime, 'limit': limit}))
        df = df.rename(columns=dict(zip(df.columns, ['open time', 'open', 'high', 'low', 'close', 'volume', 'close time',
                       'Quote asset volume', 'Number of trades', 'Taker buy base asset volume', 'Taker buy quote asset volume', 'Unused field'])))
        return df
    
    def get_all_kline_data(self,pair, interval, contractType, start_time, sleep_time = 0.1):
        start = datetime.datetime.strptime(start_time, "%Y-%m-%d %H:%M:%S").timestamp()*1000
        end = datetime.datetime.now().timestamp()*1000
        result = pd.DataFrame()
        while start<end:
            time.sleep(sleep_time)
            try:
                df = self.get_klines_data(pair,interval,contractType,startTime=start,endTime=end)
                end = df.iloc[0]['open time'] -1
                result = pd.concat([df,result])
            except Exception as e:
                print(e)
                break
        result['datetime'] = pd.to_datetime(result['open time'],unit='ms')
        result = result.set_index('datetime').drop_duplicates()
        result = result.reset_index()
        return result


    def place_order(self, symbol, side, _type, timeInForce, quantity, price, reduce_only):
        # GTX type for post only
        return self._post(f'/fapi/v1/order', params={
            'symbol': symbol,
            'side': side,
            'type': _type,
            'timeInForce': timeInForce,
            'quantity': quantity,
            "price": price,
            "reduceOnly": reduce_only
        }
        )

    def cancel_order(self, symbol, orderId=None, origClientOrderId=None):
        params = {"symbol": symbol}
        if(orderId is not None):
            params['orderId'] = orderId
        if((origClientOrderId is not None)):
            params['origClientOrderId'] = origClientOrderId
        return self._delete(f'/fapi/v1/order', params=params)

    def get_open_orders(self):
        return self._get(path='/fapi/v1/openOrders', params={})

    def cancel_orders(self, symbol):
        return self._delete(f'/fapi/v1/allOpenOrders', params={
            'symbol': symbol
        }
        )

    def get_statistics(self):
        return self._get(path='/fapi/v1/ticker/24hr', params={})

    def get_trades(self, symbol, limit=500):
        return self._get(f'/fapi/v1/trades', params={'symbol': symbol, 'limit': limit})


class BinanceDeliveryClient:
    _EndPoint = 'https://dapi.binance.com'

    def __init__(self, api_key,
                 api_secret):
        self._api_key = api_key
        self._api_secret = api_secret
        self.header = {'X-MBX-APIKEY': self._api_key}

    def _addSign(self, param, recvWindow=5000):
        timestamp = int((datetime.datetime.now(
            tz) - datetime.datetime.utcfromtimestamp(0).replace(tzinfo=tz)).total_seconds() * 1000)
        param['timestamp'] = timestamp
        param['recvWindow'] = recvWindow
        hashString = ''
        for key in param.keys():
            hashString += key + '=' + str(param[key]) + '&'
        hashString = hashString[:-1]
        signature = hmac.new(bytes(self._api_secret, 'latin-1'),
                             msg=bytes(hashString, 'latin-1'),
                             digestmod=hashlib.sha256).hexdigest()
        param['signature'] = signature
        return param

    def _process_response(self, response):
        try:
            data = response.json()
        except ValueError:
            response.raise_for_status()
            raise
        if type(data) == dict:
            if 'code' in data.keys():
                if data['code'] != 200:
                    raise Exception(str(data))
                else:
                    return data
            else:
                return data
        else:
            return data

    def _get(self, path, params):
        r = requests.get(self._EndPoint+path, headers=self.header,
                         params=self._addSign(params))

        return self._process_response(response=r)

    def _post(self, path, params):
        r = requests.post(self._EndPoint+path,
                          headers=self.header, params=self._addSign(params))
        return self._process_response(response=r)

    def strToTimestamp(self, dt_str):
        dt = datetime.datetime.strptime(dt_str, "%Y-%m-%d %H:%M:%S")
        timestamp = time.mktime(dt.timetuple()) * 1000
        return timestamp

    def place_marketOrder(self, symbol, side, quantity):
        return self._post(path='/dapi/v1/order', params={'symbol': symbol, 'side': side, 'type': 'MARKET', 'quantity': quantity})

    def query_order(self, symbol, orderId):
        return self._get(path='/dapi/v1/order', params={'symbol': symbol, 'orderId': orderId})

    def get_position(self):
        return self._get(path='/dapi/v1/positionRisk', params={})

    def get_account(self):
        return self._get(path='/dapi/v1/account', params={})

    def get_balance(self):
        return self._get(path='/dapi/v1/balance', params={})

    def get_IncomeHistory(self, symbol, incomeType, startTime='', endTime='', limit=1000):
        startTimestamp = self.strToTimestamp(startTime)
        endTimestamp = self.strToTimestamp(endTime)
        return self._get(path='/dapi/v1/income', params={'symbol': symbol, 'startTime': startTimestamp, 'endTime': endTimestamp, 'incomeType': incomeType, 'limit': limit})

    def get_TradeHistory(self, symbol, limit=100):  # , startTime, endTime
        #startTimestamp = self.strToTimestamp(startTime)
        #endTimestamp = self.strToTimestamp(endTime)
        # 'startTime': startTimestamp, 'endTime': endTimestamp,
        return self._get(path='/dapi/v1/userTrades', params={'symbol': symbol, 'limit': limit})

    def get_exchange_information(self):
        return self._get(path='/dapi/v1/exchangeInfo', params={})

    def change_leverage(self, symbol, leverage):
        return self._post(path='/dapi/v1/leverage', params={'symbol': symbol, 'leverage': int(leverage)})

    def change_margin_type(self, symbol, marginType):
        return self._post(path='/dapi/v1/marginType', params={'symbol': symbol, 'marginType': marginType})

    def change_position_mode(self, dualSidePosition):
        return self._post(path='/dapi/v1/positionSide/dual', params={'dualSidePosition': dualSidePosition})

    def get_price(self, symbol):
        return self._get(path='/dapi/v1/ticker/price', params={'symbol': symbol})

    def get_fundingrate(self, symbol, limit=1000):
        return self._get(path='/dapi/v1/fundingRate', params={'symbol': symbol, 'limit': limit})

    def get_marketrule(self):
        return self._get(path='/dapi/v1/exchangeInfo', params={})

    def get_orderbook(self, symbol):
        return self._get(path='/dapi/v1/depth', params={'symbol': symbol})

    def get_server_time(self):
        return self._get(path='/dapi/v1/time', params={})


# ----------------------------------------------------------------------------------------


def checkKeyAddValue(targetDict, inputKey, inputValue):
    if inputKey in targetDict.keys():
        targetDict[inputKey] += inputValue
    else:
        targetDict[inputKey] = inputValue


def getAllAccountByAsset(spotClient, futuresClient, deliveryClient):
    assetDict = {}
    for asset in spotClient.get_Account()['balances']:
        if float(asset['free']) != 0 or float(asset['locked']) != 0:
            if asset['asset'][:2] != 'LD':
                spotAmount = float(asset['free'])+float(asset['locked'])
                checkKeyAddValue(assetDict, asset['asset'], spotAmount)

    for asset in spotClient.get_lending_Account()['positionAmountVos']:
        if float(asset['amount']) != 0:
            savingAmount = float(asset['amount'])
            checkKeyAddValue(assetDict, asset['asset'], savingAmount)

    for asset in spotClient.get_margin_Account()['userAssets']:
        if float(asset['netAsset']) != 0:
            marginAmount = float(asset['netAsset'])
            checkKeyAddValue(assetDict, asset['asset'], marginAmount)

    for asset in futuresClient.get_account()['assets']:
        if float(asset['marginBalance']) != 0:
            futuresAmount = float(asset['marginBalance'])
            checkKeyAddValue(assetDict, asset['asset'], futuresAmount)

    for asset in deliveryClient.get_account()['assets']:
        if float(asset['marginBalance']) != 0:
            coinFuturesAmount = float(asset['marginBalance'])
            checkKeyAddValue(assetDict, asset['asset'], coinFuturesAmount)

    return assetDict
