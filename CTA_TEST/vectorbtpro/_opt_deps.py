# Copyright (c) 2021-2023 Oleg Polakow. All rights reserved.

"""Optional dependencies."""

from vectorbtpro.utils.config import HybridConfig

__all__ = []

__pdoc__ = {}

opt_dep_config = HybridConfig(
    dict(
        yfinance=dict(
            link="https://github.com/ranaroussi/yfinance",
            version=">=0.2.18",
        ),
        binance=dict(
            dist_name="python-binance",
            link="https://github.com/sammchardy/python-binance",
            version=">=1.0.16",
        ),
        ccxt=dict(
            link="https://github.com/ccxt/ccxt",
            version=">=1.89.14",
        ),
        ta=dict(
            link="https://github.com/bukosabino/ta",
        ),
        pandas_ta=dict(
            link="https://github.com/twopirllc/pandas-ta",
        ),
        talib=dict(
            dist_name="TA-Lib",
            link="https://github.com/mrjbq7/ta-lib",
        ),
        bottleneck=dict(
            link="https://github.com/pydata/bottleneck",
        ),
        numexpr=dict(
            link="https://github.com/pydata/numexpr",
        ),
        ray=dict(
            link="https://github.com/ray-project/ray",
            version=">=1.4.1",
        ),
        dask=dict(
            link="https://github.com/dask/dask",
        ),
        matplotlib=dict(
            link="https://github.com/matplotlib/matplotlib",
            version=">=3.2.0",
        ),
        plotly=dict(
            link="https://github.com/plotly/plotly.py",
            version=">=5.0.0",
        ),
        ipywidgets=dict(
            link="https://github.com/jupyter-widgets/ipywidgets",
            version=">=7.0.0",
        ),
        kaleido=dict(
            link="https://github.com/plotly/Kaleido",
        ),
        telegram=dict(
            dist_name="python-telegram-bot",
            link="https://github.com/python-telegram-bot/python-telegram-bot",
            version=">=13.4",
        ),
        quantstats=dict(
            link="https://github.com/ranaroussi/quantstats",
            version=">=0.0.37",
        ),
        dill=dict(
            link="https://github.com/uqfoundation/dill",
        ),
        alpaca=dict(
            dist_name="alpaca-py",
            link="https://github.com/alpacahq/alpaca-py",
        ),
        polygon=dict(
            dist_name="polygon-api-client",
            link="https://github.com/polygon-io/client-python",
            version=">=1.0.0",
        ),
        bs4=dict(
            dist_name="beautifulsoup4",
            link="https://www.crummy.com/software/BeautifulSoup/bs4/doc/",
        ),
        nasdaqdatalink=dict(
            dist_name="nasdaq-data-link",
            link="https://github.com/Nasdaq/data-link-python",
        ),
        pypfopt=dict(
            dist_name="PyPortfolioOpt",
            link="https://github.com/robertmartin8/PyPortfolioOpt",
            version=">=1.5.1",
        ),
        universal=dict(
            dist_name="universal-portfolios",
            link="https://github.com/Marigold/universal-portfolios",
        ),
        plotly_resampler=dict(
            dist_name="plotly-resampler",
            link="https://github.com/predict-idlab/plotly-resampler",
        ),
        technical=dict(
            link="https://github.com/freqtrade/technical",
        ),
        riskfolio=dict(
            dist_name="Riskfolio-Lib",
            link="https://github.com/dcajasn/Riskfolio-Lib",
            version=">=3.3.0",
        ),
        pathos=dict(
            link="https://github.com/uqfoundation/pathos",
        ),
        lz4=dict(
            link="https://github.com/python-lz4/python-lz4",
        ),
        blosc=dict(
            link="https://github.com/Blosc/python-blosc",
        ),
        tables=dict(
            link="https://github.com/PyTables/PyTables",
        ),
        optuna=dict(
            link="https://github.com/optuna/optuna",
        ),
    )
)
"""_"""

__pdoc__[
    "opt_dep_config"
] = f"""Config for optional packages.

```python
{opt_dep_config.prettify()}
```
"""
