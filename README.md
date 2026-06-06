# NiftyStrategy




#
C:\Users\Vidya sagar\OneDrive\Desktop\myscripts\NiftyMA\Strategy\MA>python .\EMA_14_34_90_Bounce_Backtest.py --timeframes 1m 5m --target-points 50 --use-bos-filter false --use-chop-filter true --chop-lookback 5 --chop-min-ema-overlaps 3 --max-chop-overlap-bars 3 --use-rsi-setup true --trade-rsi-setup true --recent 20 --best-trades 20


Chop Filter = True
The script will actively block trades if market looks sideways.

Chop Lookback = 10
Before taking a new trade, script checks the previous 10 candles.

Chop Min EMA Overlaps = 2
A candle is called “choppy” if its high/low touches at least 2 EMAs.

Example: if candle range touches EMA14 and EMA34, that candle is choppy.

Max Choppy Overlap Bars = 3
Out of the previous 10 candles, if more than 3 candles are choppy, the script blocks the trade.

Simple example:

Previous 10 candles checked
5 candles touched 2 EMAs
Max allowed is 3
Result: trade blocked
So this setting is strict because many candles touch 2 EMAs.

Your winning trade got blocked because before that trade, the market had too many candles touching at least 2 EMAs.

The looser setting was:

Chop Lookback = 5
Chop Min EMA Overlaps = 3
Max Choppy Overlap Bars = 2
This is easier to pass because:

only checks 5 candles
candle must touch all 3 EMAs to count as choppy
so fewer candles are marked choppy
That is why your profitable trade stayed.