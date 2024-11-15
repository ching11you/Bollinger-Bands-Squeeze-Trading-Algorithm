Fascinated by the increase in volatility in financial markets since the COVID pandemic due to factors such as rising inflation, monetary policy changes and geopolitical tensions, I became interested in identifying opportunities amongst this chaos.
I researched into different trading strategies and wanted to design a strategy that is unique and was underexplored.
I came across the Bollinger Bands Squeeze Theory.
The theory measures the standard deviation of a security over a period of time and it visualised by an upper and lower band around a moving average of a security.
The theory suggests that as volatility decreases in a usually volatile security, the 'bands' contract and is due to expand as volatility picks up.
Hence, this strategy is a great alternative breakout predictor.
By measuring the width of the upper and lower bands, I can adjust the condition to when the signal is triggered as the width expands to a certain threshold.
Combined with measuring the price change of the day, the strategy has a high probability of identifying the direction of the security to take an appropriate position.

I first began exploring the strategy through a PineScript code on TradingView where I manually adjusts parameters in securities like trial and error to achieve its optimal settings.
I have since transferred this to a grid search optimisation technique in Python using the ranges I have deemed to be best performing in my past experiments.
Hence, the strategy now performs backtesting to not just reveal the performance metrics of the strategy, but also identifies the optimal setting for any security.

I hope you find this strategy interesting!
