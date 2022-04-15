# FAANG

Compare FAANG stocks' performance.

## How to run

Install requirements from requirements.txt using `pip install -r requirements.txt`
and run main.py. Dash will launch a dashboard by default on a localhost:8050/.
<p>To exist out gracefully, hit cntrl+c. This will issue sig kill to stop the server</p>

##TIP
Currently, changing data selection or chart options deselects the previously selected legends. To easily select one and deselect all or to select all, try double-clicking on legends.
<br>Some selections are computationally expensive so it will take time to update the chart e.g. checking percent change daily on the whole data period (1984-2017).
## About Project

This project was made as a part of the interview process. FAANG stocks (Facebook, Apple, Amazon, Netflix and Google)
data is given and their performance needs to be evaluated.
<p>The Dashboard gives the ability to view the performance of different stocks and indexes at the same time. Users can select 
between percentage change, closing price change and candlestick chart. To make these charts more meaningful, a 
volume chart is used along with a selected view.</p>
<p>User can also change the period for which data should be displayed from the last date in data. He/She can 
also choose the sampling frequency of data e.g. whether to sample data weekly or monthly when viewing over a 
5 years.</p>

<p> Along with FAANG stocks, the FAANG index and NASDAQ index are also used in the comparison.
The FAANG index is calculated DOW Jones-style where stock prices are averaged unless there is a change in the process.
The algorithm to calculate index was taken from 
<a href="https://www.investopedia.com/articles/investing/082714/what-dow-means-and-why-we-calculate-it-way-we-do.asp">
Investopedia.com</a>
Initially, only Apple's stock was in the calculation for FAANG's index. Now FAANG Index can't be there with only Apple but 
this was for demonstration purposes. If FAANG were to be calculated when all 5 stocks were available, 
calculations would have been straightforward. The stock prices summation only needed to be averaged out (by n=5). 
</p>
<p>The Implemented algorithm considers the new stock is added when the market opens up and the divisor is updated to add new stock without
causing sudden fluctuation in the Index. This index is used to compare with NASDAQ which is calculated by weighted average.
Weight is assigned as the market capitalization of stock. FAANG Index can also be calculated similarly but both models have their pros and cons.</p>

##Screenshots
<img src="https://drive.google.com/file/d/1uQ6a0qLVGU3bHGDevCPuqlKnltYd2IAW/view?usp=sharing"/>
<img src="https://drive.google.com/file/d/1MVqZfKhds7Hm7FQthO1WY736anlN82ap/view?usp=sharing"/>
<img src="https://drive.google.com/file/d/1RPip8w1hqpmhd9cAIiQlZtw6GSB4I9J0/view?usp=sharing"/>