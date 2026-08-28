# Olist Delivery Performance

Python · pandas · scikit-learn · matplotlib · seaborn

I looked at 96,470 delivered orders from Olist, Brazil's largest marketplace, September 2016
to August 2018.

## Summary

Olist gives every customer a delivery date at checkout. It hits that date 93% of the time, and
pays for it with a 12-day buffer on average.

Missing it is expensive. Late orders get 2.27 stars against 4.29 for on-time ones. Late
deliveries are 6.7% of all orders and 32.5% of 1-2 star reviews.

Most of them come from a small group. **50 sellers out of 2,954 cause 38% of all late
deliveries.** That is the fastest thing to fix.

I also tested whether a model could replace Olist's estimate. It predicts more accurately, MAE
3.55 days against 4.42 for a per-state median. It still cannot promise better. To be on time
as often as Olist, it has to promise 23.6 days against their 22.4.

## Notebooks

**[01_eda](notebooks/01_eda.ipynb)** — cleaning, delivery time, how good the promise is,
geography and seasonality

**[02_delivery_and_reviews](notebooks/02_delivery_and_reviews.ipynb)** — what a late delivery
does to the review score, and where late orders come from

**[03_prediction_model](notebooks/03_prediction_model.ipynb)** — can delivery time be
predicted at checkout, and what a promise built on the model would look like

## Where the 12 days go

![Olist estimate error](images/olist_estimate_error.png)

Almost everything sits right of zero. The estimate works as a buffer. It tracks actual
delivery at only 0.38 correlation.

One note on the metric. Olist's date carries a time of `00:00:00`, so an order arriving at 2pm
on the promised day counts as late under a direct comparison, 8.11%. I read it as "by end of
day D", the way a customer would, which gives 6.77%.

## A late order costs two stars

![Review score by delivery delay](images/score_by_delay.png)

Coming early buys nothing. Scores hold at 4.1 to 4.3 whether the order arrived one day or two
weeks ahead. Missing the date drops the score to 3.29 immediately, and after a week it settles
at 1.7 and goes no lower.

The penalty is one-sided, which explains the buffer. Early costs Olist nothing in reviews.
Late costs them two stars.

## It comes down to 50 sellers

50 sellers out of 2,954 carry 2,441 late orders out of 6,381. The worst one carries 168.

A seller shipping to Pará looks bad because of the distance, so I compared each seller against
the average late rate on their own destinations. The ranking holds. The worst sellers ship to
ordinary routes, 6 to 8% expected, and still run 17 to 29% late.

Product category barely moves the number. Late rate runs 4.2% to 8.0% across categories,
against 0% to 29% across sellers.

Geography splits in two. The far North takes 19 to 27 days and misses the date only 3% of the
time, because the estimate already covers the distance. The Northeast is slow and unreliable
at once, AL is 21% late. RJ fits neither group: delivery is average at 15.3 days, yet 12%
arrive late, on the second largest order volume in the country.

![Late rate by month](images/late_rate_by_month.png)

Late rate also tracks volume. Around 0.03 in a normal month, 0.12 in November 2017, 0.19 in
March 2018.

## Can a model do better?

Measured on a time-based test set, June to August 2018:

| | MAE, days |
|---|---|
| Naive train mean | 6.23 |
| Per-state median | 4.42 |
| **Final model, HGB + distance** | **3.55** |

Olist's estimate scores 14.04 here, but it is built to avoid being late rather than to be
accurate, so MAE is the wrong lens for it.

![Model comparison](images/model_comparison.png)

I set the per-state median as the bar rather than the naive mean. A model that cannot beat a
`groupby` is not worth having, and my first RandomForest did not beat it. The final model
comes in 20% under the median.

The biggest single gain came from matching the loss to the metric. Switching squared error to
absolute error moved validation MAE from 5.09 to 4.52, more than any model or feature I tried.
Tuning changed nothing.

**A better prediction does not give a shorter promise.** A prediction sits in the middle of the
distribution, so half the orders arrive after it. Turning it into a promise means adding a
margin, and the margin has to cover the worst orders. Those are exactly what the model misses:
R2 is 0.23, and orders that really take 30+ days get predicted at 10 to 20. Per-state margins
help, from 10.3 days in SP to 30.2 in CE, but not enough to get under Olist's 22.4.

## What I would recommend

1. **Start with the 50 sellers behind 38% of late deliveries.** The list is in notebook 02,
   and it is the smallest change with the largest effect.

2. **Widen the promise before known peaks.** November 2017 ran 7,237 orders against 4,446 the
   month before, and late rate went from 0.04 to 0.12. The promise stayed the same.

3. **Set the margin per route.** SP needs 10 days, CE needs 30. One number cannot serve both.

4. **Do not replace the estimate with this model yet.** It predicts better and promises worse.
   A working version would need per-route margins and refitting every season.

## Method notes

**Time-based split.** Train through May 2018, test on late May to August. A random split would
leak future orders into training. Tuning ran with TimeSeriesSplit inside the training set, and
the test set was scored once.

**Only features known at checkout.** No approval time, no carrier pickup. Delivery time drops
from 11-17 days in 2017 to 8-10 from July 2018, which is why test MAE comes in below
validation MAE.

## Reproduce

```bash
pip install -r requirements.txt
```

Download the [Olist dataset](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce) into
`data/raw/` and run the notebooks in order. 01 builds `data/clean/clean.parquet`, which 02 and
03 both read.