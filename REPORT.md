# Assignment 1 Report

- **Name**: Samer
- **Student ID**:
- **Email**: sroz.ieu2022@student.ie.edu
- **Group**:

## Dataset

NBA games from 2004 to 2022, from Kaggle ([nathanlauga/nba-games](https://www.kaggle.com/datasets/nathanlauga/nba-games), `games.csv` and `teams.csv`). I committed the seasons 2011-12 to 2022-23, which is 15,371 rows and 2.3 MB, in `data/`.

One row is one game. It has the date, the season, the two team IDs, the final points for each team and some box score stats (shooting percentages, assists, rebounds).

After keeping only regular season games with a score, there are 13,445 games. I picked it because I follow basketball, and because the raw table has no useful features at all. Every feature has to be built from the history of each team, so the feature engineering really matters.

## Business / real-life framing

A sportsbook has to post an opening point spread for every game before betting opens. The spread is its guess of the home team's final margin. If the line is far off, sharp bettors bet the other side and the book loses money. So the model predicts the **home team's margin** (home points minus away points) using only what is known before tip off.

This framing drove the pipeline in four ways.

1. **Target.** A regression target in points, not "who wins". A book needs the size of the win, not only the winner. So I use linear regression.
2. **No leakage from the game itself.** The box score columns (FG%, 3P%, rebounds, assists) are only known after the game. They would make the model look great and be useless before tip off, so they are never used. Every feature is built with `shift(1)` inside each team and season, so a game only sees the games before it.
3. **Time based split, not random.** A book sets lines for a season that has not happened yet, using the seasons before it. The league also changes over time, for example the home edge drifts from season to season. A random split would mix games from every season into training and testing, so it would hide that drift and give a test score that is too optimistic. I train on 2012-13 to 2017-18, use 2018-19 to pick the regularisation strength and learning rates, and test once on 2021-22. The final models are refit on 2012-13 to 2018-19.
4. **Metric.** MAE in points, because each point of error is a point a bettor can use. The models are still fitted with squared error, because that is what linear regression minimises. Squared error punishes big misses more, which a book also cares about. I also report the share of games within 3 points and how often the sign picks the winner. In the dashboard a slider shows how many games the line misses by more than N points.

I left out 2019-20 and 2020-21. COVID stopped the season, the playoffs were played in a bubble with no home team, and most of 2020-21 had empty arenas. Home court advantage dropped from around 2.7 points to 0.9 points in 2020-21 (see the "Home court by season" tab). Training on those seasons would teach the model a home edge that does not exist with fans in the building. 2022-23 is only half a season in the data, so it is not used.

## Data preparation & feature engineering

**Cleaning.** Game IDs starting with 2 are regular season. I dropped preseason (bench players, not real strength), playoffs and play-in (different rotations, same teams over and over), 29 duplicate game IDs and games with no score.

**Team game log.** I turned each game into two rows, one per team, with the margin from that team's side. All features are computed per team and season on this log, using only earlier games.

The final features, all as home minus away or as a flag:

| Feature | What it is | Why |
|---|---|---|
| `rating_diff` | Season to date average margin, shrunk toward a prior | Main measure of team strength |
| `form_diff` | Average margin over the last 10 games | Catches injuries, trades and hot streaks the season average is slow to show |
| `rest_diff` | Days of rest, capped at 4 | Tired teams play worse |
| `home_b2b`, `away_b2b` | Second night of a back to back | Strongest fatigue signal |
| `altitude` | Home team is Denver or Utah | Both arenas are far above sea level |

**Shrinking the team rating.** After one game a team's average margin is just that game, which is mostly noise. So the rating is a weighted average of the team's results so far and a prior: `(n * avg + 10 * prior) / (n + 10)`. The prior is last season's average margin pulled halfway to 0, because rosters change over the summer. After 10 games the team's own results count as much as the prior. Teams with no history start at 0. I set K = 10 and the half weight in advance, then checked a few other values (below).

**Feature check on the validation season.** This uses Ridge, with MAE in points (lower is better). The validation season has only 1,230 games, so small differences can be luck. Each row shows the difference from the final set and its standard error, computed game by game. A difference smaller than about two standard errors is noise.

| Features | Val MAE | vs final set | Std error |
|---|---|---|---|
| Constant (home court only) | 11.365 | +1.191 | 0.148 |
| Rating, not shrunk | 10.436 | +0.262 | 0.057 |
| Rating, shrunk (K 10, prior 0.5) | 10.252 | +0.077 | 0.028 |
| Shrink check, K 5 | 10.277 | +0.102 | 0.029 |
| Shrink check, K 20 | 10.241 | +0.067 | 0.034 |
| Shrink check, prior weight 0 | 10.322 | +0.147 | 0.037 |
| Shrink check, prior weight 1 | 10.240 | +0.065 | 0.036 |
| + form | 10.235 | +0.061 | 0.024 |
| + rest and back to backs | 10.187 | +0.013 | 0.011 |
| + altitude (final set) | 10.175 | 0 | 0 |
| Final set + away road trip | 10.178 | +0.003 | 0.003 |
| Final set + team home edge | 10.181 | +0.007 | 0.008 |

What I take from it:

- **Shrinking clearly helps.** It is 0.18 points better than the raw season average, more than 3 standard errors.
- **The prior helps.** Using last season's margin as the prior beats shrinking toward 0 by about 0.08 points, which is about 2 standard errors.
- **The exact K and prior weight do not matter.** K 5, 10 or 20 and prior weights 0.5 or 1 are all within one standard error of each other. So I kept the values I set in advance and did not tune them.
- **Form and rest are real gains.** Form adds 0.017 and rest with back to backs adds 0.048.
- **Altitude is borderline.** It adds 0.012, which is about one standard error.

**What did not work.**

- **Away road trip.** This was the number of away games in a row for the visitor, to capture travel fatigue. It changed nothing (0.003 worse) and its coefficient came out at about 0, so I dropped it. My guess is that a long road trip only hurts through tired legs on back to backs, which the back to back flag already covers, but I did not test that.
- **Team specific home edge.** This was a team's home margin minus its overall margin. It was slightly worse, well inside noise. A team has only about 41 home games a season, so that number is too noisy to help.
- **Altitude is not a clean altitude effect.** It is just a flag for two specific teams, so it also picks up anything else about playing in Denver or Utah, like their crowds. With two teams I cannot separate those.

**Scaling.** Features are standardised with `StandardScaler`, fitted on training data only. This matters for the two PyTorch versions, because gradient descent is slow when one feature is in the tens and another is 0 or 1.

## Modeling: three implementations, one model

**Model.** Linear regression with an L2 penalty (Ridge), because the target is continuous. The three versions minimise the same thing. sklearn Ridge minimises `sum((y - Xw - b)^2) + alpha * sum(w^2)`. Dividing by n gives `mean((y - Xw - b)^2) + (alpha / n) * sum(w^2)`, which is the loss I wrote in both PyTorch versions. I wrote the penalty by hand instead of using `weight_decay`, so the bias is not penalised, the same as Ridge.

- **scikit-learn:** `Ridge(alpha)`, solved exactly.
- **Manual PyTorch:** raw tensors `w` and `b` with `requires_grad`, loss, `backward()`, then `w -= lr * w.grad` inside `torch.no_grad()` and zeroing the gradients. Full batch, 1,000 epochs.
- **Standard PyTorch:** `nn.Linear`, `nn.MSELoss` plus the penalty, `torch.optim.Adam`. Full batch, 1,000 epochs.

**Tuning.**

- **Alpha, picked on 2018-19.** Values from 0.1 to 1,000 gave almost the same validation MAE (10.174 to 10.196). With 7,379 training games and 6 features there is little to overfit, so regularisation barely matters. 100 was picked.
- **Learning rate, picked by training loss.** The learning rate's job is to reach the minimum of the training loss, so I picked it by the lowest final training loss, not by validation MAE. sklearn's exact minimum is 149.0555.
  - **Manual loop:** lr 0.1 and 0.5 reach it. lr 0.01 stops at 149.0557.
  - **Adam:** lr 0.1 reaches it. 0.01 stops at 149.40 and 0.001 at 166.08.
  - Both versions use 0.1.

**Results on the test season 2021-22 (1,230 games):**

| Model | MAE | RMSE | R2 | Within 3 pts | Picks winner |
|---|---|---|---|---|---|
| Naive baseline (home team by 2.7) | 12.212 | 15.337 | -0.004 | 12.1% | 54.4% |
| scikit-learn | 11.245 | 14.256 | 0.132 | 17.3% | 64.6% |
| Manual PyTorch | 11.245 | 14.256 | 0.132 | 17.3% | 64.6% |
| Standard PyTorch | 11.245 | 14.256 | 0.132 | 17.3% | 64.6% |

All three beat the baseline by about 1 point of MAE and pick the winner 65% of the time, against 54% for "home team always wins".

**Do the three agree?** Yes. All coefficients match to 3 decimals, and the largest gap between any two models on any test game is 0.000 points.

There was one thing worth explaining during tuning. With lr 0.01 the manual loop gets the training loss almost right (149.0557 vs 149.0555), but the rating weight is visibly off (4.833 vs 4.854). The reason is that `rating_diff` and `form_diff` have a correlation of 0.78. Many mixes of the two give almost the same loss, so the loss surface is very flat in that direction. A small learning rate barely moves along it in 1,000 steps. The loss looks finished while the weights are not. A larger learning rate (0.1) gets all the way there.

**Coefficients in real units** (sklearn):

- **Team rating:** every point of rating difference is worth 0.83 points of margin.
- **Back to backs:** an away team on a back to back is worth +1.3 points to the home team. A home team on a back to back costs it 1.2.
- **Denver or Utah:** a home game there is worth +1.8 points on top of the normal home edge.
- **Base home edge:** about 2.4 points.

## Limitations & next steps

- **The test season is harder than the past.** Margins in 2021-22 are more spread out (std 15.3 vs 13.2 to 14.4 before), with more blowouts. That is why test MAE (11.2) is worse than validation (10.2). Weighting recent seasons more would help a model follow changes in the league.
- **Home court shrank after COVID.** The model learned a home edge from 2012 to 2018, but 2021-22 only had 1.7 points. On average it favours the home team by 0.9 points too much. A fix is to estimate the home edge from the most recent season or two only, instead of from all the training years.
- **No player information.** The biggest swings in a line come from who is playing (stars resting, injuries, trades). The data has no injury or lineup information. A players table (`games_details.csv`) could build a "minutes missing" feature.
- **No comparison with real lines.** Without historical betting odds I cannot say how this compares to what books actually post. Adding closing lines would show if the model ever finds games where the market is wrong.
- **Linear only.** Effects like rest probably are not linear (3 days of rest is not three times one day). Binning rest into categories is a simple next step that still keeps a linear model.

## Generative AI use disclosure

I used Claude (Anthropic) during this assignment. It suggested dataset ideas, and I chose the NBA point spread framing from them. It wrote most of the code for loading the data, the training loops, saving the models and the Gradio dashboard. It also drafted this report from the results. I went through every decision in the pipeline and can explain each one.
