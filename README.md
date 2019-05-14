# AdvertisingDIA


1. Choose a product to advertise by means of digital tools. Provide a brief description.

2. Imagine 5 advertising sub-campaigns. Imagine an average daily budget/clicks curve (providing, for every value of daily budget, the number of daily clicks) aggregating the curves of three different classes of users. Notice that, in order to define the curves, it is necessary the definition of probability distributions. Provide a description of the three classes of users. Note: the definition of the classes of the users must be done by introducing features and different values for the features (e.g., gender, interests, age).

3. Be given a cumulative daily budget constraint. Be also given a discretisation for the daily budget values. Apply the Combinatorial-GP-TS to the aggregate curve (we are implicitly assuming that the bidding is performed automatically by the advertising platform) and report how the regret varies in time.

4. Focus on a single sub-campaign. Report the the average regression error of the GP as the number of samples increases. The regression error is the maximum error among all the possible arms.

5. Suppose to apply, the first day of every week, an algorithm to identify contexts and, therefore, to disaggregate the curves if doing that is the best we can do. And, if such an algorithm suggests disaggregating the curve at time t, then, from t on, keep such curves disaggregate. In order to disaggregate the curve, it is necessary to reason on the features and the values of the features. Apply the Combinatorial-GP-TS algorithm and show, in a plot, how the regret and the reward vary in time (also comparing the regret and the reward of these algorithms with those obtained when the algorithms are applied to the aggregate curve).
