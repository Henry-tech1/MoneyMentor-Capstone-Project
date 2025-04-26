# MoneyMentor: A Machine Learning Approach to Personal Finance
**By: [Henry Mwangi]**

## Introduction

### Why I Chose This Model

A common problem faced by almost everyone around the world is the struggle to manage their money effectively. This model aims to help its users understand their spending, identify saving opportunities, and make better financial decisions with intelligent help.

### What It Does in Simple Terms

It analyzes your income and expenses and gives suggestions, like where to cut back or how much to save monthly, tailored to your habits. Think of it like a smart budgeting friend who has your best financial interests at heart.

### Problem Statement

Most people lack financial literacy and access to professional financial advisors. This project tries to solve that by using machine learning to provide similar advice, based on a user’s actual spending patterns.

### Objective

The objective of this model is to empower individuals to manage their money more effectively by providing more personalized, accessible, and automated financial guidance. This model can analyze a user’s income, expenses, and financial habits to offer tailored recommendations. This will encourage better financial habits and promote long-term stability. By making financial planning more inclusive and affordable, such a model can help reduce debt, improve savings, and support overall financial well-being.

## Approach Taken

### 1. Data Collection

* **Used:** Sample data from Kaggle, representing income, expenses based on categorized spending (e.g., rent, groceries, transport), and user financial goals.
* **Dataset:** Shriyashjagtap (2023). *Indian Personal Finance and Spending Habits* \[Dataset]. Kaggle. Retrieved from [https://www.kaggle.com/datasets/shriyashjagtap/indian-personal-finance-and-spending-habits](https://www.kaggle.com/datasets/shriyashjagtap/indian-personal-finance-and-spending-habits)
* **Why it's suitable:**
    * It's relevant to personal finance habits.
    * Contains financial behaviors to draw insights.
    * Offers both numerical and categorical features for analysis.
* **Goal:** To give personalized advice like:
    * 'Consider increasing your savings rate'
    * 'Reduce eating out frequently'
* **How:** The model learns different user patterns in the data to make these suggestions.

### 2. Data Preprocessing

After successfully identifying my ideal dataset:

* I imported all necessary libraries.
* Loaded my data into my DataFrame using Pandas.
* To gain an initial understanding of the dataset's structure and content, I executed several commands to explore the data. The outputs of these commands provide key insights into:
    * **The data's dimensions:** (20,000 rows & 27 columns)
    * **Data types:** (float: 23, integer: 2, object: 2)
    * **Sample entries:** (first and last five rows)
    * **Summary statistics:** (mean, standard deviation, min, max, quartiles) for numerical columns
    * **Missing values:** (sum of missing values = 0)

### 3. Exploratory Data Analysis

* **Income Distribution:**
    * Plotted a histogram to visualize the distribution of income in my dataset.
    * **Observations:** The histogram shows a highly skewed income distribution, where the majority of the population earns very low incomes, while a small fraction earns significantly higher incomes. This indicates substantial income inequality, with most people clustered at the lower end of the income scale and a long tail representing a few high earners.

* **Desired Savings vs. Actual Disposable Income:**
    * Plotted a scatterplot to check desired savings vs. actual disposable income.
    * **Observations:**
        * **Positive Correlation:** There is a clear positive correlation between desired savings and disposable income. As disposable income increases, desired savings also tend to increase.
        * **Clustering:** Most data points are clustered at lower values of disposable income (below 100,000) and desired savings (below 50,000), indicating that many individuals have lower incomes and savings.
        * **Outliers:** There are a few outliers with significantly higher disposable income (above 200,000) and desired savings (above 150,000), suggesting some individuals have much higher financial capacities.
        * **Savings Percentage:** The color gradient shows that individuals with higher disposable incomes tend to have higher desired savings percentages, though this is not universally true.
    * **Interpretation:** The plot suggests that as people earn more, they generally aim to save more. However, the desired savings percentage varies, indicating that saving behavior is influenced by factors beyond income, such as financial goals, expenses, or personal preferences. This visualization is useful for understanding saving trends across different income levels and can inform financial planning or policy-making.

* **Income Distribution by City Tier and Outliers:**
    * Plotted a boxplot to check:
        * **a.) Income distribution by city tier**
            * **Description:** This plot shows the distribution of income across three city tiers (Tier 1, Tier 2, Tier 3).
            * **Observations:**
                * **Tier 1:** Income is concentrated at lower values, with a few outliers reaching higher income levels.
                * **Tier 2:** Similar to Tier 1, but with slightly more variation and higher outliers.
                * **Tier 3:** Displays a wider spread of income, with some extremely high outliers.
            * **Interpretation:** Higher-tier cities tend to have more income disparity, with a few individuals earning significantly more than the majority.
        * **b.) Outliers in Income**
            * **Description:** This plot highlights the outliers in the income distribution across all city tiers.
            * **Observations:** Most data points are clustered at lower income levels, with a long tail of outliers extending to higher incomes. Outliers are more pronounced in higher-tier cities.
            * **Interpretation:** There are significant income disparities, with a small number of individuals earning much more than the average.

* **Average Monthly Spending by Categories:**
    * Plotted a Piechart to check average monthly spending by categories.
    * **Description:** This chart shows the average monthly spending across various categories.
    * **Categories and Percentages:**
        * Rent: 29.5%
        * Groceries: 16.8%
        * Loan Repayment: 6.6%
        * Transport: 8.7%
        * Eating Out: 4.7%
        * Entertainment: 4.7%
        * Utilities: 8.1%
        * Healthcare: 5.4%
        * Education: 8.1%
        * Insurance: 4.7%
        * Miscellaneous: 2.7%
    * **Interpretation:** Rent is the largest expense, followed by groceries and transport. This indicates that basic necessities and housing are the primary financial burdens.

* **Relationship Between Income and Total Expenses:**
    * Plotted a bar graph to visualize the relationship between Income and Total expenses.
    * **Description:** This chart compares income and total expenses for a sample of individuals.
    * **Observations:** Income generally exceeds expenses, but there are cases where expenses are close to or exceed income. There is significant variability in both income and expenses across individuals.
    * **Interpretation:** While most individuals manage to keep expenses below income, some struggle with expenses matching or exceeding their earnings.

* **Correlation Heatmap:**
    * Plotted a heatmap to show the correlation between different columns in my data.
    * **Description:** This matrix shows the correlation coefficients between various financial and demographic variables.
    * **Key Observations:**
        * Income has a strong positive correlation with Desired Savings (0.94) and Disposable Income (0.89).
        * Rent and Loan Repayment have moderate correlations with income (0.42 and 0.45, respectively).
        * Age and Dependents show weak correlations with most financial variables.
        * Total Expenses are strongly correlated with income (0.98).
    * **Interpretation:** Income is a strong predictor of savings and expenses. Rent and loan repayments are significant factors influencing financial behavior, while age and dependents have less impact.

### 4. Data Analysis

* **Actual vs. Desired Savings Analysis:**
    * To evaluate individuals’ saving behavior, I calculated the Actual Savings for each person by subtracting all recorded expenses from their total income. The formula used was:
        $\text{Actual Savings} = \text{Income} - (\text{Rent} + \text{Loan Repayment} + \text{Insurance} + \text{Groceries} + \text{Transport} + \text{Eating Out} + \text{Entertainment} + \text{Utilities} + \text{Healthcare} + \text{Education} + \text{Miscellaneous})$
    * Next, I computed the Savings Gap, which measures the shortfall (or surplus) between what individuals aimed to save and what they actually saved:
        $\text{Savings Gap} = \text{Desired Savings} - \text{Actual Savings}$
    * A positive value indicates that the person saved less than they intended (a shortfall), while a negative value suggests they exceeded their saving goal.
    * To visualize this, I plotted a histogram of the Savings Gap distribution. The red dashed vertical line at zero represents the point where actual and desired savings match. This helps to clearly illustrate how many individuals are under-saving or over-saving relative to their goals.

* **Correlation Analysis with Disposable Income:**
    * To identify the factors most closely related to Disposable Income, I computed the correlation matrix for all numerical variables in the dataset.
    * I then visualized the correlations with Disposable Income using a heatmap. The variables were sorted in descending order to highlight the strongest positive and negative correlations.
    * This visualization helps reveal which financial components (such as income sources or expenses) have the greatest impact on how much disposable income an individual retains after all necessary spending. Strong positive correlations suggest areas that may boost disposable income, while strong negative correlations indicate major cost drivers.
    * **Summary of Findings:** The heatmap shows that Income has a strong positive correlation with Disposable Income, as expected. On the other hand, categories like Rent, Loan Repayment, and Groceries show significant negative correlations, meaning these expenses substantially reduce what’s left as disposable income. These insights can help prioritize areas for budgeting or financial planning.

* **Overspending Analysis by Category:**
    * To pinpoint where individuals tend to overspend, I compared actual spending in various discretionary categories to their corresponding potential savings targets. These categories included Groceries, Transport, Eating Out, Entertainment, Utilities, Healthcare, Education, and Miscellaneous.
    * For each category, I created a new variable called `Overspending_`, calculated as:
        $\text{Overspending} = \text{Actual Spending} - \text{Potential Savings}$
    * This allowed me to measure how much more people are spending than they ideally should in each category.
    * I then calculated the average overspending per category across the dataset and visualized the results using a bar chart. The horizontal gray line at zero marks the break-even point — bars above it indicate overspending, while any (if present) below would represent underspending.
    * **Summary of Findings:** The bar chart reveals that categories like Eating Out, Entertainment, and Groceries are the most common areas of overspending. These discretionary expenses represent opportunities where individuals could potentially cut back to improve savings. On the other hand, categories with lower overspending (or none at all) suggest better alignment with budgeted expectations.

### 5. Feature Engineering

* **Spending Categorization and Income Ratios:**
    * To better understand how individuals allocate their income, I grouped expenses into two main categories:
        * **Essential Expenses:** These include necessary costs such as Groceries, Utilities, Healthcare, and Education.
        * **Discretionary Expenses:** These cover non-essential or lifestyle-related spending, including Eating Out, Entertainment, Miscellaneous, and Transport.
    * Using these groupings, I created several spending-to-income ratio features to assess financial balance and budgeting behavior:
        * `Rent_to_Income`: Proportion of income spent on rent
        * `Loan_to_Income`: Share of income going toward loan repayments
        * `Essentials_to_Income`: Fraction of income used on essential needs
        * `Discretionary_to_Income`: Portion of income spent on non-essentials
    * These ratios provide a clearer picture of financial habits, affordability, and potential stress points in a person's budget.
    * **Summary of Insights:** These features serve as key indicators of financial health. Higher values in ratios like `Discretionary_to_Income` may suggest overspending, while elevated `Essentials_to_Income` or `Loan_to_Income` can indicate tighter financial constraints. These insights are valuable for identifying budget imbalances and tailoring financial advice or interventions.

* **Savings Behavior Metrics:**
    * To gain deeper insight into individual saving habits and efficiency, I engineered several key savings-related metrics:
        * `Savings Rate`: Measures what percentage of income is actually saved.
            $\text{Savings\_Rate} = \frac{\text{Actual\_Savings}}{\text{Income}}$
        * `Savings vs Desired`: Compares actual savings to desired savings, helping assess how well individuals meet their saving goals.
            $\text{Savings\_vs\_Desired} = \frac{\text{Actual\_Savings}}{\text{Desired\_Savings}}$
        * `Disposable Utilization`: Reflects how effectively disposable income is being saved rather than spent.
            $\text{Disposable\_Utilization} = \frac{\text{Actual\_Savings}}{\text{Disposable\_Income}}$
    * To prevent division errors, a small constant ($1e-6$) was added to the denominators where necessary.
    * **Summary of Insights:** These metrics provide a more nuanced view of financial discipline. A high `Savings Rate` indicates efficient saving, while a `Savings_vs_Desired` value near or above 1 suggests individuals are meeting or exceeding their goals. `Disposable Utilization` helps identify whether surplus income is being wisely allocated. Together, these features support more personalized financial assessments and recommendations.

* **Adaptive Savings Goal Adjustment:**
    * To make savings goals more realistic and responsive to actual financial behavior, I introduced a dynamic feature: `Suggested Savings Adjustment`. This metric recalculates the user's savings goal based on their current saving performance:
        $\text{Suggested\_Savings\_Adjustment} = \text{Desired\_Savings} + (\text{Savings\_vs\_Desired} - 1) \times \text{Disposable\_Income}$
        * **a.)** If a user consistently under-saves ($\text{Savings\_vs\_Desired} < 1$), the adjustment will reduce their goal to something more attainable.
        * **b.)** If they consistently over-saves ($\text{Savings\_vs\_Desired} > 1$), the adjustment will increase their goal, encouraging more ambitious financial planning.
    * This approach personalizes financial targets, making them more data-driven and aligned with real spending behavior.
    * **Summary of Insights:** The `Suggested Savings Adjustment` helps bridge the gap between financial intention and actual behavior. By adapting the savings goal based on performance, users can set more achievable and motivating targets, leading to better long-term financial habits.

* **Total Overspending & Financial Alert System:**
    * To assess the overall extent of budget overruns, I calculated a `Total Overspending` metric by summing all individual overspending amounts across discretionary categories:
        $\text{Total\_Overspending} = \text{Sum of Overspending across Groceries, Transport, Eating Out, etc.}$
    * Then, to contextualize this figure, I calculated the `Overspending-to-Income Ratio` to see what portion of a person's income is lost to excess spending:
        $\text{Overspending\_to\_Income} = \frac{\text{Total\_Overspending}}{\text{Income}}$
    * To identify individuals at higher financial risk, I flagged users whose `Total Overspending` exceeded the 75th percentile (i.e., the top 25% of overspenders). This created a binary `Overspending Alert`:
        $\text{Overspending\_Alert} = \text{True if Total\_Overspending} > 75\text{th percentile threshold}$
    * **Summary of Insights:** The alert system highlights individuals who may be at risk of financial instability due to consistently high levels of overspending. By analyzing the proportion of users flagged (`value_counts(normalize=True)`), we can quantify how common this issue is and potentially target financial guidance toward the most at-risk segment. For example, if a significant percentage of users fall into the alert category, it underscores a broader pattern of poor budgeting or unrealistic saving expectations.

### 6. Data Preprocessing for Modeling

* **Feature Preparation and Transformation:**
    * To prepare the dataset for machine learning, I first organized the features into two main groups:
        * **Numerical Features:** These include continuous and ratio-based variables such as Income, Age, Savings Rate, Total Overspending, and various spending metrics.
        * **Categorical Features:** These are non-numeric variables, specifically Occupation and City Tier, which may influence spending or saving behavior.
    * To ensure that all features are in a suitable format for modeling, I applied the following transformations:
        * **Numerical Transformation:** Standardized all numerical features using `StandardScaler`, which normalizes values to have a mean of 0 and standard deviation of 1. This helps models converge faster and perform better.
        * **Categorical Transformation:** Applied `One-Hot Encoding` to categorical features, converting them into binary columns while handling any unknown categories gracefully with `handle_unknown='ignore'`.
    * These transformations were combined into a single `ColumnTransformer` object called `preprocessor`, which ensures a clean, consistent preprocessing pipeline for any downstream modeling tasks (e.g., regression or classification).

### 7. Modeling

* **Spending Categorization and Behavioral Labeling (Continued):**
    * I grouped transaction columns into four main categories:
        * **Fixed:** Rent, Loan Repayment, Insurance
        * **Essential:** Groceries, Transport, Utilities, Healthcare, Education
        * **Discretionary:** Eating Out, Entertainment, Miscellaneous
        * **Savings:** Actual Savings
    * Each transaction was assigned a default label based on its type.
    * I then enriched this with behavioral tagging by merging overspending data and calculating whether the user was:
        * **Overspending:** Spending more than 30% of income on an essential or discretionary item
        * **Undersaving:** Saving less than their declared goal
        * **Fixed/Essential/Discretionary/Savings:** When spending behavior aligned with expectations
    * This resulted in a new column, `Final_Label`, which captures spending behavior more meaningfully.

    * **Summary of Insights:** I used three visualizations to explore this behavioral labeling:

        * **a. Count Plot:** Shows how many transactions fall into each behavior category:
            * **Observations:**
                * Essential: Has the highest number of transactions, close to 100,000.
                * Fixed: Approximately 60,000 transactions.
                * Discretionary: Around 50,000 transactions.
                * Savings: About 20,000 transactions.
                * Overspending: Very few transactions, almost negligible.
            * **Interpretation:** Essential expenses are the most frequent, indicating regular spending on necessities. Savings and overspending are less common.

        * **b. Pie Chart:** Displays the proportion of transaction types & helps illustrate how much of overall behavior is accounted for by overspending or undersaving.
            * **Categories and Percentages:**
                * Essential: 41.7%
                * Fixed: 25.0%
                * Discretionary: 25.0%
                * Savings: 8.2%
                * Undersaving: 0.2%
            * **Interpretation:** Essential transactions make up the largest portion of spending, followed equally by fixed and discretionary expenses. Savings constitute a smaller portion, and undersaving is minimal.

        * **c. Bar Plot of Average Amount:** Compares the average amount spent or saved per label.
            * **Observations:**
                * Savings: Has the highest average amount spent, exceeding 10,000.
                * Fixed: Average amount spent is around 4,000.
                * Essential: Average amount spent is slightly above 2,000.
                * Discretionary: Average amount spent is below 2,000.
                * Overspending: Not represented, indicating negligible or no data.
            * **Interpretation:** Savings transactions involve the highest average amounts, suggesting larger deposits or investments. Fixed expenses also involve significant amounts, while essential and discretionary spending are lower.

        * **Summary of Interpretations:**
            * **Transaction Count:** Essential expenses are the most frequent, indicating regular spending on necessities. Savings and overspending are less common.
            * **Proportion of Transactions:** Essential spending dominates, with fixed and discretionary spending being equally significant. Savings are a smaller but notable portion.
            * **Average Amount Spent:** Savings involve the highest average amounts, indicating larger financial commitments. Fixed expenses also involve substantial amounts, while essential and discretionary spending are lower.

* **User Segmentation via Clustering:**
    * To uncover distinct groups of users based on financial patterns, I applied K-Means Clustering to the preprocessed dataset. The process involved several steps:

        * **a. Preprocessing:** Categorical features (Occupation and City Tier) were one-hot encoded. All features were then standardized using `StandardScaler` to ensure uniform influence on clustering.

        * **b. Choosing the Optimal Number of Clusters (k):** I used the Elbow Method by plotting the Sum of Squared Errors (SSE) for k = 1 to k = 10. The point where the SSE curve starts to flatten indicates the optimal k. In this case, k = 2 was chosen (adjustable based on the actual plot).
            * **Axes:**
                * X-axis (Number of clusters, k): Ranges from 1 to 10.
                * Y-axis (SSE - Sum of Squared Errors): Measures the total within-cluster variation.
            * **Observations:**
                * The SSE decreases sharply as the number of clusters increases from 1 to 3.
                * The rate of decrease slows down significantly after 3 clusters, forming an "elbow" shape.
                * Beyond 3 clusters, the SSE continues to decrease but at a much slower rate.
            * **Interpretation:** The optimal number of clusters (k) is likely 3, as this is where the elbow occurs. Adding more clusters beyond this point yields diminishing returns in terms of reducing SSE.

        * **c. Model Fitting:** Applied KMeans with the optimal cluster number. Assigned each user a `Cluster` label reflecting their financial profile group.

        * **d. Cluster Profiles:** Calculated the mean values of all numeric features per cluster to understand financial behaviors like income, savings rate, overspending, etc. This forms a data-driven financial persona for each cluster.

        * **e. Visualization with PCA:** Used Principal Component Analysis (PCA) to reduce high-dimensional data to two dimensions. Plotted the clusters using a scatter plot to show separation and overlap visually.
            * **Axes:**
                * X-axis (PCA1): First principal component.
                * Y-axis (PCA2): Second principal component.
            * **Clusters:**
                * Cluster 0 (Blue): Concentrated around the origin (0,0) with lower values on both PCA1 and PCA2.
                * Cluster 1 (Orange): Spread out more widely, with higher values on PCA1 and varying values on PCA2.
            * **Observations:**
                * Cluster 0 is densely packed, indicating similar financial habits among its members.
                * Cluster 1 is more dispersed, suggesting greater variability in financial habits.
                * There are a few outliers in Cluster 1 with very high PCA1 values.
            * **Interpretation:** The PCA plot shows two distinct groups of users based on their financial habits. Cluster 0 represents users with more uniform habits, while Cluster 1 includes users with more diverse behaviors. The outliers in Cluster 1 may indicate extreme or unique financial habits.

        * **Summary of Insights:** The analysis successfully grouped users into distinct financial profiles based on key features like income, savings habits, and spending behavior. For example:
            * One cluster might represent budget-conscious savers with high savings rates and low discretionary spending.
            * Another could consist of financially stressed individuals with high overspending and low actual savings.
            * The PCA scatterplot helps visualize how clearly these clusters separate in terms of financial behavior, which is valuable for targeted financial advice, app personalization, or policy design.

* **Clustering Evaluation Using Silhouette Score:**
    * To assess the effectiveness of the K-Means clustering and fine-tune the number of clusters, I used the Silhouette Score — a metric that quantifies how well each data point fits within its assigned cluster compared to others. The score ranges from -1 to 1:
        * Closer to 1: Well-clustered, clear separation between groups
        * Around 0: Clusters are overlapping or ambiguous
        * Negative values: Points may be misclassified
    * **Process:** Calculated the Silhouette Score for cluster sizes from k = 2 to 10. Printed each score to monitor clustering performance across different k values. Identified the optimal number of clusters as the k value with the highest Silhouette Score.
    * **Example output:**
        ```
        Optimal number of clusters (k): 3
        Highest Silhouette Score: 0.47
        ```
    * **Summary of Insights:** The Silhouette Score provides a more reliable metric than SSE (used in the Elbow Method) because it considers both intra-cluster cohesion and inter-cluster separation. Selecting the k with the highest score ensures the most meaningful user segmentation, leading to more actionable insights and better modeling outcomes.

* **Personalized Nudging System:**
    * To provide users with tailored financial guidance, I implemented a nudge generation system that flags risky or suboptimal behaviors based on individual financial metrics. Each user is evaluated against several behavioral rules, and if any are triggered, a nudge message is assigned.
    * **Key Nudging Rules:**
        * ⚠️ **Close to Overspending:** Triggered when disposable utilization is between 80–100%.
        * ❌ **Overspending Alert:** Triggered if a user is in the top 25% of total overspending.
        * 💰 **Below Savings Target:** Triggered when actual savings fall significantly short of desired savings.
        * 🛍️ **High Discretionary Spending:** Triggered if discretionary spending exceeds 30% of income.
        * 🏠 **High Rent Burden:** Rent consumes over 40% of income.
        * 📉 **High Loan Repayment Burden:** Loans consume over 30% of income.
    * Each nudge is designed to be concise and actionable, encouraging users to reflect on their financial choices for the week.

* **Nudge Priority Classification:**
    * To prioritize user attention, nudges are classified into three urgency levels:
        * **a. High Priority:** Includes any overspending alerts
        * **b. Medium Priority:** Includes warnings, such as nearing spending limits
        * **c. Low Priority:** General observations or suggestions
        * **d. None:** No issues flagged
    * **Example Output:**
        ```
        Nudges: ["❌ You’ve overspent this week.", "🛍️ Discretionary spending is quite high this week."]
        Nudge_Priority: High
        ```
    * **Summary of Insights:** This behavior-driven approach adds a personal finance coaching layer to the data, offering users proactive advice based on their current habits. It can also be used to:
        * Power in-app notifications or weekly summaries
        * Identify users for targeted budgeting tips
        * Segment audiences for financial literacy campaigns

* **Modeling Suggested Savings Adjustment:**
    * To provide data-driven guidance on how users should adjust their savings goals, I trained a Random Forest Regressor to predict the `Suggested_Savings_Adjustment` based on various financial features.
    * **Steps Taken:**
        * **Target Variable:** `Suggested_Savings_Adjustment` was selected as the prediction target. It was scaled to millions to improve numerical stability and model performance.
        * **Feature Preparation:** Used a one-hot encoded and scaled dataset (`df_encoded`) with the target column excluded from the features (X).
        * **Split Data:** Split the data into training and test sets (80/20 split).
        * **Model Training:** A Random Forest Regressor with 100 trees (`n_estimators=100`) was trained on the data.
        * **Prediction & Evaluation:** The model’s predictions were rescaled back to original currency values for interpretation. Evaluated performance using:
            * **Mean Absolute Error (MAE):** Measures average error in dollars.
            * **R² Score:** Indicates the proportion of variance explained by the model (closer to 1 is better).
    * **Output:**
        ```
        MAE: $3,240.58
        R² Score: 0.79
        ```
    * **Summary of Insights:** The model performs well, with a low MAE and a high R² score, suggesting it can reliably estimate appropriate savings targets for users based on their unique financial situations. This predictive capability could be integrated into a personal finance app to suggest customized savings plans in real time.

### 8. Analysis and Visualizations Summary

In this analysis, we performed clustering on users based on their financial behaviors and used predictive modeling to provide actionable insights (nudges) to help users optimize their savings.

* **a. Clustering Financial Profiles:**
    * We applied K-Means clustering to segment users into distinct financial profiles based on their spending and income. The elbow method was used to determine the optimal number of clusters (in this case, 3).
    * After clustering, we visualized the profiles using Principal Component Analysis (PCA), which allowed us to reduce the dimensionality and plot the data in a 2D space, showing clear clusters of users with similar financial habits.

    * **Cluster Visualization:** The scatter plot shows clusters of users, with each cluster represented by a different color. The 3D scatter plot also visualizes the relationship between financial clusters and the suggested savings adjustment, providing deeper insight into the financial profiles of users.

* **b. Actionable Nudges for Users:**
    * To guide users in improving their financial habits, we created actionable nudges based on their spending and savings behaviors. These nudges are designed to alert users when they are close to overspending, have already overspent, or are not meeting their savings goals.
    * **For example:**
        * ⚠️: "You’re close to overspending your weekly budget."
        * ❌: "You’ve overspent this week. Consider reducing non-essentials."
        * 💰: "You’re below your savings target. Try to reduce spending."
        * 🛍️: "Discretionary spending is quite high this week."

* **c. Nudge Prioritization:**
    * We classified each user's nudge priority into categories:
        * High priority nudges (e.g., overspending alerts).
        * Medium priority nudges (e.g., warnings about being close to overspending).
        * Low priority nudges (e.g., general savings suggestions).
    * These priorities help to focus on the most pressing financial issues and guide users toward immediate actions.

    * **Nudge Priority Distribution:** A bar chart visualizes the distribution of nudge priorities across the user base, which can help us see how many users need urgent financial advice (high priority).

* **d. Savings Adjustment Prediction Model:**
    * A Random Forest Regressor was used to predict the optimal savings adjustment for each user based on their financial data. The Mean Absolute Error (MAE) and R² Score were calculated to assess the model's performance:
        * **MAE:** Measures the average difference between predicted and actual savings adjustments.
        * **R² Score:** Shows how well the model explains the variance in the savings adjustment.

    * **Savings Adjustment Prediction Performance:** We visualized the predicted vs. actual savings adjustment using a scatter plot. The closer the points are to the red line, the better the predictions.

* **e. Visual Insights:**

    * **Cluster Profiles:** By clustering users and analyzing their financial behavior, we can suggest more personalized savings goals.
    * **Savings Adjustment Visualization:** The scatter plot and 3D cluster plots illustrate how well the model is capturing the financial profiles and suggesting optimal savings.

* **Detailed explanation and summary of each visualization:**

    * **Elbow Method for Optimal k:**
        * **Type:** Line chart.
        * **Axes:**
            * X-axis (Number of clusters, k): Ranges from 1 to 10.
            * Y-axis (SSE - Sum of Squared Errors): Measures the total within-cluster variation.
        * **Observations:**
            * The SSE decreases sharply as the number of clusters increases from 1 to 3.
            * The rate of decrease slows down significantly after 3 clusters, forming an "elbow" shape.
            * Beyond 3 clusters, the SSE continues to decrease but at a much slower rate.
        * **Interpretation:** The optimal number of clusters (k) is likely 3, as this is where the elbow occurs. Adding more clusters beyond this point yields diminishing returns in terms of reducing SSE.

    * **Clusters of Users by Financial Habits:**
        * **Type:** Scatter plot with PCA (Principal Component Analysis).
        * **Description:** This plot visualizes users clustered by their financial habits, projected onto two principal components (PCA1 and PCA2).
        * **Axes:**
            * X-axis (PCA1): First principal component.
            * Y-axis (PCA2): Second principal component.
        * **Clusters:**
            * Cluster 0 (Blue): Concentrated around the origin (0,0) with lower values on both PCA1 and PCA2.
            * Cluster 1 (Orange): Spread out more widely, with higher values on PCA1 and varying values on PCA2.
            * Cluster 2 (Green): Positioned between the other two clusters, with moderate values on both PCA1 and PCA2.
        * **Observations:**
            * Cluster 0 is densely packed, indicating similar financial habits among its members.
            * Cluster 1 is more dispersed, suggesting greater variability in financial habits.
            * Cluster 2 represents a middle ground in terms of the principal components.
        * **Interpretation:** The PCA plot effectively separates users into distinct financial behavior groups, allowing for targeted analysis and recommendations for each segment.

    * **3D Scatter Plot of Clusters and Suggested Savings Adjustment:**
        * **Type:** 3D Scatter Plot.
        * **Axes:**
            * X-axis (PCA1): First principal component.
            * Y-axis (PCA2): Second principal component.
            * Z-axis (Suggested_Savings_Adjustment): Scaled suggested savings adjustment.
        * **Clusters:** Colored points representing the three identified clusters.
        * **Observations:** The plot shows the distribution of suggested savings adjustments across the different financial clusters in the reduced PCA space. This helps visualize if certain financial profiles (clusters) tend to have higher or lower suggested savings adjustments.
        * **Interpretation:** This visualization provides a deeper understanding of how the model's suggested savings adjustments relate to the underlying financial behavior clusters. It can reveal patterns where certain types of users (based on their cluster) are recommended more significant adjustments than others.

    * **Predicted vs. Actual Suggested Savings Adjustment:**
        * **Type:** Scatter Plot.
        * **Axes:**
            * X-axis (Actual Suggested Savings Adjustment): The true calculated values.
            * Y-axis (Predicted Suggested Savings Adjustment): The values predicted by the Random Forest model.
        * **Observations:** The points are clustered relatively close to the diagonal red line.
        * **Interpretation:** The proximity of the points to the diagonal line indicates a good performance of the Random Forest Regressor in predicting the suggested savings adjustment. The model's predictions are generally close to the actual calculated values.

    * **Distribution of Nudge Priorities:**
        * **Type:** Bar Chart.
        * **Axes:**
            * X-axis (Nudge Priority): Categories (High, Medium, Low).
            * Y-axis (Number of Users): Count of users in each priority category.
        * **Observations:** The bar chart shows the number of users falling into each nudge priority category. This allows us to understand the distribution of financial risk and the prevalence of different behavioral patterns within the user base.
        * **Interpretation:** The distribution of nudge priorities highlights the segments of the user population that require more immediate attention (High Priority) versus those who might benefit from more general guidance (Low Priority). This is crucial for resource allocation and the design of targeted interventions.

### 9. Conclusion and Future Work

* **Summary of Findings:** This project demonstrates the potential of machine learning to provide personalized financial guidance. By analyzing user spending patterns, we can identify distinct financial profiles, predict optimal savings adjustments, and deliver timely nudges to encourage better financial habits. The clustering analysis reveals diverse user behaviors, while the predictive model for savings adjustment offers a data-driven approach to goal setting. The nudge system provides an immediate and actionable layer of financial coaching.

* **Limitations:**
    * The analysis is based on a specific dataset, and the findings might not be universally applicable.
    * The model's performance is dependent on the quality and representativeness of the input data.
    * The behavioral nudges are rule-based and could be further enhanced with more sophisticated machine learning techniques.
    * The clustering relies on K-Means, which assumes spherical clusters and may not capture more complex user groupings.

* **Future Work:**
    * **Integration with Real-Time Data:** Connecting the model to users' actual financial accounts for real-time analysis and more dynamic recommendations.
    * **Advanced Modeling Techniques:** Exploring more sophisticated machine learning models for prediction (e.g., time series analysis for spending forecasting) and clustering (e.g., DBSCAN, hierarchical clustering).
    * **Personalized Nudge Optimization:** Using reinforcement learning to optimize the timing and content of nudges for maximum impact.
    * **Incorporating External Factors:** Integrating macroeconomic indicators or user-specific financial goals to provide more context-aware advice.
    * **Developing a User Interface:** Building a user-friendly application or interface to deliver these insights and recommendations directly to individuals.
    * **Evaluating Long-Term Impact:** Conducting studies to assess the long-term effectiveness of the nudging system and personalized recommendations on users' financial well-being.

### 10. Acknowledgements

I would like to thank Shriyashjagtap for making the "Indian Personal Finance and Spending Habits" dataset publicly available on Kaggle. This project would not have been possible without this valuable resource.

### 11. Contact

## Questions?

Feel free to open an issue or reach out to mwangihenry018@gmail.com

## Final Note

If you find this project helpful, please give it a ⭐️! Contributions and feedback are welcome.
