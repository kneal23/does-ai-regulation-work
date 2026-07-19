# %% [markdown]
# # Does AI Regulation Actually Work? A Trust & Safety Analysis
#
# This project uses five linked datasets tracking global GenAI adoption,
# harm incidents, policy activity, model releases, and public sentiment
# from 2022-2025 to ask a question that matters a lot in AI governance
# circles but doesn't get asked directly very often: **does regulation
# actually change anything, or is a lot of it just activity without
# results?**
#
# **Questions I'm answering, in order:**
# 1. Does harm (incidents) tend to happen *before* policy responds, or does
#    policy ever get ahead of it?
# 2. Do specific policy events — deepfake laws specifically — produce a
#    measurable before/after change in related incidents?
# 3. Does a wave of image/video-generation model releases predict a rise
#    in deepfake-related incidents afterward?
# 4. Which countries show "regulation theater" — a lot of policy activity
#    with no accompanying improvement in public trust?
#
# **Data note:** this project uses five separate CSVs that need to be
# merged and reshaped to answer these questions:
# - `gaaii_country_year.csv` — country-year panel (adoption, regulation,
#   trust scores)
# - `gaaii_incidents.csv` — individual harm incidents (needs to be rolled
#   up to country-quarter level before I can compare it against policy
#   timing)
# - `gaaii_policy_events.csv` — individual policy/regulatory events
# - `gaaii_model_releases.csv` — individual AI model releases
# - `gaaii_survey_microdata.csv` — individual survey respondents (not used
#   directly in these four questions, but available for follow-up work)

# %% [markdown]
# ---
# ## Step 1: Import the modules

# %%
# Import the modules
import pandas as pd

# %% [markdown]
# ## Step 2: Read all five CSVs into DataFrames

# %%
# Read each CSV file into its own DataFrame
country_year_df = pd.read_csv('gaaii_country_year.csv')
incidents_df = pd.read_csv('gaaii_incidents.csv')
policy_df = pd.read_csv('gaaii_policy_events.csv')
model_releases_df = pd.read_csv('gaaii_model_releases.csv')
survey_df = pd.read_csv('gaaii_survey_microdata.csv')

# %% [markdown]
# ## Step 3: Check the shape and missing values for each table

# %%
# Check shape and nulls for each table in one pass
for name, df in [('country_year', country_year_df), ('incidents', incidents_df),
                  ('policy_events', policy_df), ('model_releases', model_releases_df),
                  ('survey_microdata', survey_df)]:
    print(f"{name}: {df.shape}")
    nulls = df.isnull().sum()
    nulls = nulls[nulls > 0]
    if len(nulls) > 0:
        print(f"  Columns with missing values: {dict(nulls)}")
    print()

# %% [markdown]
# **Answer:** Most tables are complete. `incidents` is missing
# `financial_damage_usd` for some rows, `model_releases` is missing
# `parameters_bn` and `huggingface_downloads_mn` for some models (likely
# proprietary models that don't disclose these), and `survey_microdata` is
# missing `primary_use_case` for some respondents (likely people who don't
# use GenAI regularly). None of these missing columns are ones I need for
# the four questions in this project, so no cleanup is required before
# moving on.

# %% [markdown]
# ---
# ## Groundwork: converting quarters into a single sortable number
#
# The `incidents` and `policy_events` tables both have a `year` column and
# a `quarter` column (like "Q3"). To compare timing across quarters — like
# "2 quarters before this event" — I need a single number that increases
# in order over time. I'm building this as `year * 4 + quarter_number`,
# which keeps every quarter in the correct chronological order.

# %% [markdown]
# ## Step 4: Add a year_quarter column to incidents and policy_events

# %%
# Turn "Q1", "Q2", etc. into plain numbers (1, 2, 3, 4)
incidents_df['quarter_num'] = incidents_df['quarter'].str.replace('Q', '').astype(int)
policy_df['quarter_num'] = policy_df['quarter'].str.replace('Q', '').astype(int)

# Combine year and quarter into one sortable number
incidents_df['year_quarter'] = incidents_df['year'] * 4 + incidents_df['quarter_num']
policy_df['year_quarter'] = policy_df['year'] * 4 + policy_df['quarter_num']

# Review the result
incidents_df[['year', 'quarter', 'year_quarter']].drop_duplicates().sort_values('year_quarter').head(8)

# %% [markdown]
# ## Step 5: Build the country-quarter incident rollup
#
# `incidents` has one row per individual incident. To compare it against
# policy timing, I need it rolled up to one row per country per quarter,
# with a count of how many incidents happened.

# %%
# Count incidents per country per quarter
incident_rollup = incidents_df.groupby(['country', 'year_quarter']).size().reset_index(name='incident_count')

# Review the rollup
incident_rollup.head()

# %% [markdown]
# ---
# ## Question 1: Does harm tend to happen before policy, or after?
#
# My plan: for every policy event in the dataset, look at how many
# incidents happened in that same country in the 2 quarters *before* the
# event, versus the 2 quarters *after*. If incidents are consistently
# higher before a policy event than after, that's a "reactive" pattern —
# policy responds to a spike in harm. If there's no consistent pattern,
# that tells me policy timing isn't clearly tied to harm at all.

# %% [markdown]
# ## Step 6: Loop through every policy event and calculate before/after incident counts

# %%
# This will hold one row per policy event with its before/after incident counts
timing_results = []

for index, event in policy_df.iterrows():

    # Get all incident counts for this event's country
    country_rollup = incident_rollup[incident_rollup['country'] == event['country']]

    # Sum incidents in the 2 quarters before the event
    before_count = country_rollup[
        (country_rollup['year_quarter'] >= event['year_quarter'] - 2) &
        (country_rollup['year_quarter'] < event['year_quarter'])
    ]['incident_count'].sum()

    # Sum incidents in the 2 quarters after the event
    after_count = country_rollup[
        (country_rollup['year_quarter'] > event['year_quarter']) &
        (country_rollup['year_quarter'] <= event['year_quarter'] + 2)
    ]['incident_count'].sum()

    timing_results.append({
        'country': event['country'],
        'event_type': event['event_type'],
        'year_quarter': event['year_quarter'],
        'incidents_before': before_count,
        'incidents_after': after_count
    })

timing_df = pd.DataFrame(timing_results)
print(f"Total policy events analyzed: {len(timing_df)}")
timing_df.head()

# %% [markdown]
# ## Step 7: Compare the average incidents before vs. after

# %%
# Average incidents before vs. after, across all policy events
timing_df[['incidents_before', 'incidents_after']].mean().round(2)

# %%
# How many events show incidents higher before vs. higher after?
higher_before = (timing_df['incidents_before'] > timing_df['incidents_after']).sum()
higher_after = (timing_df['incidents_after'] > timing_df['incidents_before']).sum()

print(f"Events where incidents were higher BEFORE (harm led, policy followed): {higher_before} / {len(timing_df)}")
print(f"Events where incidents were higher AFTER: {higher_after} / {len(timing_df)}")

# %% [markdown]
# **Answer:** The average incident count is nearly flat — 6.12 before vs.
# 6.52 after — and slightly more events (295 out of 520) show incidents
# *higher* after the policy event than before (194). That's not the
# reactive pattern I expected going in. There's no clear evidence in this
# ±2 quarter window that policy consistently follows a spike in harm, or
# that harm reliably drops once policy responds. This sets up Question 2:
# maybe the pattern is clearer if I narrow to a specific, well-matched
# policy type and incident category instead of looking at all policy
# events and all incidents together.

# %% [markdown]
# ---
# ## Question 2: Does a specific policy event type shift related incidents?
#
# My plan: narrow to "Deepfake Law Passed" events specifically, and
# compare them only against deepfake-related incident categories (not all
# incidents). This is a more precise test than Question 1's broad
# comparison — a copyright law shouldn't be expected to affect deepfake
# incidents, so lumping all policy types and all incident types together
# in Question 1 could have hidden a real, narrower effect.

# %% [markdown]
# ## Step 8: Define which incident categories count as "deepfake-related"

# %%
# These are the incident categories most directly related to deepfakes and synthetic media
deepfake_categories = [
    'Deepfake Political Disinformation',
    'Non-Consensual Intimate Imagery',
    'Manipulated Audio Evidence',
    'Synthetic Media Electoral Interference'
]

# Filter incidents down to just these categories, then roll up by country-quarter
deepfake_incidents = incidents_df[incidents_df['incident_category'].isin(deepfake_categories)]
deepfake_rollup = deepfake_incidents.groupby(['country', 'year_quarter']).size().reset_index(name='deepfake_incident_count')

deepfake_rollup.head()

# %% [markdown]
# ## Step 9: Filter policy events to "Deepfake Law Passed" only

# %%
# Filter to just this one policy event type
deepfake_laws = policy_df[policy_df['event_type'] == 'Deepfake Law Passed']

print(f"Number of 'Deepfake Law Passed' events: {len(deepfake_laws)}")

# %% [markdown]
# ## Step 10: Calculate before/after deepfake incident counts for each law

# %%
# Same before/after logic as Question 1, but scoped to just deepfake laws and deepfake incidents
deepfake_timing_results = []

for index, event in deepfake_laws.iterrows():

    country_rollup = deepfake_rollup[deepfake_rollup['country'] == event['country']]

    before_count = country_rollup[
        (country_rollup['year_quarter'] >= event['year_quarter'] - 2) &
        (country_rollup['year_quarter'] < event['year_quarter'])
    ]['deepfake_incident_count'].sum()

    after_count = country_rollup[
        (country_rollup['year_quarter'] > event['year_quarter']) &
        (country_rollup['year_quarter'] <= event['year_quarter'] + 2)
    ]['deepfake_incident_count'].sum()

    deepfake_timing_results.append({
        'country': event['country'],
        'year_quarter': event['year_quarter'],
        'deepfake_incidents_before': before_count,
        'deepfake_incidents_after': after_count
    })

deepfake_timing_df = pd.DataFrame(deepfake_timing_results)
deepfake_timing_df

# %% [markdown]
# ## Step 11: Compare the average deepfake incidents before vs. after

# %%
# Average deepfake incidents before vs. after
deepfake_timing_df[['deepfake_incidents_before', 'deepfake_incidents_after']].mean().round(2)

# %%
# How many countries saw a decrease after the law passed?
decreased = (deepfake_timing_df['deepfake_incidents_after'] < deepfake_timing_df['deepfake_incidents_before']).sum()
print(f"Countries where deepfake incidents decreased after the law: {decreased} / {len(deepfake_timing_df)}")

# %% [markdown]
# **Answer:** Even narrowed to this specific match, the pattern holds:
# deepfake incidents are nearly flat before vs. after a deepfake law passes
# (1.0 before vs. 1.3 after, on average), and only 9 out of 30 countries
# (30%) show a decrease. Narrowing the comparison didn't reveal a hidden
# effect — if anything, it confirms Question 1's finding: in this dataset,
# passing a deepfake law isn't associated with a measurable short-term drop
# in deepfake-related incidents. That's a real, if unglamorous, finding
# worth stating plainly rather than reaching for a more flattering
# interpretation.

# %% [markdown]
# ---
# ## Question 3: Do model releases predict a rise in related incidents?
#
# My plan: shift focus from policy to model releases. Count how many
# image-generation, video-generation, and multimodal models are released
# each quarter globally, and compare that against deepfake-related
# incidents globally in the *following* quarter. Unlike Questions 1 and 2,
# this isn't country-specific — a model release isn't tied to one country
# the way a law is, so this is a global time-series comparison instead.

# %% [markdown]
# ## Step 12: Add a year_quarter column to model_releases

# %%
# Convert the release_date column into a proper date type
model_releases_df['release_date'] = pd.to_datetime(model_releases_df['release_date'])

# Pull out the year and quarter from the release date
model_releases_df['release_year'] = model_releases_df['release_date'].dt.year
model_releases_df['release_quarter_num'] = model_releases_df['release_date'].dt.quarter
model_releases_df['year_quarter'] = model_releases_df['release_year'] * 4 + model_releases_df['release_quarter_num']

model_releases_df[['model_name', 'release_date', 'year_quarter']].head()

# %% [markdown]
# ## Step 13: Count visual-generation model releases per quarter
#
# I'm using Image Gen, Video Gen, and Multimodal as the categories most
# likely to be connected to deepfake-style risk.

# %%
# Filter to the relevant model categories
visual_gen_models = model_releases_df[model_releases_df['category'].isin(['Image Gen', 'Video Gen', 'Multimodal'])]

# Count releases per quarter
release_counts = visual_gen_models.groupby('year_quarter').size().reset_index(name='release_count')
release_counts

# %% [markdown]
# ## Step 14: Count deepfake-related incidents per quarter, globally

# %%
# Using the same deepfake_categories list from Question 2, but not filtering by country this time
deepfake_global_counts = incidents_df[
    incidents_df['incident_category'].isin(deepfake_categories)
].groupby('year_quarter').size().reset_index(name='deepfake_incident_count')

deepfake_global_counts

# %% [markdown]
# ## Step 15: Merge the two quarterly series and shift incidents back one quarter
#
# I want to compare releases in one quarter against incidents in the
# *following* quarter, so I create a "next_quarter_incidents" column using
# `.shift(-1)`, which pulls each row's value up from the row below it.

# %%
# Merge the two series on year_quarter
release_vs_incidents = release_counts.merge(deepfake_global_counts, on='year_quarter', how='outer')
release_vs_incidents = release_vs_incidents.fillna(0).sort_values('year_quarter')

# Shift deepfake_incident_count back one row so each quarter shows the FOLLOWING quarter's incidents
release_vs_incidents['next_quarter_incidents'] = release_vs_incidents['deepfake_incident_count'].shift(-1)

release_vs_incidents

# %% [markdown]
# ## Step 16: Check the correlation between releases and next-quarter incidents

# %%
# Correlation between this quarter's releases and next quarter's incidents
release_vs_incidents[['release_count', 'next_quarter_incidents']].corr()

# %% [markdown]
# **Answer:** There's a moderate positive correlation (r = 0.44) between
# visual-generation model releases in a quarter and deepfake-related
# incidents in the following quarter. That's consistent with the idea that
# a wave of new image/video-generation tools precedes a rise in related
# harm. I want to be honest about the limits here, though: this is only
# about 12-16 quarters of data, which is a small sample for a correlation
# like this — it's suggestive, not proof, and I wouldn't want to overstate
# it as a confirmed causal relationship without a lot more data or a more
# rigorous design.

# %% [markdown]
# ---
# ## Question 4: Which countries show "regulation theater"?
#
# My plan: for each country, calculate (1) how much policy activity it has
# had, and (2) whether public trust improved over the period. Countries
# with a lot of policy activity but no improvement in trust are the
# "regulation theater" candidates — visible activity without a visible
# result.

# %% [markdown]
# ## Step 17: Count total incidents per country

# %%
# Total incidents per country across the whole dataset
total_incidents_by_country = incidents_df.groupby('country').size().reset_index(name='total_incidents')
total_incidents_by_country.head()

# %% [markdown]
# ## Step 18: Summarize policy activity per country

# %%
# Count total policy events and average policy impact score per country
policy_summary = policy_df.groupby('country').agg(
    total_policy_events=('event_type', 'count'),
    avg_policy_impact_score=('policy_impact_score', 'mean')
).reset_index()

policy_summary.head()

# %% [markdown]
# ## Step 19: Calculate trust change per country
#
# I'm comparing each country's `public_trust_score` in its earliest
# available year against its latest available year.

# %%
# Sort by year, then take the first and last row per country
trust_first_year = country_year_df.sort_values('year').groupby('country').first()[['public_trust_score']]
trust_first_year = trust_first_year.rename(columns={'public_trust_score': 'trust_first_year'})

trust_last_year = country_year_df.sort_values('year').groupby('country').last()[['public_trust_score']]
trust_last_year = trust_last_year.rename(columns={'public_trust_score': 'trust_last_year'})

# Combine the two and calculate the change
trust_change = trust_first_year.join(trust_last_year)
trust_change['trust_change'] = trust_change['trust_last_year'] - trust_change['trust_first_year']
trust_change = trust_change.reset_index()

trust_change.head()

# %% [markdown]
# ## Step 20: Combine everything into one summary table

# %%
# Merge policy activity, total incidents, and trust change into one table per country
country_summary = policy_summary.merge(total_incidents_by_country, on='country', how='left')
country_summary = country_summary.merge(trust_change, on='country', how='left')
country_summary['total_incidents'] = country_summary['total_incidents'].fillna(0)

print(f"Countries with at least one policy event: {len(country_summary)}")
country_summary.head(10)

# %% [markdown]
# **Note:** only countries that had at least one policy event show up in
# this summary — countries with zero policy events aren't part of this
# comparison, since there's no policy activity to evaluate for them.

# %% [markdown]
# ## Step 21: Split countries into high/low policy activity and improved/not-improved trust

# %%
# Use the median as the cutoff for "high" vs. "low" policy activity
policy_event_median = country_summary['total_policy_events'].median()
trust_change_median = country_summary['trust_change'].median()

print(f"Median policy events: {policy_event_median}")
print(f"Median trust change: {trust_change_median}")

# %%
# Flag each country as high/low policy activity and improved/not-improved trust
country_summary['high_policy_activity'] = country_summary['total_policy_events'] > policy_event_median
country_summary['trust_improved'] = country_summary['trust_change'] > trust_change_median

country_summary[['country', 'total_policy_events', 'high_policy_activity', 'trust_change', 'trust_improved']].head(10)

# %% [markdown]
# ## Step 22: Identify the "regulation theater" countries
#
# These are countries with high policy activity where trust did NOT
# improve — lots of visible action, no visible result.

# %%
# Filter to the regulation theater quadrant
regulation_theater = country_summary[
    (country_summary['high_policy_activity']) & (~country_summary['trust_improved'])
]

regulation_theater_sorted = regulation_theater.sort_values('total_policy_events', ascending=False)
regulation_theater_sorted[['country', 'total_policy_events', 'avg_policy_impact_score', 'trust_change']]

# %% [markdown]
# **Answer:** Seven countries land in the "regulation theater" quadrant —
# high policy activity with no improvement in public trust: India, UAE,
# Turkey, Italy, France, Poland, and China. India and UAE stand out with
# the most policy events (27 each), and UAE shows the steepest trust
# decline (-6.81 points) despite that activity. When I checked region and
# income group, four of the seven (France, Turkey, Poland, Italy) are in
# Europe, but the group spans a mix of income levels (from lower-middle to
# high income), so this doesn't look like a story that's purely about
# region or wealth — high policy activity without a trust payoff shows up
# across genuinely different kinds of countries. Combined with Questions 1
# and 2, this builds a consistent picture: policy activity in this dataset
# doesn't reliably translate into either less harm or more public trust,
# at least not on the timelines and measures used here.
