# %% [markdown]
# # AI Regulation Timing & Trust Analysis
#
# This project uses five linked datasets tracking global GenAI adoption,
# harm incidents, policy activity, model releases, and public sentiment
# from 2022-2025 to ask a question that matters a lot in AI governance
# circles but doesn't get asked directly very often: **does regulation
# actually change anything, or is a lot of it just activity without
# results?** It's a timing and correlation analysis, not a causal one —
# it can show whether policy activity and outcomes move together, not
# prove that one causes the other.
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
# 5. Does the same before/after pattern hold for a completely different
#    harm category — algorithmic bias — using the closest available
#    policy match in this dataset?
#
# For every before/after comparison, I'm also running a paired
# significance test (not just eyeballing two averages), checking whether
# Question 1's finding holds up across different window widths (a
# sensitivity check), and adding a p-value to Question 3's correlation.
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
#   directly in these questions, but available for follow-up work)

# %% [markdown]
# ---
# ## Step 1: Import the modules

# %%
# Import the modules
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import geopandas as gpd
import country_converter as coco
from scipy.stats import ttest_rel, pearsonr

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
# ## Step 7b: Check whether the before/after difference is statistically significant
#
# Averages of 6.12 vs 6.52 look close, but "close" isn't the same as "not
# significantly different." I'm running a paired t-test — paired because
# each policy event contributes one before-count and one after-count, and
# I want to know if the *within-event* before/after difference is
# consistently non-zero, not just compare two unrelated groups.

# %%
# Paired t-test: is the before/after difference significantly different from zero?
t_stat, p_value = ttest_rel(timing_df['incidents_before'], timing_df['incidents_after'])
print(f"Paired t-test: t = {round(t_stat, 3)}, p = {round(p_value, 4)}")

# %% [markdown]
# **Answer:** The difference *is* statistically significant (t = -1.99,
# p = 0.047) — but not in the direction I was hoping to rule out. The
# negative t-statistic means incidents are significantly *higher* after a
# policy event than before, not lower. That's a more pointed finding than
# "no clear pattern": in this dataset, the ±2 quarter window around a
# policy event is associated with a small but real *increase* in
# incidents, not the decrease you'd want to see if policy were working.
# I want to be careful about what this can and can't support — this
# doesn't mean policy *causes* more harm (the more likely story is
# reverse timing: a visible incident spike is often what prompts a policy
# response in the first place, so some of this "after" bump may really be
# the tail end of whatever triggered the policy). But it does mean the
# original hope — "surely harm drops once policy responds" — isn't
# supported here, and the more honest data pattern is close to the
# opposite.

# %% [markdown]
# ## Step 7c: Sensitivity check — does the finding hold at other window widths?
#
# I chose ±2 quarters somewhat arbitrarily. If I'd picked ±1 or ±4 quarters
# instead, would the "no clear pattern" finding still hold, or is it an
# artifact of this specific window choice? I'm rerunning the same
# before/after comparison at three window widths to check.

# %%
# Rerun the before/after comparison at multiple window widths
window_results = []

for window in [1, 2, 3, 4]:

    widths_before = []
    widths_after = []

    for index, event in policy_df.iterrows():
        country_rollup = incident_rollup[incident_rollup['country'] == event['country']]

        before = country_rollup[
            (country_rollup['year_quarter'] >= event['year_quarter'] - window) &
            (country_rollup['year_quarter'] < event['year_quarter'])
        ]['incident_count'].sum()

        after = country_rollup[
            (country_rollup['year_quarter'] > event['year_quarter']) &
            (country_rollup['year_quarter'] <= event['year_quarter'] + window)
        ]['incident_count'].sum()

        widths_before.append(before)
        widths_after.append(after)

    t_stat_w, p_value_w = ttest_rel(widths_before, widths_after)

    window_results.append({
        'window_quarters': window,
        'avg_before': round(sum(widths_before) / len(widths_before), 2),
        'avg_after': round(sum(widths_after) / len(widths_after), 2),
        'p_value': round(p_value_w, 4)
    })

window_sensitivity_table = pd.DataFrame(window_results)
window_sensitivity_table

# %% [markdown]
# **Answer:** The finding is window-dependent, not uniformly robust — and
# that dependency itself is informative. At the narrowest window (±1
# quarter), the before/after difference isn't significant (p = 0.44). But
# it becomes significant at ±2 quarters (p = 0.047) and grows *more*
# significant at ±3 and ±4 quarters (p = 0.014, then p = 0.001), with the
# after-average pulling further ahead of the before-average at each wider
# window. That pattern — a gap that sharpens the longer you look past a
# policy event — is more consistent with a slow-building trend than a
# short-term shock right at the policy moment. It also means my original
# ±2 quarter choice wasn't cherry-picked to manufacture significance; if
# anything, a wider window would have shown an even stronger version of
# this same pattern.

# %% [markdown]
# ## Step 7d: Visualize the before/after comparison, normalized per quarter
#
# The line chart version of this had a real problem: it plotted raw sums
# across windows of different widths, so of course the ±4 quarter numbers
# looked bigger than the ±1 quarter numbers. That's mechanical (you're
# literally summing more quarters), not evidence the effect itself is
# growing. Dividing each total by its window width gives a fair,
# per-quarter comparison across widths, and a dumbbell chart (one dot for
# before, one for after, connected by a line) shows the actual gap at
# each width more directly than two overlapping line series did.

# %%
# Normalize by window width so the comparison across widths is fair
window_sensitivity_table['before_per_quarter'] = (
    window_sensitivity_table['avg_before'] / window_sensitivity_table['window_quarters']
)
window_sensitivity_table['after_per_quarter'] = (
    window_sensitivity_table['avg_after'] / window_sensitivity_table['window_quarters']
)
window_sensitivity_table

# %%
# Dumbbell chart: before vs after, per quarter, at each window width
fig, ax = plt.subplots(figsize=(8, 5))

y_pos = range(len(window_sensitivity_table))
labels = [f"±{w}" for w in window_sensitivity_table['window_quarters']]

for i, row in window_sensitivity_table.iterrows():
    ax.plot([row['before_per_quarter'], row['after_per_quarter']], [i, i],
            color='#999999', linewidth=1.5, zorder=1)

ax.scatter(window_sensitivity_table['before_per_quarter'], y_pos,
           color='#5B7B96', s=120, zorder=2, label='Before (per quarter)')
ax.scatter(window_sensitivity_table['after_per_quarter'], y_pos,
           color='#B85C38', s=120, zorder=2, label='After (per quarter)')

for i, row in window_sensitivity_table.iterrows():
    if row['p_value'] < 0.05:
        x_pos = max(row['before_per_quarter'], row['after_per_quarter']) + 0.08
        ax.annotate('*', (x_pos, i), fontsize=16, va='center')

ax.set_yticks(list(y_pos))
ax.set_yticklabels(labels)
ax.set_ylim(-0.7, len(window_sensitivity_table) - 0.3)
ax.set_ylabel('Window width (± quarters)')
ax.set_xlabel('Average incidents per quarter (normalized)')
ax.set_title('Incidents Before and After, Normalized Per Quarter')
ax.legend(loc='upper center', ncol=2, bbox_to_anchor=(0.5, -0.18), frameon=False)
ax.text(0.98, 1.02, '* p < 0.05', transform=ax.transAxes, ha='right', fontsize=9, style='italic')

plt.tight_layout()
plt.savefig('incidents_before_after_all.png', dpi=150, bbox_inches='tight')
plt.show()

# %% [markdown]
# **Answer, continued:** Normalizing changes the picture in an important
# way. The "before" rate actually *drops* as the window widens (3.26 per
# quarter at ±1, down to 2.80 at ±4), while "after" stays roughly flat
# (3.36 down to only 3.14). The gap between them still grows, from nearly
# nothing at ±1 to a full 0.34 incidents per quarter at ±4, but it's a
# smaller, more precise effect than the raw-sum version implied, and now
# it's clear the growth is a genuine widening rather than an artifact of
# summing more quarters into a bigger number.

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
# ## Step 11b: Check significance and visualize
#
# Same paired t-test as Question 1, scoped to just these 30 deepfake law
# events. With a smaller sample here, I want to be upfront that this test
# has less power to detect a real effect even if one existed.

# %%
# Paired t-test on the deepfake-specific before/after comparison
t_stat_df, p_value_df = ttest_rel(
    deepfake_timing_df['deepfake_incidents_before'],
    deepfake_timing_df['deepfake_incidents_after']
)
print(f"Paired t-test: t = {round(t_stat_df, 3)}, p = {round(p_value_df, 4)}")

# %%
# Event-study data: instead of collapsing to two aggregated numbers, break
# incidents out by their actual quarter position relative to the law
# (-2, -1, 0, +1, +2), so the shape of the pattern over real time is
# visible instead of hidden inside two averages
offsets = range(-2, 3)
event_study_results = []

for index, event in deepfake_laws.iterrows():
    country_rollup = deepfake_rollup[deepfake_rollup['country'] == event['country']]
    for offset in offsets:
        target_quarter = event['year_quarter'] + offset
        count = country_rollup[country_rollup['year_quarter'] == target_quarter]['deepfake_incident_count'].sum()
        event_study_results.append({'country': event['country'], 'offset': offset, 'count': count})

event_study_df = pd.DataFrame(event_study_results)
avg_by_offset = event_study_df.groupby('offset')['count'].mean().reset_index()
avg_by_offset

# %% [markdown]
# ## Step 11c: Check whether the +1 spike is broad or driven by outliers
#
# An average by itself can't tell me whether every event nudged up a
# little, or whether one or two extreme events dragged the whole average
# up while most events didn't move at all. I want to see the actual
# spread before trusting the spike.

# %%
# How many of the 30 events had at least one incident at each offset?
for offset in offsets:
    subset = event_study_df[event_study_df['offset'] == offset]
    nonzero = (subset['count'] > 0).sum()
    print(f"Offset {offset:+d}: {nonzero}/30 events had at least one incident, max single event = {subset['count'].max()}")

# %% [markdown]
# **Answer:** The +1 spike is broad-based, not a couple of outliers doing
# all the work. 17 of 30 events show at least one incident in the quarter
# right after the law, versus only 10 of 30 at -1 (one quarter before) —
# more countries are experiencing *some* deepfake activity right after
# their law passes, not just a handful having a bad quarter. Two events
# do reach the high end (4 incidents each) and add extra lift to the
# average, but even setting those two aside, the remaining events at +1
# still average higher than the events at -1. Real pattern, modestly
# amplified by a couple of higher-count events, not a single outlier
# story.

# %%
# Strip plot: every individual event's incident count at each offset,
# jittered so overlapping values are visible, with the average marked.
# This is a genuinely different chart type from the bar/line charts used
# elsewhere, and it's the right one here specifically because the
# question ("is this broad or outlier-driven?") is about the *spread*
# behind the average, which a bar chart hides by design.
np.random.seed(1)
fig, ax = plt.subplots(figsize=(8, 5.5))

strip_colors = {-2: '#5B7B96', -1: '#5B7B96', 0: '#999999', 1: '#B85C38', 2: '#B85C38'}

for offset in offsets:
    subset = event_study_df[event_study_df['offset'] == offset]
    jitter = np.random.uniform(-0.15, 0.15, size=len(subset))
    ax.scatter(offset + jitter, subset['count'], color=strip_colors[offset],
               alpha=0.6, s=70, edgecolor='white', linewidth=0.5)
    ax.scatter([offset], [subset['count'].mean()], color='black', marker='_', s=400, linewidth=2.5, zorder=3)

ax.axvline(0, color='black', linewidth=0.8, linestyle='--')
ax.set_xticks(list(offsets))
ax.set_xlabel('Quarters relative to deepfake law passage (0 = quarter law passed)')
ax.set_ylabel('Deepfake incidents (one dot = one law event)')
ax.set_title('Deepfake Incidents by Quarter, Relative to Law Passage')
ax.text(0.02, 0.97, 'Black dash = average at that quarter', transform=ax.transAxes,
        fontsize=8.5, style='italic', va='top')

plt.tight_layout()
plt.savefig('deepfake_before_after.png', dpi=150, bbox_inches='tight')
plt.show()

# %% [markdown]
# **Answer, continued:** The difference is not statistically significant
# (t = -1.03, p = 0.313). Unlike Question 1's broad comparison, this
# narrower, well-matched test doesn't find a significant shift in either
# direction — which is a real difference from Question 1's finding, worth
# sitting with rather than smoothing over. One likely reason: only 30
# deepfake law events, versus 520 policy events overall, means this test
# has much less statistical power to detect an effect even if a smaller
# one were present. "Not significant" here is honestly closer to
# "inconclusive with this sample size" than "confirmed no effect."
#
# The quarter-by-quarter chart adds something the two collapsed averages
# hid: incidents don't move flat and steady, they actually *spike* in the
# quarter immediately after the law passes (offset +1, the highest bar in
# the chart) before falling back down by offset +2. Averaging +1 and +2
# together into a single "after" number, like the before/after table
# does, buries that spike inside a more modest-looking average. Whether
# that immediate-aftermath spike is a real short-term effect (e.g.
# increased reporting/enforcement attention right after a law passes) or
# just noise from a small sample is something this dataset alone can't
# settle — but it's a more specific, more honest pattern than "flat" to
# report.

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
# ## Step 16b: Get the p-value for this correlation, and visualize it
#
# The correlation coefficient alone doesn't say whether r = 0.44 is likely
# to be real given how few data points I have. `pearsonr` gives me both the
# correlation and its p-value in one step.

# %%
# Drop the last row (its next_quarter_incidents is NaN since there's no
# quarter after the final one to shift in), then get correlation + p-value
corr_data = release_vs_incidents.dropna(subset=['next_quarter_incidents'])
corr_value, corr_p_value = pearsonr(corr_data['release_count'], corr_data['next_quarter_incidents'])
print(f"Correlation: r = {round(corr_value, 3)}, p = {round(corr_p_value, 4)}, n = {len(corr_data)}")

# %% [markdown]
# ## Step 16c: Check whether a broader trend explains the co-movement
#
# Before trusting r = 0.44 as evidence of a specific releases-to-incidents
# link, I want to check a simpler explanation: maybe releases and
# incidents are just both swept up in the same overall AI adoption growth
# over 2022-2025, not specifically tied to each other one quarter apart.
# If total incidents (every category, not just deepfakes) and total model
# releases (every category, not just visual-gen) show the same rise-and
# ease shape, that's a sign of a shared macro trend rather than a direct
# relationship.

# %%
# Total releases and total incidents by year (not filtered to visual-gen
# or deepfake specifically, since this is checking the broader trend)
total_releases_by_year = model_releases_df.groupby('release_year').size()
total_incidents_by_year = incidents_df.groupby('year').size()
print("Total model releases by year:")
print(total_releases_by_year)
print("\nTotal incidents by year:")
print(total_incidents_by_year)

# %%
# Index both to their 2022 value = 100, so their SHAPES are directly
# comparable despite very different scales (tens of releases vs. hundreds
# of incidents). "Indexed" here means each series is rescaled so its own
# 2022 value becomes 100 and every later year is shown as a percentage of
# that starting point, the same technique behind things like a stock
# market index. 2022 is the fixed starting line, not a claim that
# anything "began" in 2022, it's just the earliest year in this dataset.
releases_indexed = (total_releases_by_year / total_releases_by_year.iloc[0] * 100)
incidents_indexed = (total_incidents_by_year / total_incidents_by_year.iloc[0] * 100)

fig, ax = plt.subplots(figsize=(8, 5.5))
years = total_releases_by_year.index

ax.fill_between(years, releases_indexed, alpha=0.35, color='#5B7B96', label='Model releases (indexed)')
ax.fill_between(years, incidents_indexed, alpha=0.35, color='#B85C38', label='All incidents (indexed)')
ax.plot(years, releases_indexed, color='#5B7B96', linewidth=2, marker='o')
ax.plot(years, incidents_indexed, color='#B85C38', linewidth=2, marker='o')

ax.axhline(100, color='black', linewidth=0.6, linestyle=':')
ax.set_xticks(years)
ax.set_ylabel('Indexed to 2022 = 100')
ax.set_xlabel('Year')
ax.set_title('Releases and Incidents, Indexed to Their 2022 Level')
ax.legend(loc='upper left')

# Note moved below the axes entirely, outside the shaded fill, so it stays
# legible regardless of where the shading lands
fig.text(0.5, -0.02,
          'Both series rise through 2024, then ease in 2025, tracking overall GenAI adoption.',
          ha='center', fontsize=9, style='italic')

plt.tight_layout()
plt.savefig('releases_incidents_indexed_trend.png', dpi=150, bbox_inches='tight')
plt.show()

# %% [markdown]
# **Answer:** Both series climb together through 2024 and ease off in
# 2025, and this isn't unique to releases and deepfake incidents
# specifically. Total incidents across *every* category (180 to 667) and
# total model releases across *every* category (10 to 27) show the same
# rise-then-ease shape, tracking a global GenAI adoption rate that climbed
# from 38.6% to 51.9% over the same period. That's a real alternative
# explanation for the quarter-by-quarter correlation tested next: two
# series that are each swept up in the same broader adoption wave will
# tend to move together even without one specifically triggering the
# other. This doesn't rule out a real releases-to-incidents link, but it
# means the r = 0.44 correlation below should be read as "consistent with
# a link," not "evidence a link is the best explanation" — a shared driver
# fits the data just as well.

# %%
# Build readable quarter labels (e.g. "2024-Q1") for a chronological view
def yq_to_label(yq):
    year = yq // 4
    q = yq % 4
    if q == 0:
        year -= 1
        q = 4
    return f"{int(year)}-Q{int(q)}"

release_vs_incidents['label'] = release_vs_incidents['year_quarter'].apply(yq_to_label)

# %%
# Two separate charts rather than one combined figure — each one is small
# and tight when squeezed side by side, especially once GitHub scales the
# image down to fit a README's width. The time series shows the real
# temporal shape; the scatter shows the specific relationship being
# tested. Neither alone tells the full story, but each is clearer on its
# own than both jammed into one image.

# Chart 1: chronological dual-axis time series
fig, ax1 = plt.subplots(figsize=(9, 5))

ax1.plot(release_vs_incidents['label'], release_vs_incidents['release_count'],
         marker='o', color='#5B7B96', linewidth=2, label='Visual-gen model releases')
ax1.set_ylabel('Model releases this quarter', color='#5B7B96')
ax1.tick_params(axis='y', labelcolor='#5B7B96')
plt.setp(ax1.get_xticklabels(), rotation=45, ha='right')
ax1.set_title('Model Releases and Deepfake Incidents Over Time')

ax2 = ax1.twinx()
ax2.plot(release_vs_incidents['label'], release_vs_incidents['deepfake_incident_count'],
         marker='o', color='#B85C38', linewidth=2, label='Deepfake incidents')
ax2.set_ylabel('Deepfake incidents this quarter', color='#B85C38')
ax2.tick_params(axis='y', labelcolor='#B85C38')

plt.tight_layout()
plt.savefig('releases_incidents_timeseries.png', dpi=150, bbox_inches='tight')
plt.show()

# %%
# Chart 2: scatter + trend line, the actual statistical test behind the
# time series above. Title is descriptive; the r/p/n statistics move
# below the x-axis as a note instead of doubling as the title.
fig, ax3 = plt.subplots(figsize=(7, 5.5))

ax3.scatter(corr_data['release_count'], corr_data['next_quarter_incidents'], color='#2A7F7E', s=60)
trend = np.polyfit(corr_data['release_count'], corr_data['next_quarter_incidents'], 1)
x_line = np.linspace(corr_data['release_count'].min(), corr_data['release_count'].max(), 20)
ax3.plot(x_line, trend[0] * x_line + trend[1], color='#B85C38', linestyle='--')
ax3.set_xlabel('Releases this quarter')
ax3.set_ylabel('Deepfake incidents next quarter')
ax3.set_title('Model Releases vs. Next-Quarter Deepfake Incidents')
ax3.text(0.5, -0.16, f'r = {round(corr_value, 2)}, p = {round(corr_p_value, 3)}, n = {len(corr_data)} quarters',
          transform=ax3.transAxes, ha='center', fontsize=9, style='italic')

plt.tight_layout()
plt.savefig('releases_vs_incidents_scatter.png', dpi=150, bbox_inches='tight')
plt.show()

# %% [markdown]
# **Answer:** There's a moderate positive correlation (r = 0.44) between
# visual-generation model releases in a quarter and deepfake-related
# incidents in the following quarter — consistent with the idea that a
# wave of new image/video-generation tools precedes a rise in related
# harm. With only 15 usable quarters of paired data, the p-value (0.0995)
# just misses the conventional 0.05 threshold. I don't want to round that
# down to "not significant" and treat the finding as nothing, and I don't
# want to round it up to "significant" either — the honest read is that
# this sample is too small to say confidently either way, and a
# moderate correlation showing up at all with just 15 data points is
# suggestive enough to be worth someone re-testing with more quarters of
# data, not proof of a real relationship on its own.
#
# The time series panel adds a caveat the correlation coefficient alone
# doesn't show: the two series broadly rise together through 2024, but
# they diverge at the edges — 2025-Q1 shows a spike in releases (4, the
# highest in the dataset) without a matching spike in incidents, and the
# last two quarters show releases dropping to zero while incidents stay
# elevated. A single correlation number averages over that kind of
# breakdown; seeing it on the time series is a useful check against
# reading the r = 0.44 as a relationship that holds steady throughout.

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
country_summary['regulation_theater'] = country_summary['high_policy_activity'] & ~country_summary['trust_improved']

regulation_theater = country_summary[country_summary['regulation_theater']]

regulation_theater_sorted = regulation_theater.sort_values('total_policy_events', ascending=False)
regulation_theater_sorted[['country', 'total_policy_events', 'avg_policy_impact_score', 'trust_change']]

# %% [markdown]
# **Answer:** Seven countries land in the "regulation theater" quadrant —
# high policy activity with no improvement in public trust: India, UAE,
# Turkey, Italy, France, Poland, and China. India and UAE stand out with
# the most policy events (27 each), and UAE shows the steepest trust
# decline (-6.81 points) despite that activity. Combined with Questions 1
# and 2, this builds a consistent picture: policy activity in this dataset
# doesn't reliably translate into either less harm or more public trust,
# at least not on the timelines and measures used here. *(Whether this
# concentrates in particular parts of the world is checked properly, with
# actual code, in Step 22c below — the geographic pattern turns out to be
# more specific than "spans multiple income levels.")*

# %% [markdown]
# ## Step 22b: Visualize policy activity and trust change as a butterfly chart
#
# A scatter plot with median lines turned out to be hard to read at a
# glance — 30 overlapping dots don't make the pattern jump out. A
# butterfly chart (two horizontal bar panels sharing one country axis)
# pairs the same two numbers per country side by side instead.
#
# One design choice worth being deliberate about: color should mean
# exactly one thing per panel. The trust-change panel's color is the sign
# of the bar itself (declined vs. improved) — nothing else — so the color
# a country shows always matches what its own bar is doing. "Regulation
# theater" status is a *joint* condition (high activity AND below-median
# trust change, not simply negative trust change), so it gets its own
# separate marker (a bold label with a ★) rather than hijacking the color
# channel — a country can have declining trust without being "regulation
# theater" if its policy activity isn't also above median, and coloring
# both by the same joint category made that look like a contradiction
# instead of the real, more precise rule it actually is.

# %%
# Sort countries by policy activity so the most active countries anchor the chart
plot_df = country_summary.sort_values('total_policy_events', ascending=True).reset_index(drop=True)

# Color the trust panel by the sign of trust_change itself — this is the
# only color story on that panel, so it never conflicts with the bar
trust_colors = plot_df['trust_change'].apply(lambda x: '#B85C38' if x < 0 else '#2A7F7E')
activity_color = '#5B7B96'

fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(11.5, 9), sharey=True)

# Left panel: policy activity — neutral color, this panel is about magnitude only
ax_left.barh(plot_df['country'], plot_df['total_policy_events'], color=activity_color)
ax_left.invert_xaxis()
ax_left.set_xlabel('Total policy events')
ax_left.set_title('Policy activity', fontsize=11)

# Right panel: trust change — color always matches the bar's own sign
ax_right.barh(plot_df['country'], plot_df['trust_change'], color=trust_colors)
ax_right.axvline(0, color='black', linewidth=0.8)
ax_right.set_xlabel('Change in public trust score')
ax_right.set_title('Trust change', fontsize=11)

# Bold + star the regulation theater countries' labels specifically —
# this is a separate visual channel from the bar colors above
for label in ax_left.get_yticklabels():
    country_name = label.get_text()
    is_theater = plot_df.loc[plot_df['country'] == country_name, 'regulation_theater'].values[0]
    if is_theater:
        label.set_fontweight('bold')

fig.suptitle('Policy activity vs. trust change, by country', fontsize=13)

legend_elements = [
    mpatches.Patch(color='#B85C38', label='Trust declined'),
    mpatches.Patch(color='#2A7F7E', label='Trust improved'),
    mpatches.Patch(color='#5B7B96', label='Policy activity (magnitude only)'),
]
fig.legend(handles=legend_elements, loc='lower center', ncol=3, fontsize=8.5, bbox_to_anchor=(0.5, -0.03))
fig.text(0.5, -0.06, '★ Bold country name = regulation theater (high activity + below-median trust change)',
          ha='center', fontsize=8.5, style='italic')

plt.tight_layout()
plt.savefig('regulation_theater_butterfly.png', dpi=150, bbox_inches='tight')
plt.show()

# %% [markdown]
# **Answer, continued:** With color now tied only to sign, the chart tells
# a more precise story than before. Every regulation theater country
# (bolded) does show a declining trust bar — but so do a few non-theater
# countries (Kenya, the UK, Singapore), because their policy activity
# doesn't clear the median threshold that's the *other* half of the
# regulation theater definition. That's not a contradiction, it's the
# actual rule made visible: "regulation theater" specifically means high
# activity paired with a lagging trust outcome, not just any trust
# decline on its own. Reading down from the top (highest policy activity),
# the bolded countries mostly cluster with negative trust bars, while
# Canada, Brazil, and Germany show comparably high activity paired with a
# clearly positive trust bar instead — that contrast is the real finding.

# %% [markdown]
# ## Step 22c: Check whether regulation theater clusters by continent
#
# The original version of this analysis had an unverified claim in prose
# about region — I'm replacing that with an actual check, and using
# continent rather than the World Bank's region groupings specifically,
# since "region" categories like "Europe & Central Asia" or "East Asia &
# Pacific" reflect a World Bank-specific, somewhat Western-institutional
# way of grouping the world. Continent is a simpler, harder-to-dispute
# grouping for this kind of check.

# %%
# Convert each country name to its continent using country_converter
cc = coco.CountryConverter()
country_summary['continent'] = cc.convert(names=country_summary['country'].tolist(), to='continent')

# country_converter doesn't recognize "UAE" as written in this dataset —
# fixing that one manually rather than leaving it unclassified
country_summary['continent'] = country_summary.apply(
    lambda row: 'Asia' if row['country'] == 'UAE' else row['continent'], axis=1
)

# How many regulation theater countries come from each continent?
theater_by_continent = country_summary[country_summary['regulation_theater']]['continent'].value_counts()
print("Regulation theater countries by continent:")
print(theater_by_continent)

# %% [markdown]
# **Answer:** The seven regulation theater countries split across two
# continents — Asia (4: India, UAE, Turkey, China) and Europe (3: Italy,
# France, Poland) — with zero from Africa, the Americas, or Oceania. That's
# a real, verified pattern rather than the "spans multiple income levels"
# impression from before: it's concentrated on two continents specifically,
# and Asia actually contributes more countries to this pattern than Europe
# does, not fewer. That's worth stating plainly rather than defaulting to
# a framing (like World Bank region groupings) that tends to center Europe
# and North America as the reference point.

# %% [markdown]
# ## Step 22d: Map the regulation theater pattern geographically
#
# A scatter plot shows *how* a country qualifies as regulation theater; a
# map shows *where*. Given this is inherently geographic, country-level
# data, a map answers "is this concentrated in one part of the world?"
# more directly than any chart type that doesn't show geography. I'm using
# a bundled world boundaries file (`world.geojson`, included in this repo)
# rather than fetching one at runtime, so this cell doesn't depend on
# internet access to reproduce.

# %%
# Match each country to its ISO3 code so it can be joined to the map boundaries
country_summary['iso3'] = cc.convert(names=country_summary['country'].tolist(), to='ISO3')
country_summary['iso3'] = country_summary.apply(
    lambda row: 'ARE' if row['country'] == 'UAE' else row['iso3'], axis=1
)

# Load the world boundaries and join in the regulation theater flag
world = gpd.read_file('world.geojson')
world_with_data = world.merge(
    country_summary[['iso3', 'country', 'regulation_theater']],
    left_on='id', right_on='iso3', how='left'
)

# %%
# Draw the map: grey for no policy data, teal for policy activity without
# regulation theater, orange for regulation theater specifically
fig, ax = plt.subplots(figsize=(13, 7))

world.plot(ax=ax, color='#e8e8e8', edgecolor='white', linewidth=0.4)
world_with_data[world_with_data['regulation_theater'] == True].plot(
    ax=ax, color='#B85C38', edgecolor='white', linewidth=0.4)
world_with_data[world_with_data['regulation_theater'] == False].plot(
    ax=ax, color='#2A7F7E', edgecolor='white', linewidth=0.4)

ax.set_axis_off()
ax.set_title('Regulation theater is concentrated in Asia and Europe, not evenly spread', fontsize=13)

legend_elements = [
    mpatches.Patch(color='#B85C38', label='Regulation theater (high policy activity, no trust gain)'),
    mpatches.Patch(color='#2A7F7E', label='Has policy activity, not regulation theater'),
    mpatches.Patch(color='#e8e8e8', label='No policy events in this dataset'),
]
ax.legend(handles=legend_elements, loc='lower left', fontsize=9, frameon=False)

plt.tight_layout()
plt.savefig('regulation_theater_map.png', dpi=150, bbox_inches='tight')
plt.show()

# %% [markdown]
# **Answer, continued:** Laid out geographically, the concentration in
# Asia and Europe is immediate rather than something you have to read off
# a value count — no regulation theater countries appear in Africa, the
# Americas, or Oceania. Combined with the continent breakdown above, this
# is a clearer and more geographically honest picture than the original
# "spans multiple income levels" line, which was true but didn't say
# anything about *where* the pattern actually concentrates.

# %% [markdown]
# ---
# ## Question 5: Does the same pattern hold for a different harm category?
#
# Questions 1-2 focused on deepfakes specifically. I want to check whether
# "policy activity doesn't move the needle" is a deepfake-specific finding
# or a broader one. There's no incident category in this dataset labeled
# specifically "hiring bias law" the way there was a clean "Deepfake Law
# Passed" event type — so rather than force a mismatched comparison, I'm
# using the closest available match: `Algorithmic Accountability Act`
# events against the two incident categories most clearly about algorithmic
# bias, `AI Bias in Hiring` and `AI Bias in Credit Scoring`.

# %% [markdown]
# ## Step 23: Roll up algorithmic bias incidents by country-quarter

# %%
# Filter to algorithmic bias incident categories, then roll up by country-quarter
bias_categories = ['AI Bias in Hiring', 'AI Bias in Credit Scoring']
bias_incidents = incidents_df[incidents_df['incident_category'].isin(bias_categories)]
bias_rollup = bias_incidents.groupby(['country', 'year_quarter']).size().reset_index(name='bias_incident_count')

bias_rollup.head()

# %% [markdown]
# ## Step 24: Filter to Algorithmic Accountability Act events

# %%
# Filter to just this policy event type
accountability_acts = policy_df[policy_df['event_type'] == 'Algorithmic Accountability Act']
print(f"Number of 'Algorithmic Accountability Act' events: {len(accountability_acts)}")

# %% [markdown]
# ## Step 25: Same before/after logic, scoped to this category match

# %%
# Same before/after pattern as Questions 1 and 2
bias_timing_results = []

for index, event in accountability_acts.iterrows():

    country_rollup = bias_rollup[bias_rollup['country'] == event['country']]

    before = country_rollup[
        (country_rollup['year_quarter'] >= event['year_quarter'] - 2) &
        (country_rollup['year_quarter'] < event['year_quarter'])
    ]['bias_incident_count'].sum()

    after = country_rollup[
        (country_rollup['year_quarter'] > event['year_quarter']) &
        (country_rollup['year_quarter'] <= event['year_quarter'] + 2)
    ]['bias_incident_count'].sum()

    bias_timing_results.append({
        'country': event['country'],
        'year_quarter': event['year_quarter'],
        'bias_incidents_before': before,
        'bias_incidents_after': after
    })

bias_timing_df = pd.DataFrame(bias_timing_results)
bias_timing_df

# %%
# Average before vs. after, plus the same paired significance test
print(bias_timing_df[['bias_incidents_before', 'bias_incidents_after']].mean().round(2))

t_stat_bias, p_value_bias = ttest_rel(
    bias_timing_df['bias_incidents_before'], bias_timing_df['bias_incidents_after']
)
print(f"\nPaired t-test: t = {round(t_stat_bias, 3)}, p = {round(p_value_bias, 4)}")

decreased_bias = (bias_timing_df['bias_incidents_after'] < bias_timing_df['bias_incidents_before']).sum()
print(f"Countries where bias incidents decreased after the act: {decreased_bias} / {len(bias_timing_df)}")

# %% [markdown]
# **Answer:** This category tells a genuinely different story than
# deepfakes, even though neither reaches statistical significance.
# Algorithmic bias incidents actually go *down* on average after an
# Algorithmic Accountability Act passes (0.75 before vs. 0.46 after), and
# 9 of 24 countries (37.5%) saw a decrease — numerically the opposite
# direction from Question 1's broad finding, and a cleaner story than
# Question 2's deepfake-specific result. But the paired t-test isn't
# significant (t = 1.27, p = 0.216), and with only 24 events, this test
# has limited power — so I can't confidently say algorithmic bias policy
# "works" while deepfake policy doesn't. What I can say is that the
# "policy activity doesn't move outcomes" finding from Questions 1-2
# doesn't automatically generalize to every harm category — this one at
# least points a different direction numerically, and would be worth
# revisiting with a larger sample of accountability-act events before
# drawing a firm conclusion either way.
