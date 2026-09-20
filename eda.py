import pandas as pd, numpy as np, json, os
import matplotlib.pyplot as plt

ROOT=os.path.dirname(__file__)
DATA=os.path.join(ROOT,'data','INvideos.csv')
OUT=os.path.join(ROOT,'outputs'); os.makedirs(OUT,exist_ok=True)

df=pd.read_csv(DATA)
original_shape=df.shape
missing=df.isna().sum()
duplicates=int(df.duplicated().sum())

# Parse dates from dataset's yy.dd.mm format
# Keep raw strings and create analysis dates.
df['trending_date_parsed']=pd.to_datetime(df['trending_date'], format='%y.%d.%m', errors='coerce')
df['publish_datetime']=pd.to_datetime(df['publish_time'], errors='coerce', utc=True)

# Load category names
with open(os.path.join(ROOT,'data','IN_category_id.json'),encoding='utf-8') as f:
    cats=json.load(f)
cat_map={int(x['id']):x['snippet']['title'] for x in cats['items']}
df['category_name']=df['category_id'].map(cat_map).fillna(df['category_id'].astype(str))

# Exact duplicate removal for the analysis copy.
clean=df.drop_duplicates().copy()
clean_shape=clean.shape

# Export key metrics
metrics={
 'original_rows': int(original_shape[0]), 'original_columns': int(original_shape[1]),
 'missing_description': int(missing['description']), 'exact_duplicate_rows': duplicates,
 'clean_rows': int(clean_shape[0]), 'clean_columns': int(clean_shape[1]),
 'unique_videos': int(clean['video_id'].nunique()), 'categories': int(clean['category_name'].nunique()),
 'trending_start': str(clean['trending_date_parsed'].min().date()),
 'trending_end': str(clean['trending_date_parsed'].max().date()),
 'mean_views': float(clean['views'].mean()), 'median_views': float(clean['views'].median()),
 'max_views': int(clean['views'].max()), 'mean_likes': float(clean['likes'].mean()),
 'median_likes': float(clean['likes'].median()), 'mean_comments': float(clean['comment_count'].mean()),
 'median_comments': float(clean['comment_count'].median()),
}
pd.Series(metrics).to_csv(os.path.join(OUT,'key_metrics.csv'),header=['value'])

# descriptive stats
clean[['views','likes','dislikes','comment_count']].describe().T.to_csv(os.path.join(OUT,'engagement_describe.csv'))

# category summary
cat_summary=(clean.groupby('category_name')
             .agg(videos=('video_id','count'),median_views=('views','median'),median_likes=('likes','median'),median_comments=('comment_count','median'))
             .sort_values('videos',ascending=False))
cat_summary.to_csv(os.path.join(OUT,'category_summary.csv'))

# channel summary
channel_summary=(clean.groupby('channel_title')
                 .agg(videos=('video_id','count'),median_views=('views','median'),total_views=('views','sum'))
                 .sort_values('videos',ascending=False))
channel_summary.to_csv(os.path.join(OUT,'channel_summary.csv'))

# correlation
clean[['views','likes','dislikes','comment_count']].corr().to_csv(os.path.join(OUT,'engagement_correlation.csv'))

# Plots
plt.rcParams.update({'figure.figsize':(10,6),'axes.titlesize':14,'axes.labelsize':11})

# 1 missing values
miss=missing[missing>0].sort_values(ascending=False)
fig,ax=plt.subplots(); miss.plot(kind='bar',ax=ax); ax.set_title('Missing Values by Column'); ax.set_ylabel('Missing rows'); ax.tick_params(axis='x',rotation=45); fig.tight_layout(); fig.savefig(os.path.join(OUT,'01_missing_values.png'),dpi=180); plt.close(fig)

# 2 engagement distributions log x
fig,ax=plt.subplots(); ax.hist(np.log10(clean['views']+1),bins=60); ax.set_title('Distribution of Views (log10 scale)'); ax.set_xlabel('log10(Views + 1)'); ax.set_ylabel('Number of rows'); fig.tight_layout(); fig.savefig(os.path.join(OUT,'02_views_distribution_log.png'),dpi=180); plt.close(fig)

# 3 boxplots log values
fig,ax=plt.subplots(); vals=[np.log10(clean[c]+1) for c in ['views','likes','dislikes','comment_count']]; ax.boxplot(vals,labels=['Views','Likes','Dislikes','Comments']); ax.set_title('Engagement Distributions (log10 scale)'); ax.set_ylabel('log10(value + 1)'); fig.tight_layout(); fig.savefig(os.path.join(OUT,'03_engagement_boxplots.png'),dpi=180); plt.close(fig)

# 4 category counts top 12
fig,ax=plt.subplots(); s=cat_summary.head(12).sort_values('videos'); s['videos'].plot(kind='barh',ax=ax); ax.set_title('Top Categories by Number of Records'); ax.set_xlabel('Number of records'); fig.tight_layout(); fig.savefig(os.path.join(OUT,'04_category_counts.png'),dpi=180); plt.close(fig)

# 5 median views by category
fig,ax=plt.subplots(); s=cat_summary.sort_values('median_views').tail(12); s['median_views'].plot(kind='barh',ax=ax); ax.set_title('Categories with Highest Median Views'); ax.set_xlabel('Median views'); fig.tight_layout(); fig.savefig(os.path.join(OUT,'05_category_median_views.png'),dpi=180); plt.close(fig)

# 6 views likes scatter sample
sample=clean.sample(min(6000,len(clean)),random_state=42)
fig,ax=plt.subplots(); ax.scatter(sample['views'],sample['likes'],s=7,alpha=.35); ax.set_xscale('log'); ax.set_yscale('log'); ax.set_title('Views vs Likes'); ax.set_xlabel('Views (log scale)'); ax.set_ylabel('Likes (log scale)'); fig.tight_layout(); fig.savefig(os.path.join(OUT,'06_views_vs_likes.png'),dpi=180); plt.close(fig)

# 7 views comments
fig,ax=plt.subplots(); ax.scatter(sample['views'],sample['comment_count'],s=7,alpha=.35); ax.set_xscale('log'); ax.set_yscale('log'); ax.set_title('Views vs Comment Count'); ax.set_xlabel('Views (log scale)'); ax.set_ylabel('Comments (log scale)'); fig.tight_layout(); fig.savefig(os.path.join(OUT,'07_views_vs_comments.png'),dpi=180); plt.close(fig)

# 8 correlation heatmap manually (avoid seaborn dependency)
corr=clean[['views','likes','dislikes','comment_count']].corr()
fig,ax=plt.subplots(figsize=(7,6)); im=ax.imshow(corr.values); ax.set_xticks(range(4),corr.columns,rotation=45,ha='right'); ax.set_yticks(range(4),corr.index)
for i in range(4):
 for j in range(4): ax.text(j,i,f'{corr.iloc[i,j]:.2f}',ha='center',va='center')
ax.set_title('Engagement Correlation Matrix'); fig.colorbar(im,ax=ax); fig.tight_layout(); fig.savefig(os.path.join(OUT,'08_correlation_heatmap.png'),dpi=180); plt.close(fig)

# 9 trending over time
trend=clean.groupby('trending_date_parsed').agg(videos=('video_id','count'),median_views=('views','median')).sort_index()
trend.to_csv(os.path.join(OUT,'daily_trending_summary.csv'))
fig,ax=plt.subplots(); trend['videos'].plot(ax=ax); ax.set_title('Number of Trending Records Over Time'); ax.set_xlabel('Trending date'); ax.set_ylabel('Records'); fig.tight_layout(); fig.savefig(os.path.join(OUT,'09_trending_over_time.png'),dpi=180); plt.close(fig)

# 10 top channels by record count
cs=channel_summary.head(15).sort_values('videos')
fig,ax=plt.subplots(figsize=(10,7)); cs['videos'].plot(kind='barh',ax=ax); ax.set_title('Top Channels by Number of Trending Records'); ax.set_xlabel('Records'); fig.tight_layout(); fig.savefig(os.path.join(OUT,'10_top_channels.png'),dpi=180); plt.close(fig)

# 11 engagement rates (avoid division by zero)
clean['like_rate']=clean['likes']/clean['views'].replace(0,np.nan)
clean['comment_rate']=clean['comment_count']/clean['views'].replace(0,np.nan)
fig,ax=plt.subplots(); ax.scatter(sample['views'],clean.loc[sample.index,'like_rate'],s=7,alpha=.35); ax.set_xscale('log'); ax.set_yscale('log'); ax.set_title('Views vs Like Rate'); ax.set_xlabel('Views (log scale)'); ax.set_ylabel('Likes / Views (log scale)'); fig.tight_layout(); fig.savefig(os.path.join(OUT,'11_views_vs_like_rate.png'),dpi=180); plt.close(fig)

# findings text
pearson=corr.loc['views','likes']; vc=corr.loc['views','comment_count']
findings=[
 f"The raw India file contains {original_shape[0]:,} rows and {original_shape[1]} columns.",
 f"There are {duplicates:,} exact duplicate rows; the analysis copy removes these duplicates, leaving {clean_shape[0]:,} rows.",
 f"The description field has {int(missing['description']):,} missing values; the other columns have no missing values in the initial check.",
 f"Views range from {clean.views.min():,} to {clean.views.max():,}; the median is {clean.views.median():,.0f}, showing a large spread between typical and extreme observations.",
 f"The Pearson correlation between views and likes is {pearson:.3f}; correlation is descriptive and does not establish causation.",
 f"The Pearson correlation between views and comment_count is {vc:.3f}; this also represents association within this dataset.",
 f"The cleaned dataset contains {clean['category_name'].nunique()} category labels and {clean['video_id'].nunique():,} unique video IDs.",
 f"The parsed trending-date range is {clean.trending_date_parsed.min().date()} to {clean.trending_date_parsed.max().date()}.",
]
open(os.path.join(OUT,'findings.txt'),'w',encoding='utf-8').write('\n'.join('- '+x for x in findings))
print('\n'.join(findings))
