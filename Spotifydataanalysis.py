import seaborn as sw
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
tr_ds=pd.read_csv('data/tracks.csv')
print(tr_ds.head())
#check null values
a=pd.isnull(tr_ds).sum()
print(a)
print(tr_ds.info())
#sort according to song popularity
sort=tr_ds.sort_values('popularity')
top_10=tr_ds['popularity'].max()
print(top_10)
a=sort.head(1)
print(a)
#describe the dataset
des=tr_ds.describe().transpose()
print(des)
popularity_songs=tr_ds.query('popularity>90',inplace=False).sort_values('popularity',ascending=False)
#top 5 popular songs
#popularity_songs[:10].to_csv('Popularity_songs.csv')
tr_ds.set_index('release_date',inplace=True)
tr_index=pd.to_datetime(tr_ds.index)
tops=tr_ds.head(5)
print(tops)
print(tr_ds[['artists','energy']].iloc[18])
#convert the millisecond time duration to sec time duration
tr_ds['duration']=tr_ds['duration_ms'].apply(lambda x:round(x/1000))
tr_ds.drop('duration_ms',inplace=True,axis=1)
heads=tr_ds.head(5)
print(heads)
corr=tr_ds.drop(['explicit','mode','key'],axis=1).corr(method='pearson')
plt.figure(figsize=(14,16))
heatmap=sw.heatmap(corr,vmin=-1,vmax=1,center=0,cmap='inferno',annot=True,fmt='.1g',linewidths=1,linecolor='red')
heatmap.set_title('Correlation Heatmap between variables')
heatmap.set_xticklabels(heatmap.get_xticklabels(),rotation=270)
plt.show()
#for regression model
sample_df=tr_ds.sample(int(0.04*len(tr_ds)))
sample_lenght=len(sample_df)
print(sample_lenght)
#regression plot between loudness and energy so higly positive correlation
plt.figure(figsize=(14,16))
sw.regplot(data=sample_df,y='loudness',x='energy',color='y').set_title('loudness Vs energy')
#regerssion plot between austotic and popularity(highly negative correlation)
plt.figure(figsize=(14,28))
sw.regplot(data=sample_df,y='popularity',x='acousticness',color='b').set(title='Ausotic Vs Popularity')
plt.show()
tr_ds['dates']=tr_ds.index.get_level_values('release_date')
print(tr_ds['dates'])
tr_ds['dates']=pd.to_datetime(tr_ds['dates'])
tr_ds['Years']=tr_ds['dates'].dt.year
years=tr_ds['dates'].dt.year
print(years)
sw.displot(data=tr_ds['Years'],discrete=True,aspect=2,height=7,kind='hist')
plt.title('Number of songs per Year')
plt.show()
#total_duration of the song
total_duration=tr_ds['duration']
print(total_duration)
sw.set_style('whitegrid')
figs=(14,16)
plt.figure(figsize=figs)
sw.lineplot(x=years,y=total_duration).set_title('Duration Vs Years')
plt.xticks(rotation=60)
plt.xlabel('Years')
plt.ylabel('Durations')
plt.show()
sw.barplot(x=years,y=total_duration,errwidth=False).set_title('Duration Vs Years')
plt.xticks(rotation=90)
plt.xlabel('Years')
plt.ylabel('Duration')
plt.show()
sp_features=pd.read_csv('data/SpotifyFeatures.csv')
plt.title('Genre Vs Duration of Songs')
sp_features['duration']=sp_features['duration_ms'].apply(lambda x:round(x/60000))
sp_features.drop('duration_ms',inplace=True,axis=1)
sw.color_palette('bright',as_cmap=True)
sw.barplot(x='duration',y='genre',data=sp_features)
plt.xlabel('Duration in min')
plt.ylabel('Genre')
plt.show()
# songs according to popularity
# top 10 popular songs
top_10_pop=sp_features.sort_values('popularity',ascending=False).head(10)
print(top_10_pop)
sw.barplot(x='duration',y='genre',data=top_10_pop,palette='bright').set(title='Top 10 popular songs by genre and duration')
plt.xlabel('Duration Of Song')
plt.ylabel('Genre of Song')
plt.show()
# least 20 popular songs
# plot b/w genre and duration
least_10_pop=sp_features.sort_values('popularity',ascending=True).head(20)
sw.set_style('darkgrid')
sw.barplot(x='duration',y='genre',data=least_10_pop,palette='pastel').set(title='least popular songs according to genre')
plt.xlabel('Duration')
plt.ylabel('Genre')
plt.show()