import pandas as pd
d=pd.read_csv('s2_candidates.csv')
d['t']=pd.to_datetime(d.SENSING_TIME.str[:19]); d['s1t']=pd.to_datetime(d.S1,format='%Y%m%dT%H%M%S')
d['ddays']=(d.t-d.s1t).dt.total_seconds()/86400
d=d[d.ddays.abs()<=45].copy()
d['orbit']=d.PRODUCT_ID.str.extract(r'_(R\d{3})_')[0]
d=d.drop_duplicates('GRANULE_ID')
rows=[]
for s1,g in d.groupby('S1'):
    cl=g[(g.ddays.abs()<=6)&(g.CLOUD_COVER.between(15,75))]
    cf=g[(g.CLOUD_COVER<3)]
    for _,r in cl.iterrows():
        c=cf[(cf.MGRS_TILE==r.MGRS_TILE)&(cf.orbit==r.orbit)&(cf.TOTAL_SIZE>0.85*r.TOTAL_SIZE)&(cf.TOTAL_SIZE<1.15*r.TOTAL_SIZE)].copy()
        if len(c):
            c['gap']=(c.t-r.t).abs(); c=c.sort_values('gap').iloc[0]
            rows.append(dict(S1=s1,tile=r.MGRS_TILE,cloudy=r.BASE_URL,cloudy_cc=r.CLOUD_COVER,cloudy_dt=round(r.ddays,1),clear=c.BASE_URL,clear_cc=c.CLOUD_COVER,gap_days=c.gap.days,size=r.TOTAL_SIZE))
p=pd.DataFrame(rows); p.to_csv('s2_pairs.csv',index=False)
print(p.groupby('S1').size()); print(p.sort_values(['S1','gap_days']).groupby('S1').head(3).to_string())
