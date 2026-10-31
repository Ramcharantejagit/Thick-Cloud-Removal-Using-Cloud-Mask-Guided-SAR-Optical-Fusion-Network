import sys, csv
fps={'20190102T111147':(103.784,10.991,106.421,12.972),'20190102T111212':(103.472,12.503,106.125,14.48),'20190106T112258':(-85.132,7.691,-82.557,9.708),'20190108T120440':(88.767,23.724,91.628,25.686),'20190108T232857':(-78.805,-1.384,-76.23,0.821),'20190109T001121':(85.078,26.229,87.891,28.191),'20190109T101238':(-69.459,-5.819,-66.862,-3.79),'20190112T113154':(96.891,24.281,99.756,26.236),'20190123T095735':(-66.44,-10.67,-63.855,-8.842),'20190801T101449':(-71.22,-13.365,-68.527,-11.309)}
r=csv.reader(sys.stdin); h=next(r); print(','.join(h)+',S1'); idx={k:i for i,k in enumerate(h)}
for row in r:
    t=row[idx['SENSING_TIME']][:10]
    if not ('2018-11-01'<=t<='2019-03-31' or '2019-06-01'<=t<='2019-10-31'): continue
    if not row[idx['PRODUCT_ID']].startswith('S2'): continue
    try: n,s,w,e=[float(row[idx[k]]) for k in ('NORTH_LAT','SOUTH_LAT','WEST_LON','EAST_LON')]
    except: continue
    for k,(a,b,c,d) in fps.items():
        if w<c and e>a and s<d and n>b:
            print(','.join(row)+','+k); break
