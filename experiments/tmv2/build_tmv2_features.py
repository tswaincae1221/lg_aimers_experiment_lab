from __future__ import annotations
import gc, math, json, time, os
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(os.environ.get('TMV2_ROOT','/mnt/data'))
TRACKMAN=Path(os.environ.get('TMV2_TRACKMAN', ROOT/'tmv2_source/trackman_history.csv'))
TRAIN=Path(os.environ.get('TMV2_TRAIN', ROOT/'redundancy_exp/train.csv'))
MAPX=Path(os.environ.get('TMV2_MAPPING', ROOT/'redundancy_exp/trackman_id_matching_report.xlsx'))
OUT=Path(os.environ.get('TMV2_OUT', ROOT/'tmv2_experiment')); OUT.mkdir(parents=True,exist_ok=True)
CACHE=OUT/'cache'; CACHE.mkdir(exist_ok=True)
PITCH_GROUPS=['fastball','breaking','offspeed']
METRICS=['rel_speed','spin_rate','induced_vert_break','horz_break','extension','rel_height','rel_side','zone_speed']
CTX_METRICS=['rel_side','rel_height','extension','rel_speed','spin_rate']
DRIFT_METRICS=['rel_speed','rel_height','rel_side','extension']

def norm_id(s): return s.astype('string').str.replace(r'\.0$','',regex=True)
def num(s): return pd.to_numeric(s,errors='coerce').replace([np.inf,-np.inf],np.nan)

def combine(parts, keys):
    z=pd.concat(parts,ignore_index=True)
    val=[c for c in z.columns if c not in keys]
    return z.groupby(keys,observed=True,dropna=False,sort=False)[val].sum().reset_index()

def aggregate_stream(chunksize=250_000):
    group_path=CACHE/'group_v2.pkl'; ctx_path=CACHE/'context_v2.pkl'; drift_path=CACHE/'drift_v2.pkl'
    if all(p.exists() for p in [group_path,ctx_path,drift_path]):
        print('CACHE_HIT group/context/drift',flush=True)
        return pd.read_pickle(group_path),pd.read_pickle(ctx_path),pd.read_pickle(drift_path)
    use=['season','pitcher_trackman_id','trackman_game_id','pitch_no','balls_before','strikes_before','batter_hand','pitch_type_group',*METRICS]
    gp=[]; cp=[]; dp=[]; total=0
    for ch in pd.read_csv(TRACKMAN,usecols=use,chunksize=chunksize,low_memory=False):
        total+=len(ch)
        ch['season']=num(ch.season); ch['pitcher_trackman_id']=norm_id(ch.pitcher_trackman_id)
        ch['pitch_type_group']=ch.pitch_type_group.astype('string').str.lower(); ch=ch[ch.pitch_type_group.isin(PITCH_GROUPS)].copy()
        ch['batter_hand']=ch.batter_hand.astype('string').fillna('__MISSING__')
        keys=['season','pitcher_trackman_id','pitch_type_group']; w=ch[keys].copy(); w['tm_group_pitch_n']=1
        for m in METRICS:
            x=num(ch[m]); w[f'{m}__n']=x.notna().astype('int32'); w[f'{m}__sum']=x.fillna(0.); w[f'{m}__sqsum']=x.pow(2).fillna(0.)
        for a,b in [('rel_side','rel_height'),('rel_height','extension')]:
            xa=num(ch[a]); xb=num(ch[b]); ok=xa.notna()&xb.notna(); w[f'{a}__{b}__n']=ok.astype('int32'); w[f'{a}__{b}__sumxy']=(xa*xb).where(ok,0.)
        gp.append(w.groupby(keys,observed=True,dropna=False,sort=False).sum(numeric_only=True).reset_index())
        ckeys=['season','pitcher_trackman_id','balls_before','strikes_before','batter_hand','pitch_type_group']; cw=ch[ckeys].copy(); cw['ctx_pitch_n']=1
        for m in CTX_METRICS:
            x=num(ch[m]); cw[f'{m}__n']=x.notna().astype('int32'); cw[f'{m}__sum']=x.fillna(0.); cw[f'{m}__sqsum']=x.pow(2).fillna(0.)
        cp.append(cw.groupby(ckeys,observed=True,dropna=False,sort=False).sum(numeric_only=True).reset_index())
        dkeys=['season','pitcher_trackman_id','trackman_game_id']; dw=ch[dkeys].copy(); xall=num(ch.pitch_no)
        for m in DRIFT_METRICS:
            y=num(ch[m]); ok=xall.notna()&y.notna(); x=xall.where(ok,0.); yy=y.where(ok,0.)
            dw[f'{m}__n']=ok.astype('int32'); dw[f'{m}__sum_x']=x; dw[f'{m}__sum_x2']=x.pow(2); dw[f'{m}__sum_y']=yy; dw[f'{m}__sum_xy']=x*yy
        dp.append(dw.groupby(dkeys,observed=True,dropna=False,sort=False).sum(numeric_only=True).reset_index())
        print('PASS1',total,flush=True)
    group=combine(gp,['season','pitcher_trackman_id','pitch_type_group'])
    ctx=combine(cp,['season','pitcher_trackman_id','balls_before','strikes_before','batter_hand','pitch_type_group'])
    drift=combine(dp,['season','pitcher_trackman_id','trackman_game_id'])
    group.to_pickle(group_path); ctx.to_pickle(ctx_path); drift.to_pickle(drift_path)
    return group,ctx,drift

def game_bounds(chunksize=300_000):
    p=CACHE/'game_bounds.pkl'
    if p.exists(): return pd.read_pickle(p)
    parts=[]; use=['season','pitcher_trackman_id','trackman_game_id','pitch_no']
    for ch in pd.read_csv(TRACKMAN,usecols=use,chunksize=chunksize,low_memory=False):
        ch['pitcher_trackman_id']=norm_id(ch.pitcher_trackman_id); ch['pitch_no']=num(ch.pitch_no)
        g=ch.groupby(['season','pitcher_trackman_id','trackman_game_id'],observed=True,dropna=False).pitch_no.agg(['min','max']).reset_index(); parts.append(g)
    z=pd.concat(parts,ignore_index=True)
    z=z.groupby(['season','pitcher_trackman_id','trackman_game_id'],observed=True,dropna=False).agg(min=('min','min'),max=('max','max')).reset_index()
    z.to_pickle(p); return z

def aggregate_early_late(chunksize=250_000):
    p=CACHE/'early_late_v2.pkl'
    if p.exists(): return pd.read_pickle(p)
    bounds=game_bounds(); parts=[]; use=['season','pitcher_trackman_id','trackman_game_id','pitch_no','rel_side','rel_height','extension']
    for ch in pd.read_csv(TRACKMAN,usecols=use,chunksize=chunksize,low_memory=False):
        ch['pitcher_trackman_id']=norm_id(ch.pitcher_trackman_id); ch['pitch_no']=num(ch.pitch_no)
        ch=ch.merge(bounds,on=['season','pitcher_trackman_id','trackman_game_id'],how='left')
        den=(ch['max']-ch['min']).replace(0,np.nan); prog=(ch.pitch_no-ch['min'])/den
        ch['phase']=np.where(prog<=1/3,'early',np.where(prog>=2/3,'late','middle'))
        ch=ch[ch.phase.isin(['early','late'])].copy(); keys=['season','pitcher_trackman_id','trackman_game_id','phase']; w=ch[keys].copy()
        for m in ['rel_side','rel_height','extension']:
            x=num(ch[m]); w[f'{m}__n']=x.notna().astype('int32'); w[f'{m}__sum']=x.fillna(0.); w[f'{m}__sqsum']=x.pow(2).fillna(0.)
        parts.append(w.groupby(keys,observed=True,dropna=False).sum(numeric_only=True).reset_index())
    z=combine(parts,['season','pitcher_trackman_id','trackman_game_id','phase']); z.to_pickle(p); return z

def mapping_table():
    if MAPX.suffix.lower() == '.csv': m=pd.read_csv(MAPX,encoding='utf-8-sig',low_memory=False)
    else: m=pd.read_excel(MAPX,sheet_name='투수 매핑',header=2)
    m['pitcher_id']=norm_id(m.pitcher_id); m['pitcher_trackman_id']=norm_id(m.pitcher_trackman_id)
    if '신뢰등급' in m.columns: m=m[m['신뢰등급'].astype('string').isin(['확정','높음'])]
    m=m[m.pitcher_trackman_id.notna()].drop_duplicates('pitcher_id')
    return m[['pitcher_id','pitcher_trackman_id']].set_index('pitcher_id').pitcher_trackman_id

def pooled_group(hist):
    if hist.empty: return pd.DataFrame()
    keys=['pitcher_trackman_id','pitch_type_group']; vals=[c for c in hist.columns if c not in ['season',*keys]]
    p=hist.groupby(keys,observed=True,sort=False)[vals].sum().reset_index()
    for m in METRICS:
        n=p[f'{m}__n'].astype(float); sm=p[f'{m}__sum'].astype(float); ss=p[f'{m}__sqsum'].astype(float)
        p[f'{m}_mean']=sm/n.replace(0,np.nan); var=(ss-sm.pow(2)/n.replace(0,np.nan))/(n-1).replace(0,np.nan); p[f'{m}_sd']=np.sqrt(var.clip(lower=0))
    for a,b in [('rel_side','rel_height'),('rel_height','extension')]:
        n=p[f'{a}__{b}__n'].astype(float); sumxy=p[f'{a}__{b}__sumxy'].astype(float)
        cov=(sumxy-(p[f'{a}__sum']*p[f'{b}__sum']/n.replace(0,np.nan)))/(n-1).replace(0,np.nan)
        va=p[f'{a}_sd'].pow(2); vb=p[f'{b}_sd'].pow(2); det=(va*vb-cov.pow(2)).clip(lower=0); p[f'ellipse_{a}_{b}']=np.sqrt(det)
    return p

def normalized_profile(p, raw_col, out_col, reliability_n_col='tm_group_pitch_n', shrink=100.):
    raw=num(p[raw_col]); med=p.groupby('pitch_type_group',observed=True)[raw_col].transform('median').replace(0,np.nan)
    rel=num(p[reliability_n_col])/(num(p[reliability_n_col])+shrink)
    return (1+rel*(raw/med-1)).astype('float32')

def row_expected(row_tm, row_rates, pivot, field):
    nume=pd.Series(0.,index=row_tm.index); den=pd.Series(0.,index=row_tm.index)
    for g in PITCH_GROUPS:
        key=(field,g); v=row_tm.map(pivot[key]) if key in pivot.columns else pd.Series(np.nan,index=row_tm.index)
        r=num(row_rates[g]).where(lambda x:x.between(0,1)); ok=r.notna()&v.notna(); nume += (r*v).where(ok,0.); den += r.where(ok,0.)
    return (nume/den.replace(0,np.nan)).astype('float32')

def game_slopes(d):
    r=d[['season','pitcher_trackman_id','trackman_game_id']].copy()
    for m in DRIFT_METRICS:
        n=num(d[f'{m}__n']); sx=num(d[f'{m}__sum_x']); sx2=num(d[f'{m}__sum_x2']); sy=num(d[f'{m}__sum_y']); sxy=num(d[f'{m}__sum_xy']); den=n*sx2-sx.pow(2)
        r[f'{m}_slope']=((n*sxy-sx*sy)/den.replace(0,np.nan)).where(n>=10)
    return r

def early_late_game(el):
    rows=[]
    for phase in ['early','late']:
        z=el[el.phase.eq(phase)].copy(); keep=['season','pitcher_trackman_id','trackman_game_id']
        for m in ['rel_side','rel_height','extension']:
            n=num(z[f'{m}__n']); sm=num(z[f'{m}__sum']); ss=num(z[f'{m}__sqsum']); var=(ss-sm.pow(2)/n.replace(0,np.nan))/(n-1).replace(0,np.nan); z[f'{m}_sd']=np.sqrt(var.clip(lower=0)).where(n>=4); keep.append(f'{m}_sd')
        z=z[keep].set_index(['season','pitcher_trackman_id','trackman_game_id']); z.columns=[f'{c}_{phase}' for c in z.columns]; rows.append(z)
    w=rows[0].join(rows[1],how='outer').reset_index(); ratios=[]
    for m in ['rel_side','rel_height','extension']: ratios.append(num(w[f'{m}_sd_late'])/num(w[f'{m}_sd_early']).replace(0,np.nan))
    w['release_late_early_dispersion_ratio']=pd.concat(ratios,axis=1).mean(axis=1,skipna=True).clip(0,10); return w

def build_features(group,ctx,drift,el):
    raw=pd.read_csv(TRAIN,usecols=['season','pitcher_id','pitcher_hand','batter_hand','balls_before','strikes_before','asof_pitcher_fastball_rate','asof_pitcher_breaking_rate','asof_pitcher_offspeed_rate'],low_memory=False)
    raw['season']=num(raw.season).astype(int); pid=norm_id(raw.pitcher_id); mp=mapping_table(); tm=pid.map(mp)
    row_bh=raw.batter_hand.astype('string').replace({'1':'Left','2':'Right','1.0':'Left','2.0':'Right'}).fillna('__MISSING__')
    F=pd.DataFrame(index=raw.index); gsl=game_slopes(drift); elg=early_late_game(el)
    for S in sorted(raw.season.unique()):
        idx=raw.index[raw.season.eq(S)]; rtm=tm.loc[idx]; hist=group[num(group.season)<S]; pg=pooled_group(hist)
        if pg.empty: continue
        pg['n_rel_side_disp']=normalized_profile(pg,'rel_side_sd','n_rel_side_disp'); pg['n_rel_height_disp']=normalized_profile(pg,'rel_height_sd','n_rel_height_disp'); pg['n_extension_disp']=normalized_profile(pg,'extension_sd','n_extension_disp'); pg['n_ivb_disp']=normalized_profile(pg,'induced_vert_break_sd','n_ivb_disp'); pg['n_hb_disp']=normalized_profile(pg,'horz_break_sd','n_hb_disp'); pg['n_coronal_ellipse']=normalized_profile(pg,'ellipse_rel_side_rel_height','n_coronal_ellipse'); pg['n_sagittal_ellipse']=normalized_profile(pg,'ellipse_rel_height_extension','n_sagittal_ellipse')
        pivot=pg.pivot(index='pitcher_trackman_id',columns='pitch_type_group')
        rr=pd.DataFrame({g:num(raw.loc[idx,f'asof_pitcher_{g}_rate']) for g in PITCH_GROUPS},index=idx)
        fields={'tm_v2_release_side_dispersion':'n_rel_side_disp','tm_v2_release_height_dispersion':'n_rel_height_disp','tm_v2_extension_dispersion':'n_extension_disp','tm_v2_release_coronal_ellipse':'n_coronal_ellipse','tm_v2_release_sagittal_ellipse':'n_sagittal_ellipse','tm_v2_ivb_dispersion':'n_ivb_disp','tm_v2_horz_break_dispersion':'n_hb_disp'}
        for out,field in fields.items(): F.loc[idx,out]=row_expected(rtm,rr,pivot,field)
        hc=ctx[num(ctx.season)<S]
        if not hc.empty:
            keys=['pitcher_trackman_id','balls_before','strikes_before','batter_hand','pitch_type_group']; vals=[c for c in hc.columns if c not in ['season',*keys]]; pc=hc.groupby(keys,observed=True,dropna=False,sort=False)[vals].sum().reset_index()
            cidx=pd.MultiIndex.from_arrays([rtm.astype('string'),num(raw.loc[idx,'balls_before']),num(raw.loc[idx,'strikes_before']),row_bh.loc[idx].astype('string')],names=['pitcher_trackman_id','balls_before','strikes_before','batter_hand'])
            for metric in CTX_METRICS:
                val_by_group={}
                for g in PITCH_GROUPS:
                    z=pc[pc.pitch_type_group.eq(g)].copy(); nctx=num(z[f'{metric}__n']); cm=num(z[f'{metric}__sum'])/nctx.replace(0,np.nan); s=pd.Series(cm.to_numpy(),index=pd.MultiIndex.from_frame(z[['pitcher_trackman_id','balls_before','strikes_before','batter_hand']]))
                    cmean=pd.Series(s.reindex(cidx).to_numpy(),index=idx); cn=pd.Series(pd.Series(nctx.to_numpy(),index=s.index).reindex(cidx).to_numpy(),index=idx); gmean=rtm.map(pivot[(f'{metric}_mean',g)]) if (f'{metric}_mean',g) in pivot.columns else pd.Series(np.nan,index=idx); rel=cn/(cn+50.); val_by_group[g]=rel*(cmean-gmean)
                nume=pd.Series(0.,index=idx); den=pd.Series(0.,index=idx)
                for g in PITCH_GROUPS:
                    r=num(rr[g]).where(lambda x:x.between(0,1)); v=val_by_group[g]; ok=r.notna()&v.notna(); nume+=(r*v).where(ok,0.); den+=r.where(ok,0.)
                short={'rel_side':'release_side','rel_height':'release_height','extension':'extension','rel_speed':'velocity','spin_rate':'spin'}[metric]; F.loc[idx,f'tm_v2_ctx_{short}_shift']=(nume/den.replace(0,np.nan)).astype('float32')
        hs=gsl[num(gsl.season)<S]
        if not hs.empty:
            ps=hs.groupby('pitcher_trackman_id',observed=True).agg(rel_side=('rel_side_slope','mean'),rel_height=('rel_height_slope','mean'),extension=('extension_slope','mean')); F.loc[idx,'tm_v2_rel_side_drift_slope']=rtm.map(ps.rel_side).astype('float32'); F.loc[idx,'tm_v2_rel_height_drift_slope']=rtm.map(ps.rel_height).astype('float32'); F.loc[idx,'tm_v2_extension_drift_slope']=rtm.map(ps.extension).astype('float32')
        he=elg[num(elg.season)<S]
        if not he.empty:
            pr=he.groupby('pitcher_trackman_id',observed=True).release_late_early_dispersion_ratio.mean(); F.loc[idx,'tm_v2_late_early_release_dispersion_ratio']=rtm.map(pr).astype('float32')
        prev=group[num(group.season).eq(S-1)]; older=group[num(group.season)<S-1]; pprev=pooled_group(prev); pold=pooled_group(older)
        if not pprev.empty and not pold.empty:
            pv=pprev.pivot(index='pitcher_trackman_id',columns='pitch_type_group'); ov=pold.pivot(index='pitcher_trackman_id',columns='pitch_type_group')
            for metric,short in [('rel_side','release_side'),('rel_height','release_height'),('extension','extension')]:
                nume=pd.Series(0.,index=idx); den=pd.Series(0.,index=idx)
                for g in PITCH_GROUPS:
                    k=(f'{metric}_mean',g); a=rtm.map(pv[k]) if k in pv.columns else pd.Series(np.nan,index=idx); b=rtm.map(ov[k]) if k in ov.columns else pd.Series(np.nan,index=idx); r=num(rr[g]).where(lambda x:x.between(0,1)); v=a-b; ok=r.notna()&v.notna(); nume+=(r*v).where(ok,0.); den+=r.where(ok,0.)
                F.loc[idx,f'tm_v2_prev_vs_career_{short}_shift']=(nume/den.replace(0,np.nan)).astype('float32')
        usage=hist.groupby(['pitcher_trackman_id','pitch_type_group'],observed=True).tm_group_pitch_n.sum().reset_index(); up=usage.pivot(index='pitcher_trackman_id',columns='pitch_type_group',values='tm_group_pitch_n').fillna(0.); ur=up.div(up.sum(axis=1).replace(0,np.nan),axis=0); pp=pg.set_index(['pitcher_trackman_id','pitch_type_group']); prof=pd.DataFrame(index=up.index)
        for metric in ['rel_height_mean','rel_side_mean']:
            vv=pd.DataFrame(index=up.index)
            for g in PITCH_GROUPS:
                s=pp[metric].xs(g,level='pitch_type_group') if g in pp.index.get_level_values('pitch_type_group') else pd.Series(dtype=float); vv[g]=vv.index.map(s)
            prof[metric]=(vv*ur.reindex(vv.index)).sum(axis=1,min_count=1)
        st=pd.DataFrame(index=up.index)
        for fld in ['n_rel_side_disp','n_rel_height_disp','n_extension_disp']:
            vv=pd.DataFrame(index=up.index)
            for g in PITCH_GROUPS:
                s=pp[fld].xs(g,level='pitch_type_group') if g in pp.index.get_level_values('pitch_type_group') else pd.Series(dtype=float); vv[g]=vv.index.map(s)
            st[fld]=(vv*ur.reindex(vv.index)).sum(axis=1,min_count=1)
        prof['stability']=st.mean(axis=1)
        def qbin(series,q=3):
            good=series.dropna()
            if good.nunique()<q: return pd.Series('__MISSING__',index=series.index,dtype='string')
            edges=np.unique(np.quantile(good,np.linspace(0,1,q+1)))
            if len(edges)<3: return pd.Series('__MISSING__',index=series.index,dtype='string')
            return pd.cut(series,bins=edges,include_lowest=True,duplicates='drop',labels=False).astype('Int64').astype('string').fillna('__MISSING__')
        hb=qbin(prof.rel_height_mean,3); sb=qbin(prof.rel_side_mean,3); slot=(hb+'_'+sb).astype('string'); stab=qbin(prof.stability,4)
        F.loc[idx,'tm_v2_release_slot_archetype']=rtm.map(slot).astype('string').fillna('__MISSING__').astype(str); F.loc[idx,'tm_v2_release_stability_tier']=rtm.map(stab).astype('string').fillna('__MISSING__').astype(str)
        if not he.empty:
            fat=he.groupby('pitcher_trackman_id',observed=True).release_late_early_dispersion_ratio.mean(); ft=qbin(fat,4); F.loc[idx,'tm_v2_fatigue_tier']=rtm.map(ft).astype('string').fillna('__MISSING__').astype(str)
        else: F.loc[idx,'tm_v2_fatigue_tier']='__MISSING__'
        print('FEATURE_SEASON_DONE',S,flush=True)
    cats=['tm_v2_release_slot_archetype','tm_v2_release_stability_tier','tm_v2_fatigue_tier']
    for c in F.columns:
        if c in cats: F[c]=F[c].astype('string').fillna('__MISSING__').astype(str)
        else: F[c]=num(F[c]).astype('float32')
    F.to_pickle(OUT/'tmv2_new_features.pkl'); (OUT/'tmv2_new_feature_names.json').write_text(json.dumps({'all':F.columns.tolist(),'categorical':cats},ensure_ascii=False,indent=2)); print('SAVED',F.shape,flush=True)

if __name__=='__main__':
    t=time.time(); group,ctx,drift=aggregate_stream(); el=aggregate_early_late(); print('AGG_DONE',group.shape,ctx.shape,drift.shape,el.shape,'sec',time.time()-t,flush=True); build_features(group,ctx,drift,el)
