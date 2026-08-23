from __future__ import annotations
import argparse,gc,time,json,os
from pathlib import Path
import numpy as np,pandas as pd
from catboost import CatBoostClassifier,Pool,FeaturesData
from sklearn.metrics import brier_score_loss,roc_auc_score

ROOT=Path(os.environ.get('TMV2_ROOT','/mnt/data')); OUT=Path(os.environ.get('TMV2_OUT',ROOT/'tmv2_experiment')); OUT.mkdir(parents=True,exist_ok=True)
BASE=ROOT/'native_handfix_run/base89.pkl'; TM=ROOT/'native_handfix_run/tm_fix.pkl'; NEW=OUT/'tmv2_new_features.pkl'; META=ROOT/'native_handfix_run/meta.pkl'
OLD_CAT=['top_bottom','game_type','pitcher_hand','batter_hand','pitcher_team_id','batter_team_id','base_state','count_state','hand_pair','base_out_state','tm_mapping_grade']
NEW_CAT=['tm_v2_release_slot_archetype','tm_v2_release_stability_tier','tm_v2_fatigue_tier']
BLOCKS={
 'release_axis':['tm_v2_release_side_dispersion','tm_v2_release_height_dispersion','tm_v2_extension_dispersion'],
 'ellipse':['tm_v2_release_coronal_ellipse','tm_v2_release_sagittal_ellipse'],
 'context':['tm_v2_ctx_release_side_shift','tm_v2_ctx_release_height_shift','tm_v2_ctx_extension_shift','tm_v2_ctx_velocity_shift','tm_v2_ctx_spin_shift'],
 'fatigue':['tm_v2_rel_side_drift_slope','tm_v2_rel_height_drift_slope','tm_v2_extension_drift_slope','tm_v2_late_early_release_dispersion_ratio'],
 'movement':['tm_v2_ivb_dispersion','tm_v2_horz_break_dispersion'],
 'mechanics_change':['tm_v2_prev_vs_career_release_side_shift','tm_v2_prev_vs_career_release_height_shift','tm_v2_prev_vs_career_extension_shift'],
 'archetype':NEW_CAT,
}
OLD_REMOVE=['tm_available','tm_mapping_purity','tm_mapping_grade','tm_expected_release_dispersion','tm_expected_movement_dispersion','tm_context_expected_release_dispersion','tm_context_expected_movement_dispersion','tm_release_drift_abs_slope']

def metric(y,p):
 r=float(y.mean()); base=r*(1-r); b=float(brier_score_loss(y,p)); return dict(n=len(y),target_rate=r,prediction_mean=float(p.mean()),bias=float(p.mean()-r),brier=b,score=max(0.,100000*(1-b/base)),auc=float(roc_auc_score(y,p)))

def cat_matrix(df,cats):
 a=np.empty((len(df),len(cats)),dtype=object)
 for j,c in enumerate(cats):
  vals=df[c].astype('string').fillna('__MISSING__'); codes,uniques=pd.factorize(vals,sort=True); lut=np.array([str(x).encode('utf-8') for x in uniques],dtype=object); a[:,j]=lut[codes]
 return a

def variants():
 v={}; v['A_baseline104']=[]; cur=[]
 for key,label in [('release_axis','B_release_axis'),('ellipse','C_plus_ellipse'),('context','D_plus_true_context'),('fatigue','E_plus_fatigue'),('movement','F_plus_movement'),('mechanics_change','G_plus_mechanics_change')]:
  cur=cur+BLOCKS[key]; v[label]=cur.copy()
 v['H_G_clean_old_tm']=cur.copy(); v['I_G_clean_plus_archetype']=cur+BLOCKS['archetype']
 for k in ['ellipse','context','fatigue','movement','mechanics_change','archetype']:
  v['X_'+k+'_only']=BLOCKS[k].copy()
 v['X_release_archetype']=BLOCKS['release_axis']+BLOCKS['archetype']; v['X_mechchange_archetype']=BLOCKS['mechanics_change']+BLOCKS['archetype']; v['X_baseline_clean']=[]; v['X_clean_archetype']=BLOCKS['archetype'].copy()
 full=[]
 for k in ['release_axis','ellipse','context','fatigue','movement','mechanics_change']: full += BLOCKS[k]
 full += BLOCKS['archetype']
 for k in ['release_axis','ellipse','context','fatigue','movement','mechanics_change']: v['I_minus_'+k]=[x for x in full if x not in BLOCKS[k]]
 return v

def run_one(label,valid,iterations=216):
 meta=pd.read_pickle(META); mask=meta.season.le(valid); meta=meta.loc[mask].reset_index(drop=True); y=meta.control_success.to_numpy('int8'); season=meta.season.to_numpy(int); del meta; gc.collect()
 base=pd.read_pickle(BASE).loc[mask].reset_index(drop=True); tm=pd.read_pickle(TM).loc[mask].reset_index(drop=True); new=pd.read_pickle(NEW).loc[mask].reset_index(drop=True)
 adds=variants()[label]; X=pd.concat([base,tm],axis=1); del base,tm; gc.collect()
 if adds: X=pd.concat([X,new[adds]],axis=1)
 del new; gc.collect()
 if label.startswith('H_') or label.startswith('I_') or label in ['X_baseline_clean','X_clean_archetype'] or label.startswith('I_minus_'): X=X.drop(columns=[c for c in OLD_REMOVE if c in X.columns])
 cats=[c for c in OLD_CAT+NEW_CAT if c in X.columns]; nums=[c for c in X.columns if c not in cats]; tr=season<valid; va=season==valid
 num=np.ascontiguousarray(X[nums].to_numpy(dtype=np.float32,copy=True)); cat=cat_matrix(X,cats); del X; gc.collect(); tr_idx=np.flatnonzero(tr); va_idx=np.flatnonzero(va)
 train_fd=FeaturesData(num_feature_data=np.ascontiguousarray(num[tr_idx]),cat_feature_data=cat[tr_idx].copy(),num_feature_names=nums,cat_feature_names=cats); valid_fd=FeaturesData(num_feature_data=np.ascontiguousarray(num[va_idx]),cat_feature_data=cat[va_idx].copy(),num_feature_names=nums,cat_feature_names=cats)
 train_pool=Pool(train_fd,label=y[tr_idx]); valid_pool=Pool(valid_fd,label=y[va_idx]); del num,cat,train_fd,valid_fd,tr_idx,va_idx; gc.collect()
 model=CatBoostClassifier(loss_function='Logloss',eval_metric='BrierScore',iterations=iterations,learning_rate=.03,depth=7,l2_leaf_reg=5,random_strength=.5,random_seed=42,max_ctr_complexity=1,one_hot_max_size=255,border_count=128,allow_writing_files=False,verbose=0,thread_count=-1,task_type='CPU',used_ram_limit='4gb')
 t=time.time(); model.fit(train_pool); p=model.predict_proba(valid_pool)[:,1]; elapsed=time.time()-t; m=metric(y[va],p); m.update(label=label,valid_season=valid,iterations=iterations,feature_count=len(nums)+len(cats),categorical_count=len(cats),elapsed_seconds=elapsed)
 imp=pd.DataFrame({'feature':nums+cats,'importance':model.get_feature_importance(type='PredictionValuesChange')}).sort_values('importance',ascending=False); imp.to_csv(OUT/f'importance_{label}_{valid}.csv',index=False)
 pd.DataFrame({'control_success':y[va],'pred':p}).to_csv(OUT/f'pred_{label}_{valid}.csv.gz',index=False,compression='gzip'); pd.DataFrame([m]).to_csv(OUT/f'score_{label}_{valid}.csv',index=False); print('RESULT',json.dumps(m),flush=True); return m

if __name__=='__main__':
 ap=argparse.ArgumentParser(); ap.add_argument('--label',required=True); ap.add_argument('--valid',type=int,default=2023); ap.add_argument('--iterations',type=int,default=216); a=ap.parse_args(); run_one(a.label,a.valid,a.iterations)
