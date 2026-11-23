"""Frozen, explicit round-two protocol. No changes to round-one implementation."""
import json,hashlib
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
from r0.runtime import read_config,file_sha256,write_json
REPO=Path(__file__).resolve().parents[2]
R0=Path('/root/autodl-tmp/experiments/R0');ROOT=R0/'controls_round2'
SOURCES=['S1','S2'];SEEDS=[73,74,75]
METHODS=['trained73','trained74','trained75','random73','random74','random75','l2','energy','channel_variance','mean_error']
LEVELS=['s0_stride8','s1_stride16','s2_stride32','s3_stride64','overall']
PYTHON='/root/autodl-tmp/experiments/B0_PROB_OWOBJ/env/bin/python'
def config(source='S2'):return read_config(REPO/f'r0/configs/R0-{source}.json')
def original_checkpoint(source):return R0/f'R0-{source}/attempt_00/checkpoint.pth'
def stamp():return datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(timespec='seconds')
def source_hashes():
    return {str(p.relative_to(REPO)):file_sha256(p) for folder in ['r0','models','datasets','util'] for p in (REPO/folder).rglob('*.py') if '__pycache__' not in p.parts}
def sequence():return (R0/'formal_20261111/train_image_sequence.txt').read_text().splitlines()
def train_path(source,seed):return original_checkpoint(source) if seed==73 else ROOT/'training'/f'{source}_seed{seed}'/'checkpoint.pth'
def init_path(seed):return ROOT/'initializations'/f'seed{seed}.pth'
def manifest_check(path):
    x=json.loads(Path(path).read_text())
    for name,h in x.get('sha256',{}).items():
        if file_sha256(Path(path).parent/name)!=h:raise RuntimeError('Corrupt artifact: '+name)
    return x
PROTOCOL={
 'version':'R0-controls-round2-v1','seed_policy':{'training':[73,74,75],'new_training':[74,75],'image_order_seed':73,'training_steps':2000,'evaluation_mask_seed':73,'evaluation_repeats':4},
 'directions':{m:'higher is foreground; fixed before controls evaluation' for m in METHODS},
 'simple_maps':{'l2':'sqrt(sum_c f_c^2) per spatial point','energy':'mean_c f_c^2 per spatial point','channel_variance':'mean_c (f_c-mean_c(f_c))^2, population variance','zero_prediction_mse':'exact alias of energy; no independent evidence','mean_error':'mean_c (f_c-mu_c)^2; mu is valid-position-weighted training-2000-image feature mean at each source/scale'},
 'aggregation':'channel statistic first at each spatial position, then original support-normalized ROIAlign; overall is raw arithmetic mean of four valid scale scores',
 'ae':'original four independent scale AEs; original mask .6, valid-position MSE, AdamW, gradient clip and 2000-step budget unchanged',
 'random_initialization':'independent seeds73/74/75; emulate original seed -> frozen detector construction -> bank initialization RNG sequence; same untrained bank used for S1/S2 because original initial banks were identical',
 'masking':'random and trained banks use identical image/level/4-repeat masks from original seed73 evaluation; constant predictors ignore mask content but share targets/support',
 'candidate_pools':['original GT-reference','all 100 original final queries'],
 'primary_background':'disjoint_gt; preserve original iou_only labels for provenance, not primary reliable-background claim',
 'geometry_matching':{'features':['ln original pixel area','ln relative image area','ln width/height'],'joint_caliper':'absolute difference <= ln(1.25) in EACH of three features','selection':'global nearest Euclidean log-geometry without replacement within pool/comparison; positives in fixed SHA256(image_id/pool/candidate_id/seed73) order; ties canonical candidate index','unmatched':'retain every unmatched ID and reason; report retention and limited common-support interpretation; never use a score to match','comparisons':['known','unknown','foreground','b0_raw_uncovered','b0_missed_B_no_candidate'],'small_object_check':'fixed original pixel area <32^2, medium <96^2 and large; matched pairs retain their positive area stratum'},
 'query_scores':{'gaussian':'exp(-(1.3/256)*pred_obj)','classifier_unknown':'sigmoid(pred_logits[:,-1])','context':'same batch1 forward as all spatial controls; original query ID0..99 and raw + clipped box hashes recorded; no GT-reference Gaussian proxy'},
 'query_diagnostics':{'low_gaussian':'bottom 25% of Gaussian scores over ALL query candidates before using group labels','fixed_strata':'Gaussian all-query score deciles; compare Unknown vs background within each; no calibration/fusion training','correlations':'Spearman and Kendall ordering agreement on exact common query IDs; rank displacements for true Unknown queries'},
 'statistics':{'bootstrap_unit':'unique image, same 200 multinomial bootstrap draws seed73 for every method and paired difference','primary_comparisons':'trained73 vs every predeclared control; also three-seed paired trained-minus-matching-random and mean/energy controls; no best-control selection','seed_summary':'all73/74/75 values, sample mean and sample std ddof1; bootstrap describes image sampling not training-seed uncertainty'},
 'no_test_selection':True,'shutdown':'pipeline writes exit status; never shuts down AutoDL'
}
