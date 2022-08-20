mkdir -p ./event/gw151226_ias_points_no_m1m2_reweighting/outdir_highq/result/
mkdir -p ./event/gw151226_ias_points_no_m1m2_reweighting/outdir_lowq/result/


mkdir -p ./injection_study/gw151226_like_injection/followup_pt1/out_followup1/result
mkdir -p ./injection_study/gw151226_like_injection/followup_pt2/out_followup2/result
mkdir -p ./injection_study/gw151226_like_injection/general_pe/out_nlive2000_nact10/result


scp avajpeyi@ozstar.swin.edu.au:/fred/oz980/avajpeyi/projects/deep_gw_pe_followup/studies/event/gw151226_ias_points_no_m1m2_reweighting/outdir_highq/result/highq_0_result.json ./event/gw151226_ias_points_no_m1m2_reweighting/outdir_highq/result/highq_0_result.json
scp avajpeyi@ozstar.swin.edu.au:/fred/oz980/avajpeyi/projects/deep_gw_pe_followup/studies/event/gw151226_ias_points_no_m1m2_reweighting/outdir_loqq/result/lowq_0_result.json ./event/gw151226_ias_points_no_m1m2_reweighting/outdir_lowq/result/lowq_0_result.json


scp avajpeyi@ozstar.swin.edu.au:/fred/oz980/avajpeyi/projects/deep_gw_pe_followup/studies/injection_study/gw151226_like_injection/followup_pt1/out_followup1/result/followup1_0_result.json ./injection_study/gw151226_like_injection/followup_pt1/out_followup1/result/followup1_0_result.json
scp avajpeyi@ozstar.swin.edu.au:/fred/oz980/avajpeyi/projects/deep_gw_pe_followup/studies/injection_study/gw151226_like_injection/followup_pt2/out_followup2/result/followup2_0_result.json ./injection_study/gw151226_like_injection/followup_pt2/out_followup2/result/followup2_0_result.json
scp avajpeyi@ozstar.swin.edu.au:/fred/oz980/avajpeyi/projects/deep_gw_pe_followup/studies/injection_study/gw151226_like_injection/general_pe/out_nlive2000_nact10/result/GW151226_injection_0_result.json ./injection_study/gw151226_like_injection/general_pe/out_nlive2000_nact10/result/GW151226_injection_0_result.json
