from deep_gw_pe_followup.utils.calc_numerical_posterior_odds import extract_res_info
from bilby.gw.result import CBCResult
from deep_gw_pe_followup.utils.get_odds_summary import get_odds_summary
from tabulate import tabulate

TABLE_HEADER = ["Pts", "KDE Post Odds", "Prior Odds", "Bayes Factor", "'Deep' Post Odds"]


def print_gw150914_odds():
    gw150914_mateu = CBCResult.from_json("event/gw150914/out_res/Mateu_GW150914_result.json")
    gw150914_ptA = extract_res_info("event/gw150914/out_res/ptA_0_result.json")
    gw150914_ptB = extract_res_info("event/gw150914/out_res/ptC_0_result.json")

    print(f"GW150914 PtA q-xeff: {gw150914_ptA['q'], gw150914_ptA['xeff']}")
    print(f"GW150914 PtB q-xeff: {gw150914_ptB['q'], gw150914_ptB['xeff']}")

    table = [
        TABLE_HEADER,
        get_odds_summary(gw150914_mateu, gw150914_ptA, gw150914_ptB, "GW150914 A VS B")
    ]
    print(tabulate(table))


def print_gw151226_odds():
    gw151226_mateu = CBCResult.from_json(
        "event/gw151226/out_res/S2_GW151226_IMRPhenomXPHM_N512_NA10_Dmarg_merged_result.json")
    gw151226_ptMateu = extract_res_info("event/gw151226/out_res/mateu_2_0_result.json")
    gw151226_ptChia = extract_res_info("event/gw151226/out_res/chia_2_0_result.json")
    gw151226_ptNitz = extract_res_info("event/gw151226/out_res/nitz_2_0_result.json")

    table = [
        TABLE_HEADER,
        get_odds_summary(gw151226_mateu, gw151226_ptMateu, gw151226_ptChia, "GW151226 Mateu Vs Chia"),
        get_odds_summary(gw151226_mateu, gw151226_ptMateu, gw151226_ptNitz, "GW151226 Mateu Vs Nitz")
    ]

    print(tabulate(table))


def print_injection_odds():
    injection_mateu = CBCResult.from_json(
        "injection_study/gw151226_like_injection/general_pe/out_nlive2000_nact10/result/GW151226_injection_0_result.json")
    injection_ptA = extract_res_info(
        "injection_study/gw151226_like_injection/followup_pt2/out_followup2/result/followup2_0_result.json")
    injection_ptB = extract_res_info(
        "injection_study/gw151226_like_injection/followup_pt1/out_followup1/result/followup1_0_result.json")

    table = [
        TABLE_HEADER,
        get_odds_summary(injection_mateu, injection_ptA, injection_ptB, "Injection A VS B"),
    ]

    print(tabulate(table))


def print_extra_kde_odds():
    mateu_kde = CBCResult.from_json("event/gw150914/out_res/Mateu_GW150914_result.json")
    nitz_kde = CBCResult.from_json("event/gw151226/out_res/nitz_2_0_result.json")
    chia_kde = CBCResult.from_json("event/gw151226/out_res/chia_2_0_result.json")

    mateu_pt = extract_res_info("event/gw150914/out_res/mateu_2_0_result.json")
    nitz_pt = extract_res_info("event/gw151226/out_res/nitz_2_0_result.json")
    chia_pt = extract_res_info("event/gw151226/out_res/chia_2_0_result.json")

    table = [
        TABLE_HEADER,
        get_odds_summary(mateu_kde, mateu_pt, nitz_pt, "GW150914-Mateu Mateu VS Nitz"),
        get_odds_summary(mateu_kde, mateu_pt, chia_pt, "GW150914-Mateu Mateu VS Chia"),
        get_odds_summary(nitz_kde, nitz_pt, chia_pt, "GW151226-Nitz Nitz VS Chia"),
    ]

    print(tabulate(table))


if __name__ == "__main__":
    print_gw150914_odds()
    print_gw151226_odds()
    print_injection_odds()
