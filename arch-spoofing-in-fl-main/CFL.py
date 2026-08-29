"""
Demonstration of architecture-spoofing attack and fingerprinting defence on
architecture-clustered FL (example IDS data, non-IID).

Runs three conditions and reports the comparison the report evaluates:
  1. baseline (no attack)
  2. attack: client 0 spoofs its architecture into the DNN cluster, then sign-flips
  3. attack + behavioural-fingerprinting defence

Also writes CSVs and plots (into results/) for the report: convergence, fairness,
detection quality, attack success rate, and a per-client audit trail.

Regenerate data first:  python make_example_data.py
Then:                    python CFL.py
"""

import analysis
from run_arch_cfl import main

RESULTS_DIR = "results"
VICTIM = "dnn"


if __name__ == "__main__":
    common = dict(rounds=3, epochs=5, num_clients=6,
                  malicious_clients=(0,), spoof_as="dnn")

    print("\n========== 1. BASELINE (no attack) ==========")
    base = main(**common, enable_arch_spoof=False, enable_sign_flip=False)

    print("\n========== 2. ATTACK (spoof + poison, no defence) ==========")
    atk = main(**common, enable_arch_spoof=True, enable_sign_flip=True)

    print("\n========== 3. ATTACK + FINGERPRINTING DEFENCE ==========")
    dfd = main(**common, enable_arch_spoof=True, enable_sign_flip=True, enable_defense=True)

    runs = {"baseline": base, "attack (no defence)": atk, "attack (defended)": dfd}

    print("\n========== SUMMARY ==========")
    for label, history in runs.items():
        print(f"\n--- {label} ---")
        print(analysis.summarize(history, family=VICTIM))

    success_rate = analysis.attack_success_rate(base, atk, VICTIM)
    degradation = analysis.victim_degradation(base, atk, VICTIM)
    recovered = analysis.final_accuracy(dfd, VICTIM) - analysis.final_accuracy(atk, VICTIM)
    print(f"\nAttack success rate (rounds corrupted)   : {success_rate:.0%}")
    print(f"Victim ('{VICTIM}') degradation, no defence : {degradation:+.4f}")
    print(f"Victim ('{VICTIM}') accuracy gained back by defence : {recovered:+.4f}")

    print("\n========== EXPORTING ANALYSIS ==========")
    analysis.save_round_csv(runs, f"{RESULTS_DIR}/round_metrics.csv")
    analysis.save_client_csv(runs, f"{RESULTS_DIR}/client_log.csv")
    analysis.plot_accuracy(runs, VICTIM, f"{RESULTS_DIR}/accuracy_{VICTIM}.png",
                           title=f"Convergence under attack/defence: '{VICTIM}' cluster")
    analysis.plot_fairness(runs, f"{RESULTS_DIR}/fairness.png")
    analysis.plot_detection(dfd, f"{RESULTS_DIR}/detection.png")
