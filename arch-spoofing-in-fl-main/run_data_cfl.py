"""Distribution-based clustered FL (+ optional metadata / weight attacks)."""

from attacks import CompositeAttack, LabelHistSpoof, NoAttack, WeightSignFlip
from clustering import DistributionClusterer
from data import DEFAULT_CSV, prepare_fl_data
from defense import DistributionFingerprintVerifier
from fl_loop import FederatedServer
from seeding import set_global_seeds


def main(
    rounds=2,
    epochs=5,
    num_clients=6,
    n_clusters=2,
    dirichlet_alpha=0.5,
    csv_path=DEFAULT_CSV,
    malicious_clients=(0,),
    enable_label_hist_spoof=False,
    enable_sign_flip=False,
    spoof_target_class=0,
    base_arch="mlp",
    enable_defense=False,
    seed=42,                    # vary across repeat runs to get error bars
    weighted_aggregation=True,  # False = unweighted mean (aggregation ablation)
):
    # Pin every RNG before the first model is built, so two conditions that
    # differ only in the attack flag really do differ only in that.
    set_global_seeds(seed)

    fl_data = prepare_fl_data(
        csv_path=csv_path,
        num_clients=num_clients,
        partition="dirichlet",
        dirichlet_alpha=dirichlet_alpha,
        random_state=seed,
    )

    # All clients share one architecture; clustering is on data metadata.
    client_archs = [base_arch] * num_clients

    attacks = []
    if enable_label_hist_spoof:
        attacks.append(
            LabelHistSpoof(
                malicious_clients=malicious_clients,
                target_class=spoof_target_class,
                num_classes=fl_data.num_classes,
            )
        )
    if enable_sign_flip:
        attacks.append(WeightSignFlip(malicious_clients=malicious_clients))
    attack = CompositeAttack(*attacks) if attacks else NoAttack()

    server = FederatedServer(
        input_dim=fl_data.input_dim,
        num_classes=fl_data.num_classes,
        clusterer=DistributionClusterer(n_clusters=n_clusters),
        client_archs=client_archs,
        attack=attack,
        base_arch=base_arch,
        mode="distribution",
        defense=DistributionFingerprintVerifier(dirichlet_alpha=dirichlet_alpha, seed=seed)
        if enable_defense else None,
        weighted_aggregation=weighted_aggregation,
    )
    return server.fit(
        fl_data.client_data,
        fl_data.X_test,
        fl_data.y_test,
        rounds=rounds,
        epochs=epochs,
        # Server-held reference data, never seen by clients and disjoint from
        # the test set -- see fl_loop.FederatedServer.fit.
        X_ref=fl_data.X_ref,
        y_ref=fl_data.y_ref,
    )


def demo(results_dir: str = "results", malicious_client: int = 0):
    """Three-condition comparison: baseline, attack, attack with the defence.

    Folded in from the former `DCFL.py`, which was a thin runner around this
    module's `main()` and nothing else. Keeping it here means the distribution
    demo lives beside the loop it drives, matching how `CFL.py` sits beside
    `run_arch_cfl.py`.

    Distribution-mode cluster ids are reassigned by the clustering every round
    and do not reliably name the same group of clients across rounds, unlike
    architecture mode's fixed "mlp"/"dnn". So this tracks the accuracy of
    whichever cluster the malicious client actually landed in each round rather
    than assuming a fixed name; see `analysis.client_cluster_accuracy`.

    Regenerate the data first with `python make_example_data.py`.
    """
    import analysis

    common = dict(rounds=3, epochs=3, num_clients=6, n_clusters=2,
                  dirichlet_alpha=0.5, malicious_clients=(malicious_client,),
                  spoof_target_class=0)

    print("\n========== 1. BASELINE (no attack) ==========")
    base = main(**common, enable_label_hist_spoof=False, enable_sign_flip=False)

    print("\n========== 2. ATTACK (label-hist spoof + poison, no defence) ==========")
    atk = main(**common, enable_label_hist_spoof=True, enable_sign_flip=True)

    print("\n========== 3. ATTACK + FINGERPRINTING DEFENCE ==========")
    dfd = main(**common, enable_label_hist_spoof=True, enable_sign_flip=True,
               enable_defense=True)

    print("\n========== VICTIM CLUSTER (wherever the attacker actually landed) ==========")
    for label, history in [("baseline", base), ("attack (no defence)", atk),
                           ("attack (defended)", dfd)]:
        print(f"{label}:")
        for row in analysis.client_cluster_accuracy(history, malicious_client):
            print(f"  round {row['round']}: cluster={row['cluster']} "
                  f"accuracy={row['accuracy']:.4f}")

    finals = {k: analysis.client_cluster_accuracy(h, malicious_client)[-1]["accuracy"]
              for k, h in (("base", base), ("atk", atk), ("dfd", dfd))}
    print(f"\nVictim degradation, no defence         : "
          f"{finals['base'] - finals['atk']:+.4f}")
    print(f"Victim accuracy gained back by defence : "
          f"{finals['dfd'] - finals['atk']:+.4f}")

    print("\n========== EXPORTING ANALYSIS ==========")
    runs = {"baseline": base, "attack (no defence)": atk, "attack (defended)": dfd}
    analysis.save_round_csv(runs, f"{results_dir}/dist_round_metrics.csv")
    analysis.save_client_csv(runs, f"{results_dir}/dist_client_log.csv")
    analysis.plot_fairness(runs, f"{results_dir}/dist_fairness.png",
                           title="Cross-cluster fairness (distribution-clustered FL)")
    analysis.plot_detection(dfd, f"{results_dir}/dist_detection.png")
    return runs


if __name__ == "__main__":
    import sys

    # `python run_data_cfl.py demo` runs the three-condition comparison;
    # bare `python run_data_cfl.py` runs a single federation as before.
    if len(sys.argv) > 1 and sys.argv[1] == "demo":
        demo()
    else:
        main()
