"""Architecture-based clustered FL (+ optional metadata / weight attacks / defence)."""

from attacks import ArchSpoof, CompositeAttack, NoAttack, WeightSignFlip
from clustering import ArchitectureClusterer
from data import DEFAULT_CSV, prepare_fl_data
from defense import FingerprintVerifier
from fl_loop import FederatedServer
from seeding import set_global_seeds


def main(
    rounds=2,
    epochs=5,
    num_clients=6,
    csv_path=DEFAULT_CSV,
    malicious_clients=(0,),
    enable_arch_spoof=False,
    enable_sign_flip=False,
    spoof_as="dnn",
    spoof_adaptive=True,        # False = naive spoof (trains true arch; gets shape-rejected)
    partition="dirichlet",      # report setting is non-IID; use "iid" for the simple case
    dirichlet_alpha=0.5,
    enable_defense=False,
    seed=42,                    # vary across repeat runs to get error bars
    weighted_aggregation=True,  # False = unweighted mean (aggregation ablation)
):
    # Pin every RNG before the first model is built, so two conditions that
    # differ only in `enable_arch_spoof` really do differ only in that.
    set_global_seeds(seed)

    fl_data = prepare_fl_data(
        csv_path=csv_path, num_clients=num_clients,
        partition=partition, dirichlet_alpha=dirichlet_alpha,
        random_state=seed,
    )

    # Half mlp / half dnn (matches original CFL.py assignment).
    mid = num_clients // 2
    client_archs = ["mlp"] * mid + ["dnn"] * (num_clients - mid)

    attacks = []
    if enable_arch_spoof:
        attacks.append(ArchSpoof(malicious_clients=malicious_clients, spoof_as=spoof_as,
                                  adaptive=spoof_adaptive))
    if enable_sign_flip:
        attacks.append(WeightSignFlip(malicious_clients=malicious_clients))
    attack = CompositeAttack(*attacks) if attacks else NoAttack()

    server = FederatedServer(
        input_dim=fl_data.input_dim,
        num_classes=fl_data.num_classes,
        clusterer=ArchitectureClusterer(),
        client_archs=client_archs,
        attack=attack,
        mode="architecture",
        defense=FingerprintVerifier(seed=seed) if enable_defense else None,
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


if __name__ == "__main__":
    main()
