"""
ATTACK: architecture spoofing in Clustered Federated Learning.

Two malicious clients whose true home is the MLP cluster DECLARE the DNN family,
so the server places them inside the high-capacity DNN cluster they do not belong
to. Once inside, they deliver a poisoning payload (default: sign-flip) against the
DNN cluster's global model. This is the two-stage attack the report describes:
spoof to get in, poison from within.

Run:  python attack_spoofing.py

Outputs a baseline-vs-attack accuracy table, the aim-#4 impact metrics
(victim-cluster degradation, attack success rate, cross-cluster fairness), and
saves attack_result.png.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import fl_common as fl

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
SEED = 42
N_CLIENTS = 10
ALPHA = 0.5                     # Dirichlet non-IID skew
ROUNDS = 5
EPOCHS = 1
# clients 0-4 truly belong to the MLP cluster, 5-9 to the DNN cluster
HOME_ASSIGNMENT = ["mlp"] * 5 + ["dnn"] * 5
SPOOFERS = [0, 1]              # MLP-home clients that will declare DNN
VICTIM_FAMILY = "dnn"
PAYLOAD = "sign_flip"          # 'sign_flip' | 'boost' | 'label_flip'
DEGRADE_MARGIN = 0.05          # accuracy drop that counts as a corrupted round


def main():
    fl.set_global_seeds(SEED)
    client_data, test, probe = fl.load_federated_data(
        n_clients=N_CLIENTS, alpha=ALPHA, seed=SEED)

    # ----- Run A: baseline, everyone honest --------------------------------- #
    print("########## BASELINE (no attack) ##########")
    honest = fl.build_honest_clients(client_data, HOME_ASSIGNMENT)
    base = fl.simulate(honest, ROUNDS, EPOCHS, test, probe)

    # ----- Run B: spoofing attack ------------------------------------------- #
    print("\n########## ATTACK (spoof + poison) ##########")
    spoofed = fl.build_spoofed_clients(
        client_data, HOME_ASSIGNMENT, SPOOFERS, VICTIM_FAMILY, payload=PAYLOAD)
    atk = fl.simulate(spoofed, ROUNDS, EPOCHS, test, probe)

    # ----- Impact metrics (report aim #4) ----------------------------------- #
    print("\n########## IMPACT ##########")
    print(f"{'round':<6}{'DNN base':<12}{'DNN attack':<12}{'MLP base':<12}{'MLP attack':<12}")
    corrupted = 0
    for r in range(ROUNDS):
        db, da = base.history["dnn"][r], atk.history["dnn"][r]
        mb, ma = base.history["mlp"][r], atk.history["mlp"][r]
        if da < db - DEGRADE_MARGIN:
            corrupted += 1
        print(f"{r + 1:<6}{db:<12.4f}{da:<12.4f}{mb:<12.4f}{ma:<12.4f}")

    victim_drop = base.history["dnn"][-1] - atk.history["dnn"][-1]
    base_final = [base.history[f][-1] for f in fl.FAMILIES]
    atk_final = [atk.history[f][-1] for f in fl.FAMILIES]

    print(f"\nVictim (DNN) cluster degradation, final round : {victim_drop:+.4f}")
    print(f"Attack success rate (rounds corrupted)        : {corrupted}/{ROUNDS}")
    print(f"Cross-cluster accuracy variance  baseline={np.var(base_final):.5f} "
          f"attack={np.var(atk_final):.5f}   (higher = less fair, q-FFL framing)")
    print(f"Worst-group accuracy             baseline={min(base_final):.4f} "
          f"attack={min(atk_final):.4f}")

    # ----- Plot ------------------------------------------------------------- #
    rounds_axis = range(1, ROUNDS + 1)
    plt.figure(figsize=(7, 4.5))
    plt.plot(rounds_axis, base.history["dnn"], "o-", label="DNN cluster (baseline)")
    plt.plot(rounds_axis, atk.history["dnn"], "x--", label="DNN cluster (under attack)")
    plt.plot(rounds_axis, base.history["mlp"], "o-", color="gray", alpha=0.6,
             label="MLP cluster (baseline)")
    plt.xlabel("Communication round")
    plt.ylabel("Global test accuracy")
    plt.title("Architecture-spoofing attack on the DNN cluster")
    plt.legend()
    plt.tight_layout()
    plt.savefig("attack_result.png", dpi=130)
    print("\nSaved plot -> attack_result.png")


if __name__ == "__main__":
    main()
