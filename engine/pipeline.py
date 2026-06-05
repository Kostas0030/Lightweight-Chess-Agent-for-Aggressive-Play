import os
import torch
from self_play_game_parallel_aggressive import SelfPlayRunnerBatch  
from self_play_train import SelfPlayTrainer   
from collections import namedtuple

GameData = namedtuple("GameData", ["position", "pi", "z"])

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ================== CONFIG ==================
    START_VERSION = 4
    NUM_ITERATIONS = 10                  
    SAMPLES_PER_ITERATION = 1_000_000
    NUM_PROCESSES = 16
    NUM_SIMULATIONS = 100
    BATCH_SIZE_SELFPLAY = 128

    TRAIN_EPOCHS = 20
    TRAIN_BATCH_SIZE = 256
    LR = 1e-4

    DATA_DIR = "data/self_play_data/aggressive/lambda_0.5/"
    os.makedirs(DATA_DIR, exist_ok=True)
    # ============================================

    current_policy = f"model/POLICY_MODEL_10EPOCHS_V{START_VERSION}.pth"
    current_value  = f"model/VALUE_MODEL_10EPOCHS_V{START_VERSION}.pth"

    for it in range(NUM_ITERATIONS):
        version = it + START_VERSION
        print(f"\n{'='*70}")
        print(f"ITERATION {it+1}/{NUM_ITERATIONS} -> Generating V{version}")
        print(f"{'='*70}")

        # === 1. Create new data file for this version ===
        data_path = f"data/self_play_data/aggressive/lambda_0.5/V{version}_100_1m.pkl"

        runner = SelfPlayRunnerBatch(
            policy_path=current_policy,
            value_path=current_value,
            save_path=data_path,                   
            num_simulations=NUM_SIMULATIONS,
            batch_size=BATCH_SIZE_SELFPLAY,
        )

        print(f"Generating new self-play data -> {data_path}")
        runner.run(num_samples=SAMPLES_PER_ITERATION, num_processes=NUM_PROCESSES)

        # === 2. Train new models on this fresh data ===
        trainer = SelfPlayTrainer(
            policy_model_path=current_policy,
            value_model_path=current_value,
            dataset_path=data_path,
            device=device,
            batch_size=TRAIN_BATCH_SIZE,
            lr=LR,
        )

        new_version_str = f"V{version + 1}"

        print(f"Training policy model -> {new_version_str}")
        trainer.train_policy(
            epochs=TRAIN_EPOCHS,
            save_epochs=[3, 10, 20],
            version=new_version_str
        )

        print(f"Training value model -> {new_version_str}")
        trainer.train_value(
            epochs=TRAIN_EPOCHS,
            save_epochs=[3, 10, 20],
            version=new_version_str
        )

        # === 3. Update paths for next iteration ===
        current_policy = f"model/POLICY_MODEL_10EPOCHS_{new_version_str}.pth"
        current_value  = f"model/VALUE_MODEL_10EPOCHS_{new_version_str}.pth"

        print(f"-> New models ready: {new_version_str}")

    print("\n=== TRAINING LOOP FINISHED SUCCESSFULLY ===")

if __name__ == "__main__":
    main()