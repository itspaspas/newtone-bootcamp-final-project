#!/usr/bin/env python3
"""
SAC Training Runner with Control Flags
=====================================

This script allows you to control the SAC training process with flags.
"""

import argparse
from train_sac_comprehensive import train_sac_comprehensive

def main():
    """Main function to run SAC training with configurable parameters."""
    parser = argparse.ArgumentParser(description="Run SAC training with configurable parameters.")
    parser.add_argument('--episodes', type=int, default=2000, help='Total number of training episodes (default: 2000)')
    parser.add_argument('--train-interval', type=int, default=1000, help='Number of episodes per training phase before evaluation (default: 1000)')
    parser.add_argument('--eval-episodes', type=int, default=100, help='Number of evaluation episodes per evaluation phase (default: 100)')
    parser.add_argument('--max-steps', type=int, default=60, help='Maximum steps per episode (default: 60)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed (default: 42)')
    parser.add_argument('--quick-test', dest='enable_quick_test', action='store_true', help='Enable quick test after training')
    parser.add_argument('--no-quick-test', dest='enable_quick_test', action='store_false', help='Disable quick test after training')
    parser.set_defaults(enable_quick_test=True)
    parser.add_argument('--quick-test-episodes', type=int, default=200, help='Number of episodes for quick test (default: 200)')
    parser.add_argument('--no-train', dest='do_train', action='store_false', help='Skip training and only run quick test')
    parser.add_argument('--train', dest='do_train', action='store_true', help='Run training (default)')
    parser.set_defaults(do_train=True)
    args = parser.parse_args()

    print("=== SAC Training with Control Flags ===")
    print("Configure your training parameters below:")
    print()
    print(f"Training Configuration:")
    print(f"  Episodes: {args.episodes}")
    print(f"  Train interval: {args.train_interval}")
    print(f"  Eval episodes per phase: {args.eval_episodes}")
    print(f"  Max steps per episode: {args.max_steps}")
    print(f"  Seed: {args.seed}")
    print()
    print(f"Quick Test Configuration:")
    print(f"  Enable quick test: {args.enable_quick_test}")
    if args.enable_quick_test:
        print(f"  Quick test episodes: {args.quick_test_episodes}")
    print()

    # Confirm before starting
    response = input("Start training? (y/n): ").lower().strip()
    if response != 'y':
        print("Training cancelled.")
        return

    # Run training
    try:
        if args.do_train:
            agent, training_scores, eval_scores, eval_checkpoints = train_sac_comprehensive(
                episodes=args.episodes,
                train_interval=args.train_interval,
                eval_episodes=args.eval_episodes,
                max_steps=args.max_steps,
                seed=args.seed,
                enable_quick_test=args.enable_quick_test,
                quick_test_episodes=args.quick_test_episodes
            )
            print("\n=== TRAINING COMPLETED SUCCESSFULLY ===")
            print("Files generated:")
            print("  - trained_sac_comprehensive_weights.pth")
            print("  - training_results.json")
            print("  - sac_comprehensive_results.jpg")
            if args.enable_quick_test:
                print("  - sac_test_results.jpg")
        else:
            print("Training skipped. Only quick test will run (if enabled).")
            # Optionally, you could load the agent and run quick test here if needed.
    except Exception as e:
        print(f"\nERROR: Training failed with error: {e}")
        print("Please check the error message and try again.")

if __name__ == "__main__":
    main() 