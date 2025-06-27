def main():
    import yaml
    from training_runner import TrainingRunner

    with open("ga_config.yaml", "r") as f:
        config = yaml.safe_load(f)
        trainer = TrainingRunner(config)
        trainer.prepare_environment()
        trainer.run_training(generations=config.get("generations", 30))
        trainer.save_results()

if __name__ == "__main__":
    main()
