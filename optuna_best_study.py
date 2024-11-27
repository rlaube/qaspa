# Usage: python optuna_best_study.py path_to_study_name
import sys
import optuna

def main():
    # print(transformers.__version__)
    print('Study name', sys.argv[1])
    study = optuna.load_study(study_name=sys.argv[1], storage="sqlite:///{}.db".format(sys.argv[1]))
    print("Best trial until now:")
    print(" Value: ", study.best_trial.value)
    print(" Params: ")
    for key, value in study.best_trial.params.items():
        print(f"    {key}: {value}")

if __name__ == "__main__":
   main()        