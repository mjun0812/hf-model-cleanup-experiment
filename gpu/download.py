from utils import MODELS, get_hf_models, get_st_models


def main():
    for model_name in MODELS["hf"]:
        get_hf_models(model_name)

    for model_name in MODELS["st"]:
        get_st_models(model_name)


if __name__ == "__main__":
    main()
