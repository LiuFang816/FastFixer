import argparse
from config import *
from fix.codellama_instruct_fix import codellama_instuct_fix
from evaluate.evaluate_buggy import evalute_only, get_global_value
from statistic.statistic_each_problem_fix_rate import statistic_each_problem_fix_rate
from fix.llama_decoding.utils import augment_generate

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--model', type=str, default='codellama-instrcut-7b-hf', required=True)
    parser.add_argument('--msg-passed', type=str, default='', required=False)
    parser.add_argument('--finetune-method', type=str, default='', required=False)
    parser.add_argument('--augment', type=bool, default=True, required=True, action=argparse.BooleanOptionalAction)
    

    args = parser.parse_args()

    model = args.model
    msg_passed = args.msg_passed
    finetune_method = args.finetune_method
    augment = args.augment

    print(augment)

    if msg_passed:
        FINETUNE_SHARED.MODEL = f"{model}-with-{msg_passed}"
    else:
        FINETUNE_SHARED.MODEL = model
    if finetune_method:
        FINETUNE_SHARED.IS_FINETUNED = True
        FINETUNE_SHARED.FINETUNE_METHOD = finetune_method
        FINETUNE_SHARED.FIXED_DIR = (FINETUNE_SHARED.MODEL + FINETUNE_SHARED.FINETUNE_METHOD) if FINETUNE_SHARED.IS_FINETUNED else FINETUNE_SHARED.MODEL
        FINETUNE_SHARED.FINETUNE_MODEL_PATH = os.path.join(BASE_DIR, "checkpoints", FINETUNE_SHARED.FIXED_DIR)
    else:
        FINETUNE_SHARED.IS_FINETUNED = False
        FINETUNE_SHARED.FINETUNE_METHOD = "origin-model"
        FINETUNE_SHARED.FIXED_DIR = FINETUNE_SHARED.MODEL

    # if not augment:
    #     import lade
    #     lade.augment_all()
    #     lade.config_lade(LEVEL=5, WINDOW_SIZE=7, GUESS_SET_SIZE=7, DEBUG=0) 

    print(f"FINETUNE_SHARED.MODEL: {FINETUNE_SHARED.MODEL}")
    if finetune_method:
        print(f"FINETUNE_SHARED.FINETUNE_METHOD: {FINETUNE_SHARED.FINETUNE_METHOD}")

    # if model.startswith('codellama-in'):
    #     if augment:
    #         augment_generate()
    #     # codellama_instuct_fix(augment)

    get_global_value(test=True, pre_execute_needed=True, label=False, test_fixed=True)
    evalute_only()
    statistic_each_problem_fix_rate()


if __name__ == "__main__":
    main()

    
