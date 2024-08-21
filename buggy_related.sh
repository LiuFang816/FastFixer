#!/bin/bash
cd "$(dirname "$0")"

BASE_DIR='/home/FastFixer/FastFixer/'
export PYENV_ROOT="$HOME/.pyenv"
command -v pyenv >/dev/null || export PATH="$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init -)"
pyenv activate FastFixer
TEST=true
LABEL=true
TEST_FIXED=false
PRE_EXECUTE_NEEDED=true

command="PYTHONPATH=$BASE_DIR python evaluate/evaluate_buggy.py"

if [ "$TEST" = true ]; then
    command="$command --test"
fi

if [ "$PRE_EXECUTE_NEEDED" = true ]; then
    command="$command --pre-execute-needed"
fi

if [ "$LABEL" = true ]; then
    command="$command --label"
fi

if [ "$TEST_FIXED" = true ]; then
    command="$command --test-fixed"
fi

echo $command

eval $command

if [ "$LABEL" = true ]; then
    command="PYTHONPATH=$BASE_DIR python statistic/statistic_buggy_type_labelling.py"
    if [ "$TEST" = true ]; then
        command="$command --fix"
    fi
    eval $command
fi

if [ "$LABEL" = false ] && [ "$TEST" = true ] && [ "$TEST_FIXED" = false ]; then
  echo "Test buggy pass num"
  PYTHONPATH="$BASE_DIR" python statistic/statistic_test_buggy_pass_num.py
fi