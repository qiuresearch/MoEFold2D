#!/usr/bin/env bash

set -e -o posix

declare -a loco_types=(
    "16S-rRNA"
    "23S-rRNA"
    "5S-rRNA"
    "gpI-intron"
    "RNaseP"
    "SRP"
    "TERC"
    "tmRNA"
    "tRNA"
    "all"
)

loco_models=("${loco_types[@]/#/metafam3.metafam2d.l4c64.validobj.}")
loco_models_upsample=("${loco_models[@]/%/.upsample}")

default_model=${loco_models_upsample[-1]}
# echo " ${loco_models[@]} "
# echo " ${loco_models_upsample[@]} "

[[ ${#@} -lt 1 ]] && {
    echo -ne "
Usage: ${0##*/} action data_files [cmOptions]

Arguments:
    action          : one of train, predict, or brew_dbn/brew_ct/brew_bpseq
    data_files      : pkl file(s) for train, fasta file(s) for predict, folder for brew_dbn/ct/bpseq
    -model       [] : choose a single model from the following:
"
    printf "                      %-72s\n" "${loco_models_upsample[@]}"
    # -devset      [] : one of stral-ND, stral-NR80, bprna-TR0VL0 (case insensitive)
    # -params      [] : one of 400K, 960K, 1.4M, 3.5M (case insensitive)
echo -e "
    -cmdArgs        : all other options are passed to fly_paddle.py as-is

    SPACE in folder/file names will very likely BREAK the code!!!
"
    exit 0
}

start_time=`date +%s`
BASH_HIST="$(pwd)/.bash_history"
echo "=======>>> $(date +'%Y-%m-%d %H:%M:%S') RUN-$$> ${0##*/} ${@}" >> $BASH_HIST

# get the optional and positional parameters
num_psargs=0
while read -r var ; do vars_before+=("$var") ; done <<< $(set)
while [ $# -gt 0 ]; do
    case "$1" in
        -h|--help) $0 ; exit 0 ;;
        *-dryrun)
            declare ${1##*-}=TRUE ; shift
            ;;
        *-model|*-devset|*-params)
            if [ -n "$2" ] && [ "${2:0:1}" != "-" ]; then
                declare ${1##*-}="$2"
                shift 2
            else
                echo "Error: Argument for $1 is missing" >&2
                exit 1
            fi
            ;;
        -*) # unrecognized options will be saved to cmdArgs
            if [ -n "$2" ] && [ "${2:0:1}" != "-" ]; then
                [[ -z "${cmdArgs}" ]] && cmdArgs="$1 $2" || cmdArgs="${cmdArgs} $1 $2"
                shift 2
            else
                [[ -z "${cmdArgs}" ]] && cmdArgs="$1" || cmdArgs="${cmdArgs} $1"
                shift
            fi
            ;;
        *) # positional arguments
            num_psargs=$((num_psargs + 1))
            case ${num_psargs} in
                1) action="$1" ;;
                *) [[ -z "${data_files}" ]] && data_files="$1" || data_files="${data_files} $1" ;;
            esac
            shift
            ;;
    esac
done

case "${action^^}" in
    TRAIN|BREW*|PREDICT)
        [[ -z "${data_files}" ]] && {
            echo "ERROR:: data files must be provided!!!"
            exit 1
            }
        ;;
    *)
        echo "ERROR: unsupported action: ${action}"
        exit 1
        ;;
esac

pkg_dir="$(dirname $0)"
src_dir="$(dirname $0)/src"

# show changed variables
while read -r var ; do vars_after+=("$var") ; done <<< $(set)
echo "Parameters:"
for ((i = 0; i < ${#vars_after[@]}; i++)) ; do
    var="${vars_after[$i]}"
    [[ ${var:0:1} == '[' ]] && continue # skip command history
    [[ "vars_before vars_after BASH_REMATCH PIPESTATUS" =~ "${var%%=*}" ]] && continue # skip myself
    [[ " ${vars_before[@]} " =~ " $var " ]] && continue # skip unchanged
    echo "    $var"
    echo "$var" >> $BASH_HIST
done

# check if ${src_dir} in $PYTHONPATH
[[ ${PYTHONPATH} == *"${src_dir}"* ]] || export PYTHONPATH=${src_dir}:${PYTHONPATH}

case "${action^^}" in
    TRAIN)
        python3 ${src_dir}/fly_paddle.py train -load_dir ${load_dir} -save_dir ./ -data_dir ./ \
            -data_name ${data_files} ${cmdArgs}
    ;;
    # EVALUATE|EVAL)
    #     python3 ${src_dir}/fly_paddle.py evaluate -load_dir ${load_dir} -save_dir ./ \
    #         ${data_files} ${cmdArgs}
    # ;;
    PREDICT)
        i=-1
        predict_pkls=()
        for loco_model in ${loco_models[@]}; do
            i=$((i + 1))
            echo "Predicting with ${loco_model} ..."
            model_dir="${pkg_dir}/models/${loco_model}"

            python3 ${src_dir}/fly_paddle.py predict -load_dir ${model_dir} -save_dir ./loco_outputs/${loco_types[$i]} \
                    ${data_files} ${cmdArgs} -save_subdir "" -save_lumpsum -save_individual -named_after id -verbose 0
                
            predict_pkls+=("./loco_outputs/${loco_types[$i]}/predict_all.pkl")
        done

        python3 ${src_dir}/eval_mitas.py ctmat_moefold_predict  -loco_home ./loco_outputs -loco_prefix "" \
            -loco_types "${loco_types[@]}" -loco_suffix "" -loco_output predict_all.pkl \
            -phys_exes "${pkg_dir}/bin/linearfold" -save_dir moe_outputs -save_lumpsum -save_individual
    ;;
    *)
        echo "ERROR: unsupported action: ${action}!"
        exit 1
    ;;
esac

end_time=`date +%s`
run_time=$((end_time - start_time))
echo "Total running time: ${run_time} seconds (`date -d@${run_time} -u +%Hh:%Mm:%Ss`)"
echo "(^_^) @ $(date)"
echo "$(date +'%Y-%m-%d %H:%M:%S') END-$$> ${0##*/} ${@}" >> $BASH_HIST
exit 0
