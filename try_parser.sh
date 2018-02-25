
# we didn't use validation data set. and we leave the volume of test set ~ 15 so that 
# the run_parser can terminate quickly.
python2.7 parser.py --test=False \
    --train_p=0.99 --valid_p=0.008 --test_p=0.002 \
    --output_dir=./test_output_trial.txt \
    --test_dir=./test_trial.txt \
    --remove_hyphen=True
