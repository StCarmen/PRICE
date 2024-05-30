#!/bin/bash
BASEDIR=$(cd $(dirname "$0") && pwd -P)
bin_size=40

mkdir -p $BASEDIR/features/pretrain
mkdir -p $BASEDIR/features/test
mkdir -p $BASEDIR/features/finetune

mkdir -p $BASEDIR/features_log/pretrain
mkdir -p $BASEDIR/features_log/test
mkdir -p $BASEDIR/features_log/finetune

# dataset to pretrain
for db in accidents airline baseball basketball carcinogenesis ccs chembl consumer credit employee financial fnhk grants hepatitis hockey legalacts movielens sakila sap seznam ssb talkingdata telstra tournament tpc_h tubepricing
do
    nohup python -u $BASEDIR/features_generate.py --db $db --bin_size $bin_size --usage pretrain > $BASEDIR/features_log/pretrain/${db}_${bin_size}.log 2>&1 &
done

# unseen dataset to test
for db in imdb stats ergastf1 genome
do
    nohup python -u $BASEDIR/features_generate.py --db $db --bin_size $bin_size --usage test > $BASEDIR/features_log/test/${db}_${bin_size}.log 2>&1 &
done

# finetune
for db in imdb stats ergastf1 genome
do
    nohup python -u $BASEDIR/features_generate.py --db $db --bin_size $bin_size --usage finetune > $BASEDIR/features_log/finetune/${db}_${bin_size}.log 2>&1 &
done
