#!/bin/bash
BASEDIR=$(cd $(dirname "$0") && pwd -P)
bin_size=40

mkdir -p $BASEDIR/logs/pretrain/histogram
mkdir -p $BASEDIR/logs/test/histogram
mkdir -p $BASEDIR/logs/finetune/histogram

# dataset to pretrain
for db in accidents airline baseball basketball carcinogenesis ccs chembl consumer credit employee financial fnhk grants hepatitis hockey legalacts movielens sakila sap seznam ssb talkingdata telstra tournament tpc_h tubepricing
do
    nohup python -u $BASEDIR/gen_histogram.py --db $db --bs $bin_size --usage pretrain > $BASEDIR/logs/pretrain/histogram/${db}_${bin_size}.log 2>&1 &
done

# unseen dataset to test
for db in imdb stats ergastf1 genome
do
    nohup python -u $BASEDIR/gen_histogram.py --db $db --bs $bin_size --usage test > $BASEDIR/logs/test/histogram/${db}_${bin_size}.log 2>&1 &
done

# finetune
for db in imdb stats ergastf1 genome
do
    nohup python -u $BASEDIR/gen_histogram.py --db $db --bs $bin_size --usage finetune > $BASEDIR/logs/finetune/histogram/${db}_${bin_size}.log 2>&1 &
done
