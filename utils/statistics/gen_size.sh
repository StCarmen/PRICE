#!/bin/bash
BASEDIR=$(cd $(dirname "$0") && pwd -P)

mkdir -p $BASEDIR/logs/pretrain/size
mkdir -p $BASEDIR/logs/test/size
mkdir -p $BASEDIR/logs/finetune/size

# dataset to pretrain
for db in accidents airline baseball basketball carcinogenesis ccs chembl consumer credit employee financial fnhk grants hepatitis hockey legalacts movielens sakila sap seznam ssb talkingdata telstra tournament tpc_h tubepricing
do
    nohup python -u $BASEDIR/gen_size.py --db $db --usage pretrain > $BASEDIR/logs/pretrain/size/${db}.log 2>&1 &
done

# unseen dataset to test
for db in imdb stats ergastf1 genome
do
    nohup python -u $BASEDIR/gen_size.py --db $db --usage test > $BASEDIR/logs/test/size/${db}.log 2>&1 &
done

# finetune
for db in imdb stats ergastf1 genome
do
    nohup python -u $BASEDIR/gen_size.py --db $db --usage finetune > $BASEDIR/logs/finetune/size/${db}.log 2>&1 &
done
