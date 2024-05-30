# Statistics Generation

We provide files for generating statistics in this directory. In the process of generating statistics, PostgresSQL is not required. The statistics can be generated with the scripts below:

```shell
sh gen_hisogram.sh
sh gen_summary.sh
sh gen_fanout.sh
sh gen_size.sh
```

If you want to use a different bin size or use your own datasets, you can modify the settings in the scripts.

## Important Notes
- Generating these statistics may take some time, so please be patient.
- Ensure that `sh gen_fanout.sh` is run after all histograms are generated.
- Logs for the generation process can be found in `utils/statistics/logs`.
- The code to generate `abbrev_col_type.pkl` files is not provided. You will need to generate them yourself for new datasets.

