import datetime
import numpy as np
from pilotscope.DBInteractor.PilotDataInteractor import PilotDataInteractor
from pilotscope.PilotConfig import PostgreSQLConfig
from pilotscope.Common.CardMetricCalc import p_error_calc


def calc_percentile_list(l, percentile_list):
    p_list = []
    for percentile in percentile_list:
        assert percentile >= 0 and percentile <= 100
        per_val = np.percentile(l, percentile)
        p_list.append(round(per_val, 4))
    return p_list


def calc_test_set_p_error(test_set_filename, test_set_db, pg_or_model, db_host="localhost", db_user="postgres", db_user_pwd="postgres", db_port=5456):
    """
    :param test_set_filename: the file path of test set (sql, true_card, model_est_card)
    :param test_set_db: the database of test set
    :param pg_or_model: test pg's or model's p-error
    """
    assert pg_or_model in ["pg", "model"], "pg_or_model must be pg or model"

    config = PostgreSQLConfig()
    config.db = test_set_db
    config.db_host = db_host
    config.db_user = db_user
    config.db_user_pwd = db_user_pwd
    config.db_port = db_port

    data_interactor = PilotDataInteractor(config)

    with open(test_set_filename, 'r') as f:
        lines = f.readlines()
    
    all_p_errors = []
    for i, line in enumerate(lines):
        sql = line.split("||")[0]

        # NOTE: calculate origin query's p-error
        if sql.startswith("select"):
            data_interactor.pull_subquery_card()
            result = data_interactor.execute(sql)    
            
            # NOTE: get true_card and model_est_card
            true_cards, pg_or_model_est_cards = {}, {}
            cnt = 0
            for k in result.subquery_2_card.keys():
                subquery = lines[i + cnt + 1].split("||")[0]
                assert not subquery.startswith("select"), "subquery must not start with select"
                true_card = lines[i + cnt + 1].split("||")[1]
                model_est_card = lines[i + cnt + 1].split("||")[2]

                true_cards[k] = int(true_card)
                if float(model_est_card) >= 0:
                    if pg_or_model == "pg":
                        pg_or_model_est_cards[k] = result.subquery_2_card[k]
                    elif pg_or_model == "model":
                        pg_or_model_est_cards[k] = float(model_est_card)
                # NOTE: single-table subquery has no model_est_card
                # to reduce the uncertainty associated with the estimates
                # we use the true cardinality for single-table subqueries
                else:
                    pg_or_model_est_cards[k] = true_cards[k]
                cnt += 1
            
            p_error = p_error_calc(config, sql, pg_or_model_est_cards, true_cards)
            all_p_errors.append(round(p_error, 4))

    return all_p_errors


if __name__ == '__main__':
    test_set_filename = f""
    test_set_db = f""
    print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    print(f"test_set_db: {test_set_db}, test_set_filename: {test_set_filename}")

    all_p_errors = calc_test_set_p_error(test_set_filename, test_set_db, pg_or_model="pg")
    print(f"pg's p-error: {all_p_errors}")
    print(f"pg's p-error 30% 50% 80% 90% 95% 99%: {calc_percentile_list(all_p_errors, [30, 50, 80, 90, 95, 99])}")
    print("="*30)
    
    all_p_errors = calc_test_set_p_error(test_set_filename, test_set_db, pg_or_model="model")
    print(f"model's p-error: {all_p_errors}")
    print(f"model's p-error 30% 50% 80% 90% 95% 99%: {calc_percentile_list(all_p_errors, [30, 50, 80, 90, 95, 99])}")
    print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

    print("="*30)
