import datetime
from pilotscope.DBInteractor.PilotDataInteractor import PilotDataInteractor
from pilotscope.PilotConfig import PostgreSQLConfig


def get_test_set_e2e_time(test_set_filename, test_set_db, optimal_pg_or_model, db_host="localhost", db_user="postgres", db_user_pwd="postgres", db_port=5456):
    """
    :param test_set_filename: the file path of test set (sql, true_card, model_est_card)
    :param test_set_db: the database of test set
    :param optimal_pg_or_model: test optimal's, pg's or model's end-to-end time
    """
    assert optimal_pg_or_model in ["optimal", "pg", "model"], "optimal_pg_or_model must be optimal or pg or model"

    config = PostgreSQLConfig()
    config.db = test_set_db
    config.db_host = db_host
    config.db_user = db_user
    config.db_user_pwd = db_user_pwd
    config.db_port = db_port
    # NOTE: some query's execution time is too long, so we need to set the timeout
    config.sql_execution_timeout = 300000

    data_interactor = PilotDataInteractor(config)

    with open(test_set_filename, 'r') as f:
        lines = f.readlines()
    
    all_e2e_times, total_cnt = [], 0
    for i, line in enumerate(lines):
        sql = line.split("||")[0]

        # NOTE: test origin query's execution time and skip all subqueries
        if sql.startswith("select"):
            total_cnt += 1
            data_interactor.pull_subquery_card()
            result = data_interactor.execute(sql)    
            
            # NOTE: get true_card and model_est_card
            true_cards, optimal_pg_or_model_est_cards = {}, {}
            cnt = 0
            for k in result.subquery_2_card.keys():
                subquery = lines[i + cnt + 1].split("||")[0]
                assert not subquery.startswith("select"), "subquery must not start with select"
                true_card = lines[i + cnt + 1].split("||")[1]
                model_est_card = lines[i + cnt + 1].split("||")[2]

                true_cards[k] = int(true_card)
                if optimal_pg_or_model != "optimal":
                    if float(model_est_card) >= 0:
                        if optimal_pg_or_model == "pg":
                            optimal_pg_or_model_est_cards[k] = result.subquery_2_card[k]
                        elif optimal_pg_or_model == "model":
                            optimal_pg_or_model_est_cards[k] = float(model_est_card)
                    # NOTE: single-table subquery has no model_est_card
                    # to reduce the uncertainty associated with the estimates
                    # we use the true cardinality for single-table subqueries
                    else:
                        optimal_pg_or_model_est_cards[k] = true_cards[k]
                else:
                    optimal_pg_or_model_est_cards[k] = true_cards[k]
                cnt += 1

            data_interactor.push_card(optimal_pg_or_model_est_cards)
            data_interactor.pull_execution_time()
            e2e_time = data_interactor.execute(sql).execution_time
            all_e2e_times.append(round(e2e_time, 4))
            print(f"total_time: {round(sum(all_e2e_times), 4)}, {total_cnt}sql: {sql}, e2e_time: {e2e_time}")

    print('-' * 30)
    return all_e2e_times


if __name__ == '__main__':
    test_set_filename = f""
    test_set_db = f""
    print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    print(f"test_set_db: {test_set_db}, test_set_filename: {test_set_filename}")

    all_e2e_times = get_test_set_e2e_time(test_set_filename, test_set_db, optimal_pg_or_model="pg")
    print(f"total: {sum(all_e2e_times)}, pg's e2e time: {all_e2e_times}")

    all_e2e_times = get_test_set_e2e_time(test_set_filename, test_set_db, optimal_pg_or_model="optimal")
    print(f"total: {sum(all_e2e_times)}, optimal's e2e time: {all_e2e_times}")
    
    all_e2e_times = get_test_set_e2e_time(test_set_filename, test_set_db, optimal_pg_or_model="model")
    print(f"total: {sum(all_e2e_times)}, model's e2e time: {all_e2e_times}")

    print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    print("=" * 30)
