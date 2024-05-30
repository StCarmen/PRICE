import sqlglot
from pilotscope.DBInteractor.PilotDataInteractor import PilotDataInteractor
from pilotscope.PilotConfig import PostgreSQLConfig


if __name__ == '__main__':
    dataset_name = ''
    workload_in_path = f''
    workload_out_path = f''
    print(f"dataset: {dataset_name}, workload_in_path: {workload_in_path}, workload_out_path: {workload_out_path}")

    config = PostgreSQLConfig()
    config.db_host = "localhost"
    config.db_user = "postgres"
    config.db_user_pwd = "postgres"
    config.db_port = 5456
    config.sql_execution_timeout = 3600
    config.db = dataset_name
    data_interactor = PilotDataInteractor(config)

    with open(workload_in_path, 'r') as read_files:
        sqls = []
        for line in read_files:
            sql = line.strip()
            sqls.append(sql)

    subsqls, count = [], 0
    for sql in sqls:
        data_interactor.pull_subquery_card()
        result = data_interactor.execute(sql)
        subquerys = list(result.subquery_2_card.keys())
        subsqls.append(sql)
        count += 1
        if count % 100 == 0:
            print(f"{count} sqls has been processed")
        for subquery in subquerys:
            try:
                tables = [table for table in sqlglot.parse_one(subquery).find_all(sqlglot.exp.Table)]
            except Exception as e:
                print(f"sql: {sql}")
                print(f"subquery: {subquery}")
                break
            if len(tables) == 1: # skip single table subquery
                continue
            subsqls.append(subquery)

    with open(workload_out_path, "w") as f:
        for sql in subsqls:
            f.write(sql + "\n")

    print(f"dataset: {dataset_name}")
