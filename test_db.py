from simpledb import SimpleDB

# columns = {'filename': 'text', "acc": 'text'}
# values = {"acc": '0.9123139'}
#
#
# db = SimpleDB(columns=columns, audit_db_name="testdb")
# db.init_db()
# db.start_file_process(filename="first", items=values)
# db.finalize_file_process(filename="first")
# print(db.get_data())


columns = {'metric': 'text', 'value': 'text', 'classifier': 'text', 'n_features': 'text',
           'elapsed_time': 'text', 'it': 'text', 'dataset': 'text', 'filename': 'text'}
db = SimpleDB(columns=columns, audit_db_name="testdb")
db.init_db()
df = db.get_data()
