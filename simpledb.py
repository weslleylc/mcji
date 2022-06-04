import sqlite3
from datetime import datetime

import pandas as pd


class SimpleDB:
    # the constructor
    ''''''

    def __init__(self, columns, audit_db_name, table_name="data"):
        self.columns = columns
        self.audit_db_name = audit_db_name
        self.table_name = table_name
        self.con = None

    # Create a sqlite cursor
    def get_sqlite_connection(self):
        self.con = sqlite3.connect(self.audit_db_name)

    # runs ddl to initialize the database with any tables that are needed
    def init_db(self):
        self.get_sqlite_connection()
        cur = self.con.cursor()
        columns = self.columns
        columns['startdatetime'] = "text"
        columns['enddatetime'] = "text"
        columns['status'] = "text"

        table = "{}({})".format(self.table_name,
                                ", ".join(["{} {}".format(key, value) for key, value in columns.items()]))

        cur.execute('''CREATE TABLE IF NOT EXISTS %s''' % table)
        self.con.commit()
        self.con.close()

    # returns a 1 or 0 rows containing the details for the import process for the specified filename
    def get_file_process_status(self, filename):
        self.get_sqlite_connection()
        cur = self.con.cursor()

        cur.execute('''SELECT * FROM %s  WHERE filename = "%s"''' % (self.table_name, filename))

        return cur.fetchone()

    def start_file_process(self, filename, items, status='PROCESSING'):
        self.get_sqlite_connection()
        cur = self.con.cursor()

        cur.execute('''SELECT * FROM %s WHERE filename = "%s"''' % (self.table_name, filename))

        rows = cur.fetchall()

        if len(rows) > 0:
            # update the existing row
            print(f"updating exsiting row for {filename}")

            cur.execute('''UPDATE %s SET startdatetime = "%s", status = "%s" WHERE filename = "%s"''' % (self.table_name,
                                                                                                         datetime.now(),
                                                                                                         status,
                                                                                                         filename))
            self.con.commit()
            self.con.close()
        else:
            # no rows so go ahead and insert the new filename and start the process
            print(f"inserting new row for {filename}")
            items['filename'] = filename
            items['startdatetime'] = datetime.now()
            items['status'] = status
            cur.execute('''INSERT INTO %s(%s) VALUES (%s)''' % (self.table_name,
                                                                 ", ".join(["'{}'".format(k) for k in items.keys()]),
                                                                 ", ".join(["'{}'".format(k) for k in items.values()])))
            self.con.commit()
            self.con.close()

    def finalize_file_process(self, filename, status='SUCCESS'):
        self.get_sqlite_connection()
        cur = self.con.cursor()

        cur.execute('''UPDATE %s SET enddatetime = "%s", status = "%s" WHERE filename = "%s"''' % (self.table_name,
                                                                                                     datetime.now(),
                                                                                                     status,
                                                                                                     filename))
        self.con.commit()
        self.con.close()

    def get_data(self):
        self.get_sqlite_connection()
        cur = self.con.cursor()

        cur.execute('''SELECT * FROM %s''' % (self.table_name))
        rows = cur.fetchall()
        self.con.commit()
        self.con.close()
        columns = self.columns
        columns['startdatetime'] = "text"
        columns['enddatetime'] = "text"
        columns['status'] = "text"
        return pd.DataFrame.from_records(rows, columns=columns.keys())

