import pymysql
from typing import List, Dict, Set


class MySqlInterface:
    connection: pymysql.connections.Connection
    enable_writes = True
    last_query = "NONE"

    def __init__(self, user_name: str, user_password: str, db_name: str, enable_writes: bool = True):
        # print("initializing")
        self.enable_writes = enable_writes
        self.connection = pymysql.connect(
            host='localhost', user=user_name, password=user_password, db=db_name,
            cursorclass=pymysql.cursors.DictCursor, charset='utf8mb4')
        self.connection.autocommit(True)

    def read_data(self, sql_str: str) -> List:
        self.last_query = sql_str
        with self.connection.cursor() as cursor:
            cursor.execute(sql_str)
            result = cursor.fetchall()
            return result

    def write_data(self, sql_str: str):
        self.last_query = sql_str
        if not self.enable_writes:
            return

        with self.connection.cursor() as cursor:
            cursor.execute(sql_str)

    def write_data_get_row_id(self, sql_str: str) -> int:
        self.last_query = sql_str
        if not self.enable_writes:
            return -1

        with self.connection.cursor() as cursor:
            # print(sql_str)
            cursor.execute(sql_str)
            return cursor.lastrowid

    def escape_text(self, to_escape):
        if type(to_escape) is str:
            return self.connection.escape(to_escape)

        return to_escape

    def get_last_query(self) -> str:
        return self.last_query

    def close(self):
        self.connection.close()


if __name__ == '__main__':
    msi = MySqlInterface("root", "postgres", "covid_twitter")
    sql = "select * from covid_twitter.twitter_root limit 10;"
    # sql = "select * from post_view ORDER BY post_time;"
    # sql = "select topic_id, forum_id, topic_title from phpbb_topics"
    print("{}".format(msi.read_data(sql)))
    msi.close()

'''
Create a list of all words
Create a timeline of all words
Plot that and see if any patterns present themselves
'''
